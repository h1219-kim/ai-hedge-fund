"""Point-in-time fundamentals from SEC EDGAR.

The XBRL companyfacts endpoint is free, official, and — the part that matters
here — every fact carries the accession number and the date it was *filed*.
When a later filing restates a figure the earlier value stays put with its own
filing date, so "what was knowable on 2023-06-30" is reconstructable rather
than approximated. That is the whole reason this module exists instead of a
convenience API: nearly every free price/fundamentals feed serves restated
figures with no filing date, and a backtest built on those is reading the
future.

Scope: the point-in-time machinery and the ratios derivable from the standard
us-gaap concepts. Valuation ratios that need a price (P/E, P/B, market cap)
are left null here and composed downstream, where a price source exists.
FinancialMetrics makes every ratio nullable precisely so a partial provider is
a legal one.

SEC asks for a descriptive User-Agent with a contact address and rate-limits
to 10 requests a second. Set SEC_USER_AGENT.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from hedge_fund.data.models import CompanyFacts, FinancialMetrics
from hedge_fund.paths import CACHE_DIR

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_USER_AGENT = "ai-hedge-fund (github.com/virattt/ai-hedge-fund)"

# SEC's published ceiling is 10 requests a second; sustained excess gets the
# address blocked rather than throttled.
MIN_INTERVAL = 0.11

TICKER_MAP_TTL = timedelta(days=7)
FACTS_TTL = timedelta(days=1)

# Companies tag the same line differently, and the tag changes over a
# company's life (ASC 606 moved most revenue reporting in 2018). Each tuple is
# tried in order and the first one with usable facts wins.
REVENUE = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
NET_INCOME = ("NetIncomeLoss", "ProfitLoss")
GROSS_PROFIT = ("GrossProfit",)
OPERATING_INCOME = ("OperatingIncomeLoss",)
OPERATING_CASH_FLOW = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
INTEREST_EXPENSE = ("InterestExpense", "InterestExpenseNonoperatingNet")
DIVIDENDS_PAID = ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends")
EPS_DILUTED = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")

ASSETS = ("Assets",)
LIABILITIES = ("Liabilities",)
EQUITY = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
ASSETS_CURRENT = ("AssetsCurrent",)
LIABILITIES_CURRENT = ("LiabilitiesCurrent",)
CASH = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)
INVENTORY = ("InventoryNet",)
RECEIVABLES = ("AccountsReceivableNetCurrent",)
SHARES_OUTSTANDING = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "CommonStockSharesOutstanding",
)

ANNUAL_FORMS = ("10-K", "20-F", "40-F")
QUARTERLY_FORMS = ("10-Q",)


class EdgarError(RuntimeError):
    """An EDGAR request failed.

    Raised rather than returning empty: under the DataClient contract empty
    means the filing genuinely does not exist, and a 403 that returned one
    would poison a backtest with silent gaps.
    """


class Fact:
    """One reported value, with the date it became public.

    `end` is the period end (the balance-sheet date for an instant, the
    period close for a duration). `filed` is when the filing carrying it hit
    EDGAR — the only date a point-in-time query may compare against.
    """

    __slots__ = ("start", "end", "filed", "val", "form", "accn", "fy", "fp")

    def __init__(self, raw: dict) -> None:
        self.start = _date(raw.get("start"))
        self.end = _date(raw.get("end"))
        self.filed = _date(raw.get("filed"))
        self.val = raw.get("val")
        self.form = raw.get("form") or ""
        self.accn = raw.get("accn") or ""
        self.fy = raw.get("fy")
        self.fp = raw.get("fp")

    @property
    def days(self) -> int | None:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).days

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Fact({self.form} {self.end} filed={self.filed} val={self.val})"


class EdgarClient:
    """Fundamentals from EDGAR, filtered to what was public on a given date."""

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        session: requests.Session | None = None,
        cache_dir: Path | None = None,
        min_interval: float = MIN_INTERVAL,
        timeout: float = 30.0,
    ) -> None:
        self._agent = user_agent or os.getenv("SEC_USER_AGENT") or DEFAULT_USER_AGENT
        self._session = session or requests.Session()
        self._cache = cache_dir or (CACHE_DIR / "edgar")
        self._min_interval = min_interval
        self._timeout = timeout
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._ciks: dict[str, int] | None = None
        self._facts: dict[str, dict] = {}

    # -- DataClient surface -----------------------------------------------

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        cik = self._cik_for(ticker)
        if cik is None:
            return None
        facts = self._company_facts(ticker)
        return CompanyFacts(
            ticker=ticker.upper(),
            name=facts.get("entityName"),
            cik=f"{cik:010d}",
            sec_filings_url=(
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik:010d}&type=10-K"
            ),
        )

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Metrics as they stood on *end_date*, newest period first.

        Only filings public by end_date are considered, and where a period was
        later restated the version filed by that date is the one used.
        """
        as_of = _iso(end_date)
        series = self._series(ticker, as_of)
        if not series:
            return []

        periods = _period_ends(series, period, as_of)
        out: list[FinancialMetrics] = []
        for i, end in enumerate(periods[:limit]):
            prior = periods[i + 1] if i + 1 < len(periods) else None
            out.append(_metrics_for(ticker.upper(), series, end, prior, period))
        return out

    def shares_outstanding(self, ticker: str, end_date: str) -> float | None:
        """Diluted share count public on *end_date*.

        Exposed on its own because market cap is shares times a price, and the
        price lives in another provider entirely.
        """
        series = self._series(ticker, _iso(end_date))
        return _latest(series, SHARES_OUTSTANDING, _iso(end_date))

    # -- the point-in-time core -------------------------------------------

    def _series(self, ticker: str, as_of: date) -> dict[str, list[Fact]]:
        """Every us-gaap/dei concept as facts public on *as_of*.

        Two filters, and both matter. Facts filed after as_of are dropped —
        that is the lookahead guard. Then, per concept and period end, only
        the latest surviving filing is kept: a restatement filed before as_of
        was public knowledge, an original that it superseded was not.
        """
        raw = self._company_facts(ticker)
        series: dict[str, list[Fact]] = {}

        for taxonomy in ("us-gaap", "dei"):
            for concept, body in (raw.get("facts", {}).get(taxonomy) or {}).items():
                best: dict[tuple, Fact] = {}
                for unit_facts in (body.get("units") or {}).values():
                    for entry in unit_facts:
                        fact = Fact(entry)
                        if fact.filed is None or fact.filed > as_of:
                            continue
                        if fact.end is None or fact.val is None:
                            continue
                        key = (fact.start, fact.end)
                        prior = best.get(key)
                        if prior is None or fact.filed > prior.filed:
                            best[key] = fact
                if best:
                    series.setdefault(concept, []).extend(best.values())

        for facts in series.values():
            facts.sort(key=lambda f: (f.end, f.filed))
        return series

    # -- fetch + cache -----------------------------------------------------

    def _cik_for(self, ticker: str) -> int | None:
        if self._ciks is None:
            payload = self._cached_json(
                "company_tickers.json", TICKERS_URL, TICKER_MAP_TTL)
            rows = payload.values() if isinstance(payload, dict) else payload
            self._ciks = {
                str(r["ticker"]).upper(): int(r["cik_str"])
                for r in rows
                if isinstance(r, dict) and r.get("ticker") and r.get("cik_str")
            }
        return self._ciks.get(ticker.strip().upper())

    def _company_facts(self, ticker: str) -> dict:
        key = ticker.strip().upper()
        if key in self._facts:
            return self._facts[key]
        cik = self._cik_for(key)
        if cik is None:
            raise EdgarError(
                f"{key} is not in the SEC ticker map. US listings only — "
                "ADRs and foreign private issuers may file under another name."
            )
        payload = self._cached_json(
            f"facts-{cik:010d}.json", FACTS_URL.format(cik=cik), FACTS_TTL)
        self._facts[key] = payload
        return payload

    def _cached_json(self, name: str, url: str, ttl: timedelta) -> dict:
        path = self._cache / name
        try:
            age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            if age < ttl:
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # absent, stale, or corrupt all mean "fetch it"

        payload = self._get(url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass  # an uncacheable response is still a usable one
        return payload

    def _get(self, url: str) -> dict:
        self._throttle()
        try:
            resp = self._session.get(
                url,
                headers={"User-Agent": self._agent,
                         "Accept-Encoding": "gzip, deflate"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise EdgarError(f"EDGAR request failed: {exc}") from exc

        if resp.status_code == 403:
            raise EdgarError(
                "EDGAR returned 403. SEC requires a descriptive User-Agent "
                "with a contact address — set SEC_USER_AGENT."
            )
        if resp.status_code != 200:
            raise EdgarError(f"EDGAR returned HTTP {resp.status_code} for {url}")
        return resp.json()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# -- period selection ------------------------------------------------------


def _period_ends(
    series: dict[str, list[Fact]], period: str, as_of: date
) -> list[date]:
    """Reporting period ends available on *as_of*, newest first.

    Anchored on revenue because it is the one line every filer reports and the
    one whose duration says which cadence a period belongs to.
    """
    facts = _concept(series, REVENUE)
    if not facts:
        facts = _concept(series, NET_INCOME)
    wanted = _is_annual if period == "annual" else _is_quarterly
    ends = {f.end for f in facts if f.end <= as_of and wanted(f)}
    if period == "ttm" and not ends:
        # A filer with no clean quarters (many 20-F filers) still has years.
        ends = {f.end for f in facts if f.end <= as_of and _is_annual(f)}
    return sorted(ends, reverse=True)


def _is_annual(fact: Fact) -> bool:
    days = fact.days
    return fact.form in ANNUAL_FORMS and days is not None and 330 <= days <= 400


def _is_quarterly(fact: Fact) -> bool:
    days = fact.days
    return days is not None and 80 <= days <= 100


# -- metric derivation -----------------------------------------------------


def _metrics_for(
    ticker: str,
    series: dict[str, list[Fact]],
    end: date,
    prior: date | None,
    period: str,
) -> FinancialMetrics:
    """One period's ratios.

    Flows are trailing-twelve-month when period is "ttm" — four quarters
    summed, or the annual figure when the quarters are not all there. Stocks
    (balance-sheet lines) are always the instant at the period end; averaging
    them would smear a restatement across periods it never applied to.
    """
    flow = (lambda names: _ttm(series, names, end)) if period == "ttm" else (
        lambda names: _at(series, names, end))

    revenue = flow(REVENUE)
    net_income = flow(NET_INCOME)
    gross_profit = flow(GROSS_PROFIT)
    operating_income = flow(OPERATING_INCOME)
    ocf = flow(OPERATING_CASH_FLOW)
    capex = flow(CAPEX)
    interest = flow(INTEREST_EXPENSE)
    dividends = flow(DIVIDENDS_PAID)

    assets = _instant(series, ASSETS, end)
    liabilities = _instant(series, LIABILITIES, end)
    equity = _instant(series, EQUITY, end)
    assets_cur = _instant(series, ASSETS_CURRENT, end)
    liabs_cur = _instant(series, LIABILITIES_CURRENT, end)
    cash = _instant(series, CASH, end)
    inventory = _instant(series, INVENTORY, end)
    receivables = _instant(series, RECEIVABLES, end)
    shares = _latest(series, SHARES_OUTSTANDING, end)

    fcf = _sub(ocf, capex)
    filed = _filed_on(series, end)

    prior_revenue = prior_income = prior_equity = prior_fcf = prior_op = None
    if prior is not None:
        prior_flow = (lambda names: _ttm(series, names, prior)) if period == "ttm" \
            else (lambda names: _at(series, names, prior))
        prior_revenue = prior_flow(REVENUE)
        prior_income = prior_flow(NET_INCOME)
        prior_op = prior_flow(OPERATING_INCOME)
        prior_equity = _instant(series, EQUITY, prior)
        prior_fcf = _sub(prior_flow(OPERATING_CASH_FLOW), prior_flow(CAPEX))

    return FinancialMetrics(
        ticker=ticker,
        report_period=end.isoformat(),
        period=period,
        currency="USD",
        filing_date=filed.isoformat() if filed else None,

        gross_margin=_div(gross_profit, revenue),
        operating_margin=_div(operating_income, revenue),
        net_margin=_div(net_income, revenue),
        return_on_equity=_div(net_income, equity),
        return_on_assets=_div(net_income, assets),

        asset_turnover=_div(revenue, assets),
        inventory_turnover=_div(_sub(revenue, gross_profit), inventory),
        receivables_turnover=_div(revenue, receivables),

        current_ratio=_div(assets_cur, liabs_cur),
        quick_ratio=_div(_sub(assets_cur, inventory), liabs_cur),
        cash_ratio=_div(cash, liabs_cur),
        operating_cash_flow_ratio=_div(ocf, liabs_cur),

        debt_to_equity=_div(liabilities, equity),
        debt_to_assets=_div(liabilities, assets),
        interest_coverage=_div(operating_income, interest),

        revenue_growth=_growth(revenue, prior_revenue),
        earnings_growth=_growth(net_income, prior_income),
        book_value_growth=_growth(equity, prior_equity),
        free_cash_flow_growth=_growth(fcf, prior_fcf),
        operating_income_growth=_growth(operating_income, prior_op),

        payout_ratio=_div(_abs(dividends), net_income),
        earnings_per_share=_at(series, EPS_DILUTED, end) or _div(net_income, shares),
        book_value_per_share=_div(equity, shares),
        free_cash_flow_per_share=_div(fcf, shares),
    )


def _concept(series: dict[str, list[Fact]], names: tuple[str, ...]) -> list[Fact]:
    """Facts for the first tagged concept that has any."""
    for name in names:
        facts = series.get(name)
        if facts:
            return facts
    return []


def _at(series, names: tuple[str, ...], end: date) -> float | None:
    """The value reported for the period ending exactly *end*."""
    for fact in reversed(_concept(series, names)):
        if fact.end == end:
            return _float(fact.val)
    return None


def _instant(series, names: tuple[str, ...], end: date) -> float | None:
    """The balance-sheet value at *end*, or the closest one before it.

    Balance-sheet dates drift from income-statement period ends by a few days
    for 52/53-week filers, so an exact match alone would drop them.
    """
    best = None
    for fact in _concept(series, names):
        if fact.start is None and fact.end <= end:
            if best is None or fact.end > best.end:
                best = fact
    return _float(best.val) if best else None


def _latest(series, names: tuple[str, ...], end: date) -> float | None:
    """The most recent value of any shape at or before *end*."""
    best = None
    for fact in _concept(series, names):
        if fact.end <= end and (best is None or fact.end > best.end):
            best = fact
    return _float(best.val) if best else None


def _ttm(series, names: tuple[str, ...], end: date) -> float | None:
    """Trailing twelve months ending at *end*.

    Four quarters when they are all present, else the annual figure covering
    the same close. Three quarters summed would understate by a quarter, so a
    short run is discarded rather than returned.
    """
    facts = _concept(series, names)
    quarters = [f for f in facts if _is_quarterly(f) and f.end <= end]
    window = [f for f in quarters if f.end > end - timedelta(days=366)]
    if len(window) >= 4:
        chosen = sorted(window, key=lambda f: f.end, reverse=True)[:4]
        return sum(_float(f.val) or 0.0 for f in chosen)

    for fact in reversed(facts):
        if fact.end == end and _is_annual(fact):
            return _float(fact.val)
    return None


def _filed_on(series: dict[str, list[Fact]], end: date) -> date | None:
    """When the period ending *end* was first reported."""
    for names in (REVENUE, NET_INCOME, ASSETS):
        for fact in _concept(series, names):
            if fact.end == end and fact.filed:
                return fact.filed
    return None


# -- arithmetic that tolerates missing data --------------------------------


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _sub(a: float | None, b: float | None) -> float | None:
    if a is None:
        return None
    return a - (b or 0.0)


def _abs(value: float | None) -> float | None:
    return None if value is None else abs(value)


def _growth(now: float | None, before: float | None) -> float | None:
    """Period-over-period change.

    None when the base is zero or negative: a loss narrowing from -100 to -50
    is not -50% growth, and feeding that number to a persona is worse than
    telling it nothing.
    """
    if now is None or before is None or before <= 0:
        return None
    return (now - before) / before


def _date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _iso(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError) as exc:
        raise EdgarError(f"expected ISO YYYY-MM-DD, got {value!r}") from exc
