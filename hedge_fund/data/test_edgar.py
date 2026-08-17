"""The point-in-time filter is the only thing standing between this provider
and a backtest that reads the future, so most of what follows is aimed at it.

The rest — ratio arithmetic, TTM assembly, concept fallbacks — is checked
against a synthetic companyfacts payload rather than a recorded one, because
what needs proving is the selection rule, not any particular company.
"""

from __future__ import annotations

from datetime import date

import pytest
import requests

from hedge_fund.data.edgar import (
    EdgarClient,
    EdgarError,
    _growth,
    _is_annual,
    _is_quarterly,
)

TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "TEST", "title": "Test Corp"}}


def _f(val, end, filed, start=None, form="10-Q"):
    entry = {"val": val, "end": end, "filed": filed, "form": form, "accn": "a-1"}
    if start:
        entry["start"] = start
    return entry


def _facts(**concepts) -> dict:
    return {
        "cik": 320193,
        "entityName": "Test Corp",
        "facts": {"us-gaap": {
            name: {"units": {"USD": entries}} for name, entries in concepts.items()
        }},
    }


# Four quarters of 2025, each filed about a month after its close, plus the
# 10-K. Q4 revenue is the one that moves in the restatement tests.
QUARTERS = [
    _f(100, "2025-03-31", "2025-04-30", start="2025-01-01"),
    _f(110, "2025-06-30", "2025-07-31", start="2025-04-01"),
    _f(120, "2025-09-30", "2025-10-31", start="2025-07-01"),
    _f(130, "2025-12-31", "2026-01-31", start="2025-10-01"),
]


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, facts: dict, *, status: int = 200, raises=None):
        self.facts, self.status, self.raises = facts, status, raises
        self.calls: list[str] = []

    def get(self, url, headers=None, timeout=None):
        if self.raises:
            raise self.raises
        self.calls.append(url)
        if self.status != 200:
            return FakeResponse({}, self.status)
        payload = TICKER_MAP if "company_tickers" in url else self.facts
        return FakeResponse(payload)


@pytest.fixture
def client(tmp_path):
    """A client per call, each with its own cache.

    The cache dir has to be unique: two clients built in one test sharing one
    would have the second read the first's payload off disk, and a test
    comparing their answers would be comparing one payload with itself.
    """
    built = 0

    def build(facts: dict, **kw) -> EdgarClient:
        nonlocal built
        built += 1
        session = kw.pop("session", None) or FakeSession(facts)
        api = EdgarClient(session=session, cache_dir=tmp_path / f"edgar-{built}",
                          min_interval=0.0, **kw)
        api._session_ref = session  # for assertions
        return api

    return build


class TestPointInTime:
    """The filing date, not the period end, decides what was knowable."""

    def test_a_filing_is_invisible_before_it_is_filed(self, client):
        """Q4 closed 2025-12-31 but was not filed until 2026-01-31. A query as
        of mid-January must not see it — this is the whole lookahead guard."""
        api = client(_facts(Revenues=QUARTERS))
        early = api.get_financial_metrics("TEST", "2026-01-15", period="quarterly")
        late = api.get_financial_metrics("TEST", "2026-02-15", period="quarterly")
        assert [m.report_period for m in early][0] == "2025-09-30"
        assert [m.report_period for m in late][0] == "2025-12-31"

    def test_later_filings_cannot_perturb_an_earlier_answer(self, client):
        """The property the whole module rests on: appending filings that
        postdate the query must leave that query's answer byte-identical.
        Anything that leaks — a restatement, a later period, a new concept —
        shows up here as a diff."""
        base = client(_facts(Revenues=QUARTERS[:2], NetIncomeLoss=QUARTERS[:2]))
        with_future = client(_facts(
            Revenues=QUARTERS + [
                _f(90, "2025-03-31", "2025-07-15", start="2025-01-01", form="10-K/A")],
            NetIncomeLoss=QUARTERS,
        ))
        as_of = "2025-05-31"
        assert (
            [m.model_dump() for m in
             base.get_financial_metrics("TEST", as_of, period="quarterly")]
            == [m.model_dump() for m in
                with_future.get_financial_metrics("TEST", as_of, period="quarterly")]
        )

    def test_the_restated_value_is_the_one_used_after_it_lands(self, client):
        """Same filings, but only revenue is amended: the ratio has to move."""
        revenue = QUARTERS + [
            _f(90, "2025-03-31", "2025-07-15", start="2025-01-01", form="10-K/A")
        ]
        api = client(_facts(Revenues=revenue, NetIncomeLoss=QUARTERS))
        before = api.get_financial_metrics("TEST", "2025-05-31", period="quarterly")
        after = api.get_financial_metrics("TEST", "2025-08-31", period="quarterly")
        q1 = lambda ms: [m for m in ms if m.report_period == "2025-03-31"][0]
        assert q1(before).net_margin == pytest.approx(100 / 100)
        assert q1(after).net_margin == pytest.approx(100 / 90)

    def test_the_filing_date_is_reported(self, client):
        api = client(_facts(Revenues=QUARTERS))
        latest = api.get_financial_metrics("TEST", "2026-03-01", period="quarterly")[0]
        assert latest.report_period == "2025-12-31"
        assert latest.filing_date == "2026-01-31"

    def test_nothing_filed_yet_is_an_empty_list_not_an_error(self, client):
        api = client(_facts(Revenues=QUARTERS))
        assert api.get_financial_metrics("TEST", "2024-01-01") == []


class TestTTM:
    def test_four_quarters_are_summed(self, client):
        api = client(_facts(Revenues=QUARTERS, NetIncomeLoss=QUARTERS))
        ttm = api.get_financial_metrics("TEST", "2026-03-01", period="ttm")[0]
        assert ttm.net_margin == pytest.approx(1.0)
        assert ttm.report_period == "2025-12-31"

    def test_a_short_window_is_discarded_rather_than_understated(self, client):
        """Three quarters summed is a year short by a quarter. Reporting that
        as revenue would make every margin look right and every absolute
        wrong, so it is dropped instead."""
        api = client(_facts(Revenues=QUARTERS[:3], NetIncomeLoss=QUARTERS[:3]))
        ttm = api.get_financial_metrics("TEST", "2026-03-01", period="ttm")
        assert ttm and ttm[0].net_margin is None

    def test_annual_filers_without_quarters_still_resolve(self, client):
        """20-F filers report yearly only; ttm falls back to the year."""
        annual = [_f(500, "2025-12-31", "2026-02-28", start="2025-01-01", form="10-K")]
        api = client(_facts(Revenues=annual, NetIncomeLoss=annual))
        ttm = api.get_financial_metrics("TEST", "2026-03-01", period="ttm")
        assert ttm and ttm[0].net_margin == pytest.approx(1.0)


class TestDerivation:
    @pytest.fixture
    def rich(self):
        year = dict(start="2025-01-01", form="10-K")
        return _facts(
            Revenues=[_f(1000, "2025-12-31", "2026-02-01", **year)],
            NetIncomeLoss=[_f(200, "2025-12-31", "2026-02-01", **year)],
            GrossProfit=[_f(400, "2025-12-31", "2026-02-01", **year)],
            OperatingIncomeLoss=[_f(300, "2025-12-31", "2026-02-01", **year)],
            NetCashProvidedByUsedInOperatingActivities=[
                _f(250, "2025-12-31", "2026-02-01", **year)],
            PaymentsToAcquirePropertyPlantAndEquipment=[
                _f(50, "2025-12-31", "2026-02-01", **year)],
            Assets=[_f(2000, "2025-12-31", "2026-02-01", form="10-K")],
            Liabilities=[_f(800, "2025-12-31", "2026-02-01", form="10-K")],
            StockholdersEquity=[_f(1200, "2025-12-31", "2026-02-01", form="10-K")],
            AssetsCurrent=[_f(600, "2025-12-31", "2026-02-01", form="10-K")],
            LiabilitiesCurrent=[_f(300, "2025-12-31", "2026-02-01", form="10-K")],
            InventoryNet=[_f(100, "2025-12-31", "2026-02-01", form="10-K")],
            WeightedAverageNumberOfDilutedSharesOutstanding=[
                _f(100, "2025-12-31", "2026-02-01", form="10-K")],
        )

    def test_margins_and_returns(self, client, rich):
        m = client(rich).get_financial_metrics("TEST", "2026-03-01", period="annual")[0]
        assert m.gross_margin == pytest.approx(0.4)
        assert m.operating_margin == pytest.approx(0.3)
        assert m.net_margin == pytest.approx(0.2)
        assert m.return_on_equity == pytest.approx(200 / 1200)
        assert m.return_on_assets == pytest.approx(0.1)

    def test_leverage_and_liquidity(self, client, rich):
        m = client(rich).get_financial_metrics("TEST", "2026-03-01", period="annual")[0]
        assert m.debt_to_equity == pytest.approx(800 / 1200)
        assert m.current_ratio == pytest.approx(2.0)
        assert m.quick_ratio == pytest.approx((600 - 100) / 300)

    def test_per_share_uses_free_cash_flow_net_of_capex(self, client, rich):
        m = client(rich).get_financial_metrics("TEST", "2026-03-01", period="annual")[0]
        assert m.free_cash_flow_per_share == pytest.approx((250 - 50) / 100)
        assert m.book_value_per_share == pytest.approx(12.0)

    def test_missing_concepts_leave_nulls_rather_than_failing(self, client):
        """Every ratio is nullable so a partial filer is still a legal one."""
        api = client(_facts(Revenues=QUARTERS))
        m = api.get_financial_metrics("TEST", "2026-03-01", period="quarterly")[0]
        assert m.current_ratio is None and m.debt_to_equity is None
        assert m.report_period == "2025-12-31"

    def test_the_revenue_tag_falls_back(self, client):
        """ASC 606 moved most filers onto a different revenue tag in 2018, so
        both have to resolve."""
        modern = _facts(
            RevenueFromContractWithCustomerExcludingAssessedTax=QUARTERS,
            NetIncomeLoss=QUARTERS,
        )
        m = client(modern).get_financial_metrics("TEST", "2026-03-01",
                                                 period="quarterly")[0]
        assert m.net_margin == pytest.approx(1.0)


class TestGrowth:
    def test_growth_is_period_over_period(self, client):
        api = client(_facts(Revenues=QUARTERS, NetIncomeLoss=QUARTERS))
        rows = api.get_financial_metrics("TEST", "2026-03-01", period="quarterly")
        assert rows[0].revenue_growth == pytest.approx((130 - 120) / 120)

    @pytest.mark.parametrize("before", [0, -100])
    def test_a_non_positive_base_yields_none(self, before):
        """A loss narrowing from -100 to -50 is not -50% growth, and handing a
        persona that number is worse than handing it nothing."""
        assert _growth(-50, before) is None


class TestFormClassification:
    def test_a_quarter_is_recognised_by_duration(self):
        assert _is_quarterly(_mk("2025-03-31", "2025-01-01"))

    def test_a_year_needs_the_form_as_well_as_the_duration(self):
        """A 10-Q cannot carry an annual period, and a cumulative year-to-date
        figure inside one would otherwise pass for the year."""
        assert _is_annual(_mk("2025-12-31", "2025-01-01", form="10-K"))
        assert not _is_annual(_mk("2025-12-31", "2025-01-01", form="10-Q"))


def _mk(end, start, form="10-Q"):
    from hedge_fund.data.edgar import Fact
    return Fact({"end": end, "start": start, "filed": end, "form": form, "val": 1})


class TestFetching:
    def test_facts_are_fetched_once_per_ticker(self, client):
        api = client(_facts(Revenues=QUARTERS))
        api.get_financial_metrics("TEST", "2026-03-01")
        before = len(api._session_ref.calls)
        api.get_financial_metrics("TEST", "2025-06-01")
        assert len(api._session_ref.calls) == before

    def test_an_unknown_ticker_says_so(self, client):
        api = client(_facts(Revenues=QUARTERS))
        with pytest.raises(EdgarError, match="not in the SEC ticker map"):
            api.get_financial_metrics("NOPE", "2026-03-01")

    def test_a_403_names_the_variable_to_set(self, client):
        """SEC blocks anonymous access; the fix is one env var and the error
        should say which."""
        api = client({}, session=FakeSession({}, status=403))
        with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
            api.get_financial_metrics("TEST", "2026-03-01")

    def test_a_transport_error_raises(self, client):
        api = client({}, session=FakeSession({}, raises=requests.Timeout("slow")))
        with pytest.raises(EdgarError, match="request failed"):
            api.get_financial_metrics("TEST", "2026-03-01")

    def test_shares_outstanding_respects_the_filing_date(self, client):
        shares = [
            _f(100, "2025-06-30", "2025-07-31", form="10-Q"),
            _f(150, "2025-12-31", "2026-01-31", form="10-K"),
        ]
        api = client(_facts(
            WeightedAverageNumberOfDilutedSharesOutstanding=shares,
            Revenues=QUARTERS))
        assert api.shares_outstanding("TEST", "2026-01-15") == 100
        assert api.shares_outstanding("TEST", "2026-02-15") == 150

    def test_company_facts_carry_the_cik(self, client):
        facts = client(_facts(Revenues=QUARTERS)).get_company_facts("TEST")
        assert facts is not None
        assert facts.cik == "0000320193" and facts.name == "Test Corp"
