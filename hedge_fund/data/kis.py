"""Daily US bars from Korea Investment & Securities.

Covers the price half of the DataClient protocol against KIS's overseas
endpoints. Fundamentals are not here and cannot be: KIS ships financial
statements for domestic listings only, so the point-in-time metrics the
personas reason over come from SEC EDGAR instead.

Endpoint choice is forced. KIS has two daily-bar APIs for overseas names and
only one of them is general: inquire-daily-chartprice takes an explicit date
range but, for US stocks, serves only Dow 30 / Nasdaq 100 / S&P 500 members.
Everything else has to go through dailyprice (HHDFS76240000), which walks
*backwards* from a reference date. Hence the paging in _walk_back().

Credentials come from KIS_APP_KEY / KIS_APP_SECRET in the environment. They
are never read from kis_devlp.yaml — that file is the sample app's, and
reading it here would couple this client to a layout we do not own.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from hedge_fund.data.models import Price
from hedge_fund.paths import USER_DIR

REAL_HOST = "https://openapi.koreainvestment.com:9443"
DEMO_HOST = "https://openapivts.koreainvestment.com:29443"

DAILY_PATH = "/uapi/overseas-price/v1/quotations/dailyprice"
TOKEN_PATH = "/oauth2/tokenP"

DAILY_TR = "HHDFS76240000"  # real and demo share this one

# A ticker does not say which venue lists it, and the API needs one. Probed in
# this order on first use, then remembered for the process.
US_EXCHANGES = ("NAS", "NYS", "AMS")

# KIS caps requests per second (20 real, 2 demo) and rejects bursts outright.
REAL_MIN_INTERVAL = 0.06
DEMO_MIN_INTERVAL = 0.55

# Tokens last 24h and KIS throttles *issuing* them to roughly one a minute, so
# a fresh token per process would rate-limit a backtest before it began.
TOKEN_CACHE = USER_DIR / "kis-token.json"
TOKEN_SKEW = timedelta(minutes=10)


class KisError(RuntimeError):
    """A KIS request failed.

    Raised rather than returning empty: under the DataClient contract an empty
    list means the data genuinely does not exist, and a transport failure that
    returned one would be indistinguishable from a quiet market.
    """


class KisPriceClient:
    """Daily OHLCV bars for US listings.

    Satisfies the price half of DataClient structurally. get_prices returns
    bars ascending by date, inclusive of both endpoints, with split and
    dividend adjustments applied.
    """

    def __init__(
        self,
        app_key: str | None = None,
        app_secret: str | None = None,
        *,
        demo: bool = False,
        host: str | None = None,
        session: requests.Session | None = None,
        token_cache: Path | None = None,
        min_interval: float | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._key = app_key or os.getenv("KIS_APP_KEY") or ""
        self._secret = app_secret or os.getenv("KIS_APP_SECRET") or ""
        self._host = host or (DEMO_HOST if demo else REAL_HOST)
        self._demo = demo
        self._session = session or requests.Session()
        self._token_cache = token_cache or TOKEN_CACHE
        self._min_interval = (
            min_interval if min_interval is not None
            else (DEMO_MIN_INTERVAL if demo else REAL_MIN_INTERVAL)
        )
        self._timeout = timeout
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._token: tuple[str, datetime] | None = None
        self._venues: dict[str, str] = {}

    # -- DataClient surface -----------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> list[Price]:
        """Daily bars for [start_date, end_date], both ISO YYYY-MM-DD."""
        start, end = _iso(start_date), _iso(end_date)
        if start > end:
            raise KisError(f"start_date {start_date} is after end_date {end_date}")

        symbol = ticker.strip().upper()
        rows = self._walk_back(symbol, start, end)

        # Deduplicated by date: pages are requested so they should not overlap,
        # but a boundary bar arriving on two of them would otherwise show up as
        # two bars on the same day, which every return series downstream reads
        # as a real move.
        seen: dict[str, Price] = {}
        for row in rows:
            bar = _to_price(row)
            if bar is not None and start <= _iso(bar.time) <= end:
                seen.setdefault(bar.time, bar)
        return [seen[t] for t in sorted(seen)]

    # -- paging ------------------------------------------------------------

    def _walk_back(self, symbol: str, start: date, end: date) -> list[dict]:
        """Pages backwards from *end* until the window is covered.

        dailyprice has no range parameter: it answers with the most recent N
        bars at or before BYMD. Each page therefore restarts one day before
        the oldest bar it just returned. The oldest-date check is what
        terminates it — a page that fails to move backwards would otherwise
        repeat forever, which is the failure mode this guards.
        """
        excd = self._exchange_for(symbol)
        collected: list[dict] = []
        cursor = end
        seen_oldest: date | None = None

        while cursor >= start:
            rows = self._daily_page(excd, symbol, cursor)
            if not rows:
                break

            dated = [(d, r) for r in rows if (d := _row_date(r)) is not None]
            if not dated:
                break

            collected.extend(r for _, r in dated)
            oldest = min(d for d, _ in dated)

            if seen_oldest is not None and oldest >= seen_oldest:
                break  # no progress; stop rather than loop
            seen_oldest = oldest

            if oldest <= start:
                break
            cursor = oldest - timedelta(days=1)

        return collected

    def _daily_page(self, excd: str, symbol: str, bymd: date) -> list[dict]:
        body = self._request(
            DAILY_PATH,
            DAILY_TR,
            {
                "AUTH": "",
                "EXCD": excd,
                "SYMB": symbol,
                "GUBN": "0",              # 0 daily, 1 weekly, 2 monthly
                "BYMD": bymd.strftime("%Y%m%d"),
                # Splits and dividends applied. Raw prices would put a split
                # discontinuity straight into every return series.
                "MODP": "1",
            },
        )
        rows = body.get("output2") or []
        return [r for r in rows if isinstance(r, dict)]

    def _exchange_for(self, symbol: str) -> str:
        """Which venue lists this symbol, probed once and remembered."""
        if symbol in self._venues:
            return self._venues[symbol]

        for excd in US_EXCHANGES:
            if self._daily_page(excd, symbol, date.today()):
                self._venues[symbol] = excd
                return excd

        raise KisError(
            f"{symbol} returned no bars on any of {', '.join(US_EXCHANGES)}. "
            "Check the symbol, or that the account is enabled for US markets."
        )

    # -- transport ---------------------------------------------------------

    def _request(self, path: str, tr_id: str, params: dict) -> dict:
        self._throttle()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token()}",
            "appkey": self._key,
            "appsecret": self._secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        try:
            resp = self._session.get(
                self._host + path, headers=headers, params=params,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise KisError(f"KIS request failed: {exc}") from exc

        if resp.status_code != 200:
            raise KisError(f"KIS returned HTTP {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        # rt_cd is the real verdict; KIS answers 200 on business errors too, so
        # trusting the status code alone would turn a rejection into a silent
        # empty result.
        if str(body.get("rt_cd", "0")) != "0":
            raise KisError(
                f"KIS rejected {tr_id}: {body.get('msg_cd')} {body.get('msg1')}")
        return body

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    # -- auth --------------------------------------------------------------

    def _access_token(self) -> str:
        if self._token and self._token[1] - TOKEN_SKEW > datetime.now():
            return self._token[0]

        cached = self._read_cached_token()
        if cached:
            self._token = cached
            return cached[0]

        token = self._issue_token()
        self._token = token
        self._write_cached_token(token)
        return token[0]

    def _issue_token(self) -> tuple[str, datetime]:
        if not self._key or not self._secret:
            raise KisError(
                "KIS_APP_KEY / KIS_APP_SECRET not set. Add them to "
                f"{USER_DIR / '.env'} or export them."
            )
        try:
            resp = self._session.post(
                self._host + TOKEN_PATH,
                json={"grant_type": "client_credentials",
                      "appkey": self._key, "appsecret": self._secret},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise KisError(f"KIS token request failed: {exc}") from exc

        if resp.status_code != 200:
            raise KisError(
                f"KIS token request returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}")

        body = resp.json()
        token = body.get("access_token")
        if not token:
            raise KisError(f"KIS token response carried no token: {body}")

        expires = _token_expiry(body)
        return token, expires

    def _read_cached_token(self) -> tuple[str, datetime] | None:
        """The token from disk, if one is there and still good."""
        try:
            data = json.loads(self._token_cache.read_text(encoding="utf-8"))
            token = data["access_token"]
            expires = datetime.fromisoformat(data["expires_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # absent or unreadable is not an error; reissue
        if expires - TOKEN_SKEW <= datetime.now():
            return None
        # A cache written against the other host is a token the current host
        # will reject, so it counts as a miss.
        if data.get("host") != self._host:
            return None
        return token, expires

    def _write_cached_token(self, token: tuple[str, datetime]) -> None:
        payload = {
            "access_token": token[0],
            "expires_at": token[1].isoformat(),
            "host": self._host,
        }
        try:
            self._token_cache.parent.mkdir(parents=True, exist_ok=True)
            self._token_cache.write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass  # a token that cannot be cached still works for this process


# -- helpers ---------------------------------------------------------------


def _iso(value: str) -> date:
    """An ISO YYYY-MM-DD string as a date."""
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise KisError(f"expected ISO YYYY-MM-DD, got {value!r}") from exc


def _row_date(row: dict) -> date | None:
    """The bar's date. KIS sends it as YYYYMMDD under `xymd`."""
    raw = str(row.get("xymd") or "").strip()
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def _to_price(row: dict) -> Price | None:
    """One bar, or None if the row is not a usable quote.

    The close arrives as `clos`, not `close` — the one field name that does
    not match its FD counterpart.
    """
    day = _row_date(row)
    if day is None:
        return None
    try:
        return Price(
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["clos"]),
            volume=int(float(row.get("tvol") or 0)),
            time=day.isoformat(),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _token_expiry(body: dict) -> datetime:
    """When the token dies.

    KIS sends `access_token_token_expired` as a wall-clock string and
    `expires_in` as seconds; the seconds are the portable one, and the
    fallback keeps a malformed field from costing us the whole run.
    """
    seconds = body.get("expires_in")
    if isinstance(seconds, (int, float)) and seconds > 0:
        return datetime.now() + timedelta(seconds=float(seconds))
    stamp = body.get("access_token_token_expired")
    if isinstance(stamp, str):
        try:
            return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=23)
