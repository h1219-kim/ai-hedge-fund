"""dailyprice has no range parameter, so the paging is ours to get right.

The fake session answers the way KIS does — the most recent N bars at or
before BYMD — because that shape is what the walk-back loop is written
against, and a fake that simply returned the whole window would prove
nothing. Nothing here touches the network or the token cache in $HOME.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest
import requests

from hedge_fund.data.kis import KisError, KisPriceClient, _to_price


def _weekdays(last: date, count: int) -> list[date]:
    """*count* weekdays ending at *last*, most recent first."""
    days, cursor = [], last
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def _bar(day: date, close: float) -> dict:
    return {
        "xymd": day.strftime("%Y%m%d"),
        "open": f"{close - 1:.2f}",
        "high": f"{close + 1:.2f}",
        "low": f"{close - 2:.2f}",
        "clos": f"{close:.2f}",
        "tvol": "1000000",
    }


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class FakeSession:
    """KIS as far as this client is concerned.

    `listed_on` decides which EXCD answers; anything else returns an empty
    output2, which is how the real API reports a symbol it does not carry.
    """

    def __init__(
        self,
        universe: dict[date, dict] | None = None,
        *,
        page_size: int = 3,
        listed_on: str = "NAS",
        rt_cd: str = "0",
        status: int = 200,
        raises: Exception | None = None,
    ) -> None:
        self.universe = universe or {}
        self.page_size = page_size
        self.listed_on = listed_on
        self.rt_cd = rt_cd
        self.status = status
        self.raises = raises
        self.gets: list[dict] = []
        self.posts = 0

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.posts += 1
        return FakeResponse({"access_token": f"tok-{self.posts}", "expires_in": 86400})

    def get(self, url, headers=None, params=None, timeout=None):
        if self.raises:
            raise self.raises
        self.gets.append({"params": dict(params or {}), "headers": dict(headers or {})})
        if self.status != 200 or self.rt_cd != "0":
            return FakeResponse(
                {"rt_cd": self.rt_cd, "msg_cd": "EGW00123", "msg1": "denied"},
                self.status,
            )
        if params.get("EXCD") != self.listed_on:
            return FakeResponse({"rt_cd": "0", "output2": []})

        bymd = datetime.strptime(params["BYMD"], "%Y%m%d").date()
        eligible = sorted((d for d in self.universe if d <= bymd), reverse=True)
        page = [self.universe[d] for d in eligible[: self.page_size]]
        return FakeResponse({"rt_cd": "0", "output2": page})


@pytest.fixture
def universe():
    """40 weekdays of bars ending 2026-03-31, closing at 100, 101, 102..."""
    days = _weekdays(date(2026, 3, 31), 40)
    return {d: _bar(d, 100 + i) for i, d in enumerate(reversed(days))}


@pytest.fixture
def client(tmp_path):
    def build(session: FakeSession) -> KisPriceClient:
        return KisPriceClient(
            "key", "secret",
            session=session,
            token_cache=tmp_path / "kis-token.json",
            min_interval=0.0,
        )

    return build


class TestPrices:
    def test_window_is_covered_across_pages(self, client, universe):
        """The window spans many pages of 3; every trading day must appear."""
        session = FakeSession(universe, page_size=3)
        bars = client(session).get_prices("AAPL", "2026-02-02", "2026-03-31")
        wanted = [d for d in universe if date(2026, 2, 2) <= d <= date(2026, 3, 31)]
        assert len(bars) == len(wanted)

    def test_bars_come_back_ascending_and_deduplicated(self, client, universe):
        session = FakeSession(universe, page_size=4)
        bars = client(session).get_prices("AAPL", "2026-02-02", "2026-03-31")
        times = [b.time for b in bars]
        assert times == sorted(times)
        assert len(times) == len(set(times))

    def test_an_overlapping_page_yields_one_bar_per_day(self, client, universe):
        """Pages are requested not to overlap, but if the server repeats a
        boundary bar anyway, a duplicate day reads downstream as a real move.
        """
        session = FakeSession(universe, page_size=5)
        session.get_original = session.get

        def overlapping(url, headers=None, params=None, timeout=None):
            resp = session.get_original(url, headers=headers, params=params,
                                        timeout=timeout)
            body = resp.json()
            if body.get("output2"):
                body["output2"] = body["output2"] + [body["output2"][-1]]
            return FakeResponse(body, resp.status_code)

        session.get = overlapping
        bars = client(session).get_prices("AAPL", "2026-02-02", "2026-03-31")
        times = [b.time for b in bars]
        assert len(times) == len(set(times))

    def test_bars_outside_the_window_are_dropped(self, client, universe):
        """A page can overshoot the start; the last one usually does."""
        session = FakeSession(universe, page_size=7)
        bars = client(session).get_prices("AAPL", "2026-03-02", "2026-03-13")
        assert all("2026-03-02" <= b.time <= "2026-03-13" for b in bars)

    def test_adjusted_prices_are_requested(self, client, universe):
        """MODP=0 would put a split discontinuity into every return series."""
        session = FakeSession(universe)
        client(session).get_prices("AAPL", "2026-03-25", "2026-03-31")
        assert all(g["params"]["MODP"] == "1" for g in session.gets)
        assert all(g["params"]["GUBN"] == "0" for g in session.gets)

    def test_start_after_end_is_rejected(self, client, universe):
        with pytest.raises(KisError, match="after end_date"):
            client(FakeSession(universe)).get_prices("AAPL", "2026-03-31", "2026-01-01")

    def test_a_symbol_with_no_bars_raises_rather_than_returning_empty(
        self, client, universe
    ):
        """Empty on every venue means the symbol is wrong, not that the market
        was quiet — the one case where empty would be a lie."""
        session = FakeSession(universe, listed_on="NOPE")
        with pytest.raises(KisError, match="no bars on any"):
            client(session).get_prices("WHAT", "2026-03-01", "2026-03-31")


class TestPaging:
    def test_a_page_that_does_not_move_backwards_stops_the_walk(self, client):
        """A server that keeps answering with the same newest bar would spin
        this loop forever; the oldest-date check is the only thing stopping it.
        """
        stuck = {date(2026, 3, 31): _bar(date(2026, 3, 31), 100)}
        session = FakeSession(stuck, page_size=1)
        bars = client(session).get_prices("AAPL", "2026-01-01", "2026-03-31")
        assert len(bars) == 1
        assert len(session.gets) < 10  # terminated, not spinning

    def test_the_venue_is_probed_once_and_remembered(self, client, universe):
        """The probe costs a call per venue; paying it per page would triple
        the request count on NYS names."""
        session = FakeSession(universe, listed_on="AMS", page_size=5)
        api = client(session)
        api.get_prices("AAPL", "2026-03-02", "2026-03-31")
        first = len(session.gets)
        api.get_prices("AAPL", "2026-03-02", "2026-03-31")
        probes = [g for g in session.gets[first:] if g["params"]["EXCD"] != "AMS"]
        assert probes == []


class TestFailures:
    def test_business_rejection_raises_even_on_http_200(self, client, universe):
        """KIS answers 200 with rt_cd != 0 on a rejection; trusting the status
        code would turn that into a silent empty result."""
        session = FakeSession(universe, rt_cd="1")
        with pytest.raises(KisError, match="EGW00123"):
            client(session).get_prices("AAPL", "2026-03-01", "2026-03-31")

    def test_http_error_raises(self, client, universe):
        session = FakeSession(universe, rt_cd="1", status=500)
        with pytest.raises(KisError, match="HTTP 500"):
            client(session).get_prices("AAPL", "2026-03-01", "2026-03-31")

    def test_transport_error_raises(self, client, universe):
        session = FakeSession(universe, raises=requests.ConnectionError("down"))
        with pytest.raises(KisError, match="request failed"):
            client(session).get_prices("AAPL", "2026-03-01", "2026-03-31")

    def test_missing_credentials_name_the_variables(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KIS_APP_KEY", raising=False)
        monkeypatch.delenv("KIS_APP_SECRET", raising=False)
        api = KisPriceClient(
            session=FakeSession(), token_cache=tmp_path / "t.json", min_interval=0.0)
        with pytest.raises(KisError, match="KIS_APP_KEY"):
            api.get_prices("AAPL", "2026-03-01", "2026-03-31")


class TestToken:
    def test_the_token_is_issued_once_per_client(self, client, universe):
        session = FakeSession(universe, page_size=5)
        api = client(session)
        api.get_prices("AAPL", "2026-03-02", "2026-03-31")
        api.get_prices("AAPL", "2026-03-02", "2026-03-31")
        assert session.posts == 1

    def test_a_cached_token_survives_a_new_client(self, tmp_path, universe):
        """KIS throttles issuing tokens to about one a minute, so a fresh one
        per process would rate-limit a backtest before it started."""
        cache = tmp_path / "kis-token.json"
        first = FakeSession(universe, page_size=40)
        KisPriceClient("k", "s", session=first, token_cache=cache,
                       min_interval=0.0).get_prices("AAPL", "2026-03-30", "2026-03-31")
        second = FakeSession(universe, page_size=40)
        KisPriceClient("k", "s", session=second, token_cache=cache,
                       min_interval=0.0).get_prices("AAPL", "2026-03-30", "2026-03-31")
        assert first.posts == 1 and second.posts == 0

    def test_a_token_cached_against_the_other_host_is_a_miss(self, tmp_path, universe):
        """A demo token is rejected by the real host and the reverse; reusing
        one across them turns into an auth error mid-run."""
        cache = tmp_path / "kis-token.json"
        cache.write_text(json.dumps({
            "access_token": "demo-token",
            "expires_at": (datetime.now() + timedelta(hours=20)).isoformat(),
            "host": "https://openapivts.koreainvestment.com:29443",
        }), encoding="utf-8")
        session = FakeSession(universe, page_size=40)
        KisPriceClient("k", "s", session=session, token_cache=cache,
                       min_interval=0.0).get_prices("AAPL", "2026-03-30", "2026-03-31")
        assert session.posts == 1

    def test_an_expired_cached_token_is_reissued(self, tmp_path, universe):
        cache = tmp_path / "kis-token.json"
        cache.write_text(json.dumps({
            "access_token": "stale",
            "expires_at": (datetime.now() - timedelta(minutes=1)).isoformat(),
            "host": "https://openapi.koreainvestment.com:9443",
        }), encoding="utf-8")
        session = FakeSession(universe, page_size=40)
        KisPriceClient("k", "s", session=session, token_cache=cache,
                       min_interval=0.0).get_prices("AAPL", "2026-03-30", "2026-03-31")
        assert session.posts == 1


class TestRowParsing:
    def test_close_is_read_from_clos(self):
        """`clos` is the one field name that does not match its FD counterpart,
        and a KeyError here would silently drop every bar."""
        bar = _to_price(_bar(date(2026, 3, 31), 123.45))
        assert bar is not None and bar.close == 123.45
        assert bar.time == "2026-03-31"

    @pytest.mark.parametrize("row", [
        {"xymd": "not-a-date", "open": "1", "high": "1", "low": "1", "clos": "1"},
        {"open": "1", "high": "1", "low": "1", "clos": "1"},
        {"xymd": "20260331", "open": "", "high": "1", "low": "1", "clos": "1"},
        {"xymd": "20260331", "open": "1", "high": "1", "low": "1"},
    ])
    def test_unusable_rows_are_skipped_not_raised(self, row):
        """A malformed row is one bad bar; raising would cost the whole run."""
        assert _to_price(row) is None

    def test_a_missing_volume_reads_as_zero(self):
        row = _bar(date(2026, 3, 31), 10.0)
        del row["tvol"]
        bar = _to_price(row)
        assert bar is not None and bar.volume == 0
