"""--headed --visual: the mission browser and the dashboard sit side by side, whatever the default browser is."""

import pytest

from agentic_explorer.ui import launcher, layout

SCREEN = {"w": 1600, "h": 900, "x": 0, "y": 25}


# --- the geometry ---------------------------------------------------------------------

def test_the_screen_is_split_in_two_halves_that_cover_it():
    left, right = layout.half_screen_bounds(SCREEN)
    assert left == {"left": 0, "top": 25, "width": 800, "height": 900}
    assert right == {"left": 800, "top": 25, "width": 800, "height": 900}


def test_an_odd_width_loses_no_pixel():
    left, right = layout.half_screen_bounds({"w": 1601, "h": 800})
    assert left["width"] + right["width"] == 1601 and right["left"] == left["width"]


def test_a_secondary_display_offset_is_respected():
    left, right = layout.half_screen_bounds({"w": 1000, "h": 700, "x": 1440, "y": 0})
    assert left["left"] == 1440 and right["left"] == 1940


# --- placing windows through the browser's own window API --------------------------------

class _Cdp:
    def __init__(self, calls, window_id):
        self.calls, self.window_id = calls, window_id

    async def send(self, method, params=None):
        self.calls.append((self.window_id, method, params))
        if method == "Browser.getWindowForTarget":
            return {"windowId": self.window_id}
        return {}


class _Context:
    def __init__(self, calls, window_id):
        self.calls, self.window_id = calls, window_id

    async def new_cdp_session(self, page):
        return _Cdp(self.calls, self.window_id)


class _Page:
    def __init__(self, calls, window_id, screen=None):
        self.context = _Context(calls, window_id)
        self._screen = screen
        self.goto_urls = []

    async def evaluate(self, js):
        return self._screen

    async def goto(self, url, **kwargs):
        self.goto_urls.append(url)


async def test_tiling_puts_the_mission_on_the_left_and_the_dashboard_on_the_right():
    calls = []
    app, dash = _Page(calls, 1, SCREEN), _Page(calls, 2)
    assert await layout.tile_windows(app, dash) is True

    bounds = {wid: p["bounds"] for wid, m, p in calls if m == "Browser.setWindowBounds" and "left" in p["bounds"]}
    assert bounds[1]["left"] == 0 and bounds[1]["width"] == 800
    assert bounds[2]["left"] == 800 and bounds[2]["width"] == 800


async def test_a_maximized_window_is_restored_before_it_is_moved():
    calls = []
    await layout.tile_windows(_Page(calls, 1, SCREEN), _Page(calls, 2))
    for wid in (1, 2):
        sets = [p["bounds"] for w, m, p in calls if w == wid and m == "Browser.setWindowBounds"]
        assert sets[0] == {"windowState": "normal"}  # the browser ignores a resize while maximized
        assert "left" in sets[1]


async def test_a_browser_that_cannot_place_windows_is_not_an_error():
    class Broken(_Page):
        async def evaluate(self, js):
            raise RuntimeError("no such API")

    assert await layout.tile_windows(Broken([], 1), _Page([], 2)) is False


async def test_an_unreadable_screen_size_falls_back_without_failing():
    calls = []
    assert await layout.tile_windows(_Page(calls, 1, {"w": 0, "h": 0}), _Page(calls, 2)) is False
    assert not [c for c in calls if c[1] == "Browser.setWindowBounds"]


# --- opening the dashboard window ---------------------------------------------------------------

async def _coro(value):
    return value


def _browser_with(page_factory, calls):
    class Browser:
        closed = False

        async def new_context(self, **kwargs):
            page = page_factory(calls)

            class Ctx:
                async def new_page(_self):
                    return page

            self.page = page
            return Ctx()

        async def close(self):
            self.closed = True

    return Browser()


class _Playwright:
    def __init__(self, browser):
        self.launched = []
        outer = self

        class Chromium:
            async def launch(_c, **kwargs):
                outer.launched.append(kwargs)
                return browser

        self.chromium = Chromium()


async def test_the_dashboard_gets_its_own_visible_window_and_is_tiled():
    calls = []
    browser = _browser_with(lambda c: _Page(c, 2), calls)
    pw = _Playwright(browser)
    result = await layout.open_dashboard_window(pw, _Page(calls, 1, SCREEN))

    assert result is browser and pw.launched[0]["headless"] is False
    assert browser.page.goto_urls == [launcher.DASHBOARD_URL]
    assert [c for c in calls if c[1] == "Browser.setWindowBounds"], "both windows were placed"


async def test_a_dashboard_that_is_not_up_yet_is_retried(monkeypatch):
    attempts = []

    class Flaky(_Page):
        async def goto(self, url, **kwargs):
            attempts.append(url)
            if len(attempts) < 3:
                raise RuntimeError("net::ERR_CONNECTION_REFUSED")

    browser = _browser_with(lambda c: Flaky(c, 2), [])
    monkeypatch.setattr(layout, "_RETRY_DELAY_S", 0)
    assert await layout.open_dashboard_window(_Playwright(browser), _Page([], 1, SCREEN)) is browser
    assert len(attempts) == 3


async def test_a_dashboard_that_never_comes_up_returns_none_and_closes_its_window(monkeypatch):
    class Never(_Page):
        async def goto(self, url, **kwargs):
            raise RuntimeError("refused")

    browser = _browser_with(lambda c: Never(c, 2), [])
    monkeypatch.setattr(layout, "_RETRY_DELAY_S", 0)
    assert await layout.open_dashboard_window(_Playwright(browser), _Page([], 1, SCREEN)) is None
    assert browser.closed is True


# --- the launcher no longer has to open the user's browser --------------------------------------------

@pytest.fixture
def launched(monkeypatch):
    rec = {"urls": [], "sleeps": []}

    class Emitter:
        def cleanup(self):
            pass

        def enable(self):
            pass

        def update(self, **kw):
            pass

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(launcher, "state_emitter", Emitter())
    monkeypatch.setattr(launcher.atexit, "register", lambda *a: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda s: rec["sleeps"].append(s))
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: rec["urls"].append(url))
    return rec


def test_by_default_the_launcher_still_opens_the_default_browser(launched):
    launcher.start_dashboard(app_url="u", provider="p", model_name="m")
    assert launched["urls"] == [launcher.DASHBOARD_URL]


def test_the_launcher_can_leave_the_window_to_the_caller(launched):
    launcher.start_dashboard(app_url="u", provider="p", model_name="m", open_browser=False)
    assert launched["urls"] == []
    assert launched["sleeps"] == [2]  # the server still gets its moment to start
