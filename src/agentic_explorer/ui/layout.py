"""``--headed --visual``: put the mission browser and the dashboard side by side.

Before this the dashboard was opened as a tab of the user's default browser, and the mission's
Chromium (maximized) opened afterwards on top of it, so only the demo app was visible. Now the
dashboard gets its own Chromium window and the two windows are tiled: the mission on the left
half of the screen, the dashboard on the right. It is independent of the default browser.

Windows are placed through the browser's own window API (CDP ``Browser.setWindowBounds``),
which works with Playwright's Chromium on every platform. It is best effort: if a window
cannot be placed the run goes on with the windows where they are.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Optional, Tuple

from agentic_explorer.ui.launcher import DASHBOARD_URL
from agentic_explorer.utils import console

_RETRY_DELAY_S = 1.0
_DASHBOARD_ATTEMPTS = 15
_SCREEN_JS = "() => ({w: screen.availWidth, h: screen.availHeight, x: screen.availLeft || 0, y: screen.availTop || 0})"


def half_screen_bounds(screen: Mapping[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(left half, right half) of the usable screen area as window bounds."""
    x, y = int(screen.get("x", 0)), int(screen.get("y", 0))
    width, height = int(screen["w"]), int(screen["h"])
    half = width // 2
    return (
        {"left": x, "top": y, "width": half, "height": height},
        {"left": x + half, "top": y, "width": width - half, "height": height},
    )


async def _place(page: Any, bounds: Mapping[str, int]) -> None:
    session = await page.context.new_cdp_session(page)
    window_id = (await session.send("Browser.getWindowForTarget"))["windowId"]
    # A maximized window ignores a resize: restore it first.
    await session.send("Browser.setWindowBounds", {"windowId": window_id, "bounds": {"windowState": "normal"}})
    await session.send("Browser.setWindowBounds", {"windowId": window_id, "bounds": dict(bounds)})


async def tile_windows(app_page: Any, dashboard_page: Any) -> bool:
    """Mission window on the left half, dashboard on the right. Returns whether it worked."""
    try:
        screen = await app_page.evaluate(_SCREEN_JS)
        if not screen or int(screen.get("w", 0)) <= 0 or int(screen.get("h", 0)) <= 0:
            return False
        left, right = half_screen_bounds(screen)
        await _place(app_page, left)
        await _place(dashboard_page, right)
        return True
    except Exception as exc:
        console.warn(f"Could not arrange the windows side by side ({type(exc).__name__}); they stay where they are.")
        return False


async def open_dashboard_window(playwright: Any, app_page: Any) -> Optional[Any]:
    """Open the dashboard in its own visible Chromium window and tile it next to ``app_page``.

    Returns the browser (the caller closes it) or ``None`` if the dashboard never answered.
    """
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context(no_viewport=True)
    page = await context.new_page()
    for attempt in range(_DASHBOARD_ATTEMPTS):
        try:
            await page.goto(DASHBOARD_URL)
            break
        except Exception:
            if attempt == _DASHBOARD_ATTEMPTS - 1:
                console.warn(f"The dashboard did not answer at {DASHBOARD_URL}; open it yourself in a browser.")
                await browser.close()
                return None
            await asyncio.sleep(_RETRY_DELAY_S)
    await tile_windows(app_page, page)
    return browser
