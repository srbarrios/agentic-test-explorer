import asyncio
import os
from playwright.async_api import async_playwright
from dotenv import load_dotenv

from agentic_explorer.config import load_app_config

load_dotenv()


async def save_auth_state():
    cfg = load_app_config()

    app_url = cfg.app.url or os.getenv("APP_URL")
    username = os.getenv("APP_USERNAME")
    password = os.getenv("APP_PASSWORD")

    if not all([app_url, username, password]):
        raise ValueError(
            "Missing application credentials. Set APP_URL, APP_USERNAME and APP_PASSWORD "
            "in your .env file (or supply app.url via config.yaml)."
        )

    selectors = cfg.auth.selectors or {}
    username_selector = selectors.get("username")
    password_selector = selectors.get("password")
    submit_selector = selectors.get("submit")

    if not all([username_selector, password_selector, submit_selector]):
        raise ValueError(
            "Missing auth selectors. Define auth.selectors.username / password / submit "
            "in your config.yaml."
        )

    async with async_playwright() as p:
        # Launch non-headless so you can see it working or handle unexpected prompts
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True, ignore_https_errors=True)
        page = await context.new_page()

        print(f"Navigating to {app_url} ...")
        await page.goto(app_url)

        # Wait for the login form to appear
        print("Logging in...")
        await page.wait_for_selector(username_selector)

        await page.fill(username_selector, username)
        await page.fill(password_selector, password)
        await page.click(submit_selector)

        if cfg.auth.post_login_check:
            print("Waiting for post-login element to confirm authentication...")
            await page.wait_for_selector(cfg.auth.post_login_check, timeout=15000)
        else:
            print("No auth.post_login_check configured; sleeping briefly to let session settle...")
            await page.wait_for_load_state("networkidle", timeout=15000)

        # Save the authentication state
        await context.storage_state(path="auth.json")
        print("Authentication state saved successfully to auth.json!")

        await browser.close()


def main():
    asyncio.run(save_auth_state())


if __name__ == "__main__":
    main()
