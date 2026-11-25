/**
 * Auto-generated reproduction for bug: Total ignores discount
 * Mission thread_id: t1
 * Generated: 2026-01-01 00:00:00
 *
 * Provenance:
 *   App revision: not recorded
 *   Session recorded: not recorded
 *   auth.json sha256: not recorded
 *
 * Prerequisites:
 *   npm i -D @playwright/test && npx playwright install chromium
 *   Place an authenticated storage state at ./auth.json
 * Run:
 *   npx playwright test reproduction_Total_ignores_discount_1700000000.spec.ts --headed
 * Reading a failure:
 *   SETUP_FAILURE   -> the session or starting data is not the one recorded; the bug was not reached.
 *   any other error -> the steps after '--- BUG STEPS ---' ran, so the failure is the reproduction.
 */
import { test, expect } from '@playwright/test';

test.use({ storageState: 'auth.json', actionTimeout: 5000, navigationTimeout: 15000 });

test('reproduce: Total ignores discount', async ({ page }) => {
  test.setTimeout(120_000);
  // App base URL recorded at capture time: https://app.example
  // --- SETUP CHECK (a failure here is NOT the bug) ---
  const _baseUrl = 'https://app.example';
  const _loginPath = /\/(log-?in|sign-?in|auth|sso)(\/|$)/i;
  const _recordedPaths: string[] = ["/cart"];
  try {
    await page.goto(_baseUrl, { waitUntil: 'domcontentloaded', timeout: 30_000 });
  } catch (e) {
    throw new Error('SETUP_FAILURE: cannot load ' + _baseUrl + ' - ' + (e as Error).message);
  }
  const _landedPath = new URL(page.url()).pathname;
  if (_loginPath.test(_landedPath) && !_recordedPaths.includes(_landedPath)) {
    throw new Error('SETUP_FAILURE: session expired or invalid auth.json (redirected to ' + _landedPath + ')');
  }
  // --- BUG STEPS ---
  await page.goto('https://app.example/cart');
  // observed at record time: $120.00
  const _t = await page.locator('[data-test="total"]').first().innerText();
  expect(_t).toContain('$96.00');
});
