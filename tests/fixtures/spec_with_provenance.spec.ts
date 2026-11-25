/**
 * Auto-generated reproduction for bug: Save fails
 * Mission thread_id: t1
 * Generated: 2026-01-01 00:00:00
 *
 * Provenance:
 *   App revision: abc123
 *   Session recorded: 2026-01-01T00:00:00+00:00
 *   auth.json sha256: 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a (modified 2023-11-14T22:13:20+00:00)
 *   Preconditions (starting data):
 *     - cart is empty
 *
 * Prerequisites:
 *   npm i -D @playwright/test && npx playwright install chromium
 *   Place an authenticated storage state at ./auth.json
 * Run:
 *   npx playwright test reproduction_Save_fails_1700000000.spec.ts --headed
 * Reading a failure:
 *   SETUP_FAILURE   -> the session or starting data is not the one recorded; the bug was not reached.
 *   any other error -> the steps after '--- BUG STEPS ---' ran, so the failure is the reproduction.
 */
import { test, expect } from '@playwright/test';

test.use({ storageState: 'auth.json' });

test('reproduce: Save fails', async ({ page }) => {
  test.setTimeout(120_000);
  // App base URL recorded at capture time: https://app.example
  // --- SETUP CHECK (a failure here is NOT the bug) ---
  const _baseUrl = 'https://app.example';
  const _loginPath = /\/(log-?in|sign-?in|auth|sso)(\/|$)/i;
  const _recordedPaths: string[] = ["/home"];
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
  await page.goto('https://app.example/home');
  await page.click('[data-test-subj=\'save\']');
});
