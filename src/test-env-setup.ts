// Runs before any test module imports. Sets the env vars that config.ts
// reads at import time so contract tests can build a working dashboard
// app without polluting the developer's real .env or DB.
import fs from 'fs';
import os from 'os';
import path from 'path';

process.env.DASHBOARD_TOKEN = process.env.DASHBOARD_TOKEN || 'test-contract-token';
process.env.DASHBOARD_MUTATIONS_ENABLED = process.env.DASHBOARD_MUTATIONS_ENABLED || 'true';
process.env.WARROOM_ENABLED = process.env.WARROOM_ENABLED || 'false';
// Pinned for the CSRF allowlist regression — the contract test issues
// a POST with Origin=https://dash.test.example and asserts the
// middleware lets it through. Without this, the CSRF check has no
// allowed-origin host and 403s every cross-origin POST.
process.env.DASHBOARD_URL = process.env.DASHBOARD_URL || 'https://dash.test.example';

// Sandbox the agent config dir for tests so contract tests that write
// agent.yaml fixtures (display-name resolution) don't collide with a
// developer's real ~/.claudeclaw. config.ts reads CLAUDECLAW_CONFIG at
// import time, so it must be set here, in a setupFile, BEFORE any module
// that imports config.ts is loaded.
const TEST_CLAUDECLAW_CONFIG = fs.mkdtempSync(path.join(os.tmpdir(), 'cc-test-config-'));
process.env.CLAUDECLAW_CONFIG = TEST_CLAUDECLAW_CONFIG;
// Tests that exercise loadAgentConfig('main') need a token to be present.
process.env.TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || 'test-bot-token';
