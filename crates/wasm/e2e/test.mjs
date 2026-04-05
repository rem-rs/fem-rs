// fem-rs WASM E2E test — run with: node e2e/test.mjs
//
// Requires: npx playwright install chromium (one-time)
// Usage:    cd crates/wasm && node e2e/test.mjs

import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmDir = join(__dirname, '..');

// Minimal static file server
const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.wasm': 'application/wasm',
  '.d.ts': 'text/plain',
};

function serve(root) {
  return createServer((req, res) => {
    const url = req.url === '/' ? '/e2e/index.html' : req.url;
    const fp = join(root, url);
    if (!existsSync(fp)) {
      res.writeHead(404); res.end('Not found'); return;
    }
    const ext = extname(fp);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(readFileSync(fp));
  });
}

async function main() {
  const server = serve(wasmDir);
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  console.log(`Server on http://127.0.0.1:${port}`);

  const browser = await chromium.launch();
  const page = await browser.newPage();

  page.on('console', msg => console.log(`[browser] ${msg.text()}`));
  page.on('pageerror', err => console.error(`[browser error] ${err}`));

  await page.goto(`http://127.0.0.1:${port}/`);

  // Wait for the test to complete (title changes to PASS or FAIL)
  await page.waitForFunction(() => document.title === 'PASS' || document.title === 'FAIL', {
    timeout: 30_000,
  });

  const results = await page.evaluate(() => window.__FEM_E2E_RESULTS);
  console.log('Results:', JSON.stringify(results, null, 2));

  await browser.close();
  server.close();

  if (results.pass) {
    console.log('\n✅ WASM E2E test PASSED');
    process.exit(0);
  } else {
    console.error('\n❌ WASM E2E test FAILED');
    process.exit(1);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
