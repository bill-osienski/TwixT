import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { exec, spawn } from 'child_process';

import { resolveModel } from '../server/model_manifest.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

// Overridable so the launcher itself can be exercised by tests: a regression
// test needs to start it without racing for the real ports or opening a browser
// on the developer's machine. `0` selects an ephemeral port, so an explicit 0
// must survive the fallback.
const envPort = (name, fallback) =>
  process.env[name] === undefined ? fallback : Number(process.env[name]);

const PORT = envPort('TWIXT_PORT', 5500);
const AI_PORT = envPort('TWIXT_AI_PORT', 3001);

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

/**
 * Verify the pinned model before anything starts.
 *
 * This replaces a step that re-derived the served artifact on every launch by
 * picking whatever file sorted last in a research checkpoint directory and
 * re-exporting it whenever its mtime beat the ONNX's. Nothing here writes,
 * exports, or selects a model: it reads one manifest and checks that the bytes
 * on disk are the bytes that manifest names.
 *
 * On failure it reports loudly and declines to start the AI server. It never
 * substitutes another artifact and never regenerates one.
 */
async function checkPinnedModel() {
  try {
    const { manifest, manifestPath } = await resolveModel();
    console.log(`  Model id: ${manifest.model_id}`);
    console.log(`  Manifest: ${manifestPath}`);
    return { ok: true, manifestPath };
  } catch (err) {
    console.error('');
    console.error('  ❌ MODEL VALIDATION FAILED');
    console.error(`     ${err.code || 'ERROR'}: ${err.message}`);
    console.error('     The AI server will NOT start.');
    console.error('     No model was exported, substituted, or regenerated.');
    console.error('');
    return { ok: false, manifestPath: null };
  }
}

/**
 * Start the AI inference server against the manifest this launcher validated.
 *
 * The child is given MODEL_MANIFEST, not a raw artifact path, so it repeats the
 * same validation rather than trusting the parent's word for it.
 */
function startAIServer(manifestPath) {
  const aiServer = spawn('node', ['server/index.js'], {
    cwd: ROOT_DIR,
    env: { ...process.env, MODEL_MANIFEST: manifestPath, PORT: AI_PORT.toString() },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  aiServer.stdout.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    lines.forEach((line) => console.log(`  [AI] ${line}`));
  });

  aiServer.stderr.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    lines.forEach((line) => console.error(`  [AI] ${line}`));
  });

  aiServer.on('error', (err) => {
    console.error('  [AI] Failed to start:', err.message);
  });

  return aiServer;
}

// Static file server
const server = createServer(async (req, res) => {
  try {
    // Strip query parameters (e.g., ?v=dev-003)
    let urlPath = req.url.split('?')[0];

    // Default to TwixT.html for root
    let filePath = urlPath === '/' ? '/TwixT.html' : urlPath;
    filePath = join(ROOT_DIR, filePath);

    const ext = extname(filePath);
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    const data = await readFile(filePath);
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.error('File not found:', req.url);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('404 - File Not Found');
    } else {
      console.error('Server error:', err);
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('500 - Internal Server Error');
    }
  }
});

// Open browser automatically
function openBrowser(url) {
  if (process.env.TWIXT_NO_BROWSER) return;
  const start =
    process.platform === 'darwin'
      ? 'open'
      : process.platform === 'win32'
        ? 'start'
        : 'xdg-open';

  exec(`${start} ${url}`);
}

// Main startup
async function main() {
  console.log('\n🎮 TwixT Game Server Starting...\n');

  // Registered before anything is spawned. Previously this was installed only
  // after the static server was listening, so a Ctrl+C in the window between
  // spawning the AI server and listening left the child running.
  let aiServer = null;
  const shutdown = () => {
    console.log('\n\nShutting down...');
    if (aiServer) aiServer.kill();
    server.close();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  // Validate the pinned model. No export, no selection, no fallback.
  console.log('Checking AI model...');
  const { ok: hasModel, manifestPath } = await checkPinnedModel();

  // Start AI server only if the pinned artifact validated.
  if (hasModel) {
    console.log('\nStarting AI server...');
    aiServer = startAIServer(manifestPath);
  }

  // Start static file server
  server.listen(PORT, () => {
    const url = `http://localhost:${PORT}`;
    console.log('\n✅ Servers Running!\n');
    console.log(`   Game:      ${url}`);
    if (hasModel) {
      console.log(`   AI:        http://localhost:${AI_PORT}`);
      console.log(`   WebSocket: ws://localhost:${AI_PORT}/ws`);
    } else {
      console.log(`   AI:        UNAVAILABLE - model validation failed (see error above)`);
    }
    console.log('\n   Press Ctrl+C to stop\n');

    // Open browser after a short delay
    setTimeout(() => openBrowser(url), 500).unref();
  });
}

main().catch(console.error);
