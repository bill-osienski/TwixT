import { createServer } from 'http';
import { readFile, stat } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { exec, spawn } from 'child_process';
import { readdirSync, existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

const PORT = 5500;
const AI_PORT = 3001;

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
 * Find the latest model checkpoint in a directory.
 */
function findLatestCheckpoint(checkpointDir) {
  if (!existsSync(checkpointDir)) return null;

  const files = readdirSync(checkpointDir)
    .filter((f) => f.endsWith('.safetensors') && !f.includes('partial'))
    .sort()
    .reverse();

  return files.length > 0 ? join(checkpointDir, files[0]) : null;
}

/**
 * Export checkpoint to ONNX if needed.
 */
async function ensureOnnxModel() {
  const onnxPath = join(ROOT_DIR, 'server', 'model.onnx');
  const checkpointDir = join(ROOT_DIR, 'checkpoints', 'alphazero-v2-staged');

  // Check if ONNX exists and is recent
  const latestCheckpoint = findLatestCheckpoint(checkpointDir);
  if (!latestCheckpoint) {
    console.log('  No checkpoints found - AI server will not be available');
    return false;
  }

  let needsExport = false;

  if (!existsSync(onnxPath)) {
    console.log('  ONNX model not found, exporting...');
    needsExport = true;
  } else {
    // Check if checkpoint is newer than ONNX
    const onnxStat = await stat(onnxPath);
    const checkpointStat = await stat(latestCheckpoint);
    if (checkpointStat.mtime > onnxStat.mtime) {
      console.log('  Checkpoint newer than ONNX, re-exporting...');
      needsExport = true;
    }
  }

  if (needsExport) {
    console.log(`  Checkpoint: ${latestCheckpoint}`);

    const exportArgs = `-m scripts.GPU.alphazero.export_onnx --weights "${latestCheckpoint}" --output "${onnxPath}"`;

    return new Promise((resolve) => {
      // Try python3 first
      exec(`python3 ${exportArgs}`, { cwd: ROOT_DIR }, (error, stdout, stderr) => {
        if (!error) {
          console.log('  Export complete!');
          resolve(true);
          return;
        }

        // Fall back to python
        exec(`python ${exportArgs}`, { cwd: ROOT_DIR }, (error2, stdout2, stderr2) => {
          if (error2) {
            console.error('  Export failed:', stderr2 || stderr || error2.message);
            console.error('  Make sure Python is installed with: torch, onnx, safetensors');
            resolve(false);
          } else {
            console.log('  Export complete!');
            resolve(true);
          }
        });
      });
    });
  }

  return true;
}

/**
 * Start the AI inference server.
 */
function startAIServer() {
  const onnxPath = join(ROOT_DIR, 'server', 'model.onnx');
  if (!existsSync(onnxPath)) {
    return null;
  }

  const aiServer = spawn('node', ['server/index.js'], {
    cwd: ROOT_DIR,
    env: { ...process.env, MODEL_PATH: onnxPath, PORT: AI_PORT.toString() },
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

  // Check/export ONNX model
  console.log('Checking AI model...');
  const hasModel = await ensureOnnxModel();

  // Start AI server if model available
  let aiServer = null;
  if (hasModel) {
    console.log('\nStarting AI server...');
    aiServer = startAIServer();
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
      console.log(`   AI:        Not available (no model)`);
    }
    console.log('\n   Press Ctrl+C to stop\n');

    // Open browser after a short delay
    setTimeout(() => openBrowser(url), 500);
  });

  // Clean shutdown
  process.on('SIGINT', () => {
    console.log('\n\nShutting down...');
    if (aiServer) aiServer.kill();
    server.close();
    process.exit(0);
  });
}

main().catch(console.error);
