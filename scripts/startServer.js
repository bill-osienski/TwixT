import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');

const PORT = 5500;

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

server.listen(PORT, () => {
  const url = `http://localhost:${PORT}`;
  console.log(`\n🎮 TwixT Game Server Running!`);
  console.log(`\n   Local:    ${url}`);
  console.log(`   Network:  http://127.0.0.1:${PORT}`);
  console.log(`\n   Opening browser...\n`);
  console.log(`   Press Ctrl+C to stop the server\n`);

  // Open browser after a short delay to ensure server is ready
  setTimeout(() => openBrowser(url), 500);
});
