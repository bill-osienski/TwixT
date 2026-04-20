const isBrowser =
  typeof window !== 'undefined' && typeof window.document !== 'undefined';

const DEFAULT_BROWSER_PATH = '/assets/js/ai/opening-book.json';
const NODE_CANDIDATES = [
  'assets/js/ai/opening-book.json',
  'assets/opening-book.json',
];

let _book = null;
let _loadPromise = null;

async function readTextNode(path) {
  const fs = await import('fs/promises');
  return fs.readFile(path, 'utf8');
}

async function pathExistsNode(path) {
  const fs = await import('fs/promises');
  try {
    await fs.stat(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchTextBrowser(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

async function readBookSource(pathOrUrl) {
  if (isBrowser) {
    const src = pathOrUrl || DEFAULT_BROWSER_PATH;
    const txt = await fetchTextBrowser(src);
    return JSON.parse(txt);
  }

  if (pathOrUrl) {
    const txt = await readTextNode(pathOrUrl);
    return JSON.parse(txt);
  }

  for (const candidate of NODE_CANDIDATES) {
    if (await pathExistsNode(candidate)) {
      const txt = await readTextNode(candidate);
      return JSON.parse(txt);
    }
  }

  return null;
}

export async function ensureOpeningBookLoaded(pathOrUrl) {
  if (_book) return _book;
  const payload = await readBookSource(pathOrUrl);
  _book = payload;
  return _book;
}

export function isOpeningBookLoaded() {
  return !!_book;
}

export async function maybeLoadOpeningBook(pathOrUrl = DEFAULT_BROWSER_PATH) {
  if (isOpeningBookLoaded()) return _book;
  if (_loadPromise) return _loadPromise;

  _loadPromise = ensureOpeningBookLoaded(pathOrUrl)
    .catch(() => null)
    .finally(() => {
      _loadPromise = null;
    });

  return _loadPromise;
}

export function buildOpeningKey(game) {
  const boardSize = game.boardSize || 24;
  const moveHistory = Array.isArray(game.moveHistory) ? game.moveHistory : [];
  const pegMoves = moveHistory.filter((m) => m && m.peg);

  const starting =
    game.startingPlayer ||
    (pegMoves[0] && pegMoves[0].peg && pegMoves[0].peg.player) ||
    'red';

  const parts = [`b:${boardSize}`, `s:${String(starting).toLowerCase()}`];
  for (const item of pegMoves) {
    const peg = item.peg;
    if (!peg) continue;
    parts.push(`${peg.player[0].toLowerCase()}${peg.row},${peg.col}`);
  }
  return parts.join('|');
}

export function getOpeningBookMove(game) {
  if (game && game.disableOpeningBook) return null;
  if (!_book || !_book.positions) return null;

  const key = buildOpeningKey(game);
  const entry = _book.positions[key];
  if (!entry || !Array.isArray(entry.top_k) || entry.top_k.length === 0) {
    return null;
  }

  const plies = Array.isArray(_book.plies) ? _book.plies : null;
  if (plies && typeof entry.ply === 'number') {
    const maxPly = plies.length ? Math.max(...plies) : null;
    if (maxPly !== null && entry.ply > maxPly) {
      return null;
    }
  }

  const sideToMove = String(entry.side_to_move || '').toLowerCase();
  const currentPlayer = String(game.currentPlayer || '').toLowerCase();
  if (sideToMove && currentPlayer && sideToMove !== currentPlayer) {
    return null;
  }

  const best = entry.top_k[0];
  if (!best || best.row == null || best.col == null) return null;
  return { row: best.row, col: best.col };
}

export function setOpeningBookForTests(book) {
  _book = book;
}
