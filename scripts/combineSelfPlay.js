#!/usr/bin/env node
import { program } from 'commander';
import fs from 'fs/promises';
import path from 'path';

program
  .name('combineSelfPlay')
  .description('Combine multiple self-play JSON files into a single dataset')
  .option('-o, --output <file>', 'output file to write', 'selfplay-merged.json')
  .argument('<inputs...>', 'input self-play JSON files');

program.parse();
const options = program.opts();
const inputs = program.args;

if (!inputs.length) {
  console.error('Error: provide at least one input self-play JSON file.');
  process.exit(1);
}

const merged = {
  generatedAt: new Date().toISOString(),
  gameRequested: 0,
  gameCompleted: 0,
  searchDepth: null,
  games: [],
  sources: []
};

function coerceNumber(value) {
  return Number.isFinite(value) ? value : 0;
}

for (const input of inputs) {
  let data;
  try {
    const raw = await fs.readFile(input, 'utf8');
    data = JSON.parse(raw);
  } catch (err) {
    console.error(`Failed to read or parse "${input}":`, err?.message || err);
    process.exit(1);
  }

  const games = Array.isArray(data?.games) ? data.games : [];
  merged.games.push(...games);

  merged.gameRequested += coerceNumber(data?.gameRequested);
  merged.gameCompleted += coerceNumber(data?.gameCompleted || games.length);

  if (merged.searchDepth === null) {
    merged.searchDepth = data?.searchDepth ?? null;
  } else if (data?.searchDepth !== merged.searchDepth) {
    merged.searchDepth = 'mixed';
  }

  merged.sources.push({
    file: path.resolve(input),
    gameCount: games.length,
    meta: {
      generatedAt: data?.generatedAt ?? null,
      searchDepth: data?.searchDepth ?? null,
      requested: data?.gameRequested ?? null,
      completed: data?.gameCompleted ?? games.length
    }
  });
}

merged.gameCount = merged.games.length;

try {
  await fs.writeFile(
    options.output,
    JSON.stringify(merged, null, 2)
  );
  console.log(`Combined ${merged.gameCount} games into ${options.output}`);
} catch (err) {
  console.error(`Failed to write "${options.output}":`, err?.message || err);
  process.exit(1);
}
