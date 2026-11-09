#!/usr/bin/env node
/**
 * The frozen eager-standard comparison, for a subject corpus captured under a NAMED later stage.
 *
 *   node tests/mcts_golden/compare_stage.mjs <corpus_dir> --stage <name>
 *
 * Why this exists: compare.mjs was built for one subject, the `lazy` corpus, and hard-codes LAZY_STAGE as the
 * subject's required stage. It is frozen evidence tooling and is NOT edited. This driver reuses its exported pieces
 * unchanged:
 *   - the frozen eager standard (EAGER_CORPUS) and its fingerprint check (corpusFingerprint);
 *   - the per-case comparison (compareArtifacts) on COMPARED_FIELDS, excluding EXCLUDED_FIELDS;
 *   - the same corpus validation (validateCorpus).
 * The one difference: the subject's stage is supplied, it must be a known stage other than `eager`, and its corpus
 * must validate under that stage (schema, stage field, surface digest and a uniform capture commit).
 *
 * Exit codes as compare.mjs: 0 MATCH, 1 MISMATCH, 2 usage, 3 corpus invalid, 4 error.
 * The interpretation is compare.mjs's, fixed before any lazy trace existed: a MATCH is exact agreement on those
 * fields across these 92 cases ONLY.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import {
  CaptureError,
  EXPECTED_CASE_COUNT,
  REPO_ROOT,
  artifactName,
  codeOfThrown,
  describeThrown,
  enumerateCases,
  stageConfig,
  validateCorpus,
} from './cases.mjs';
import {
  COMPARED_FIELDS,
  EAGER_CORPUS,
  EXCLUDED_FIELDS,
  EXIT_CORPUS_INVALID,
  EXIT_ERROR,
  EXIT_MATCH,
  EXIT_MISMATCH,
  EXIT_USAGE,
  compareArtifacts,
  corpusFingerprint,
} from './compare.mjs';
import { deriveExpectedFixtures } from './worker.mjs';

export function compareStageCorpus({ dir, stage }) {
  if (stage === EAGER_CORPUS.stage) throw new CaptureError('USAGE', 'the subject cannot be the eager standard');
  stageConfig(stage); // refuses an unknown stage
  const eagerDir = join(REPO_ROOT, EAGER_CORPUS.relDir);
  const expectedFixtures = deriveExpectedFixtures();
  const eagerFingerprint = corpusFingerprint(eagerDir);
  if (eagerFingerprint !== EAGER_CORPUS.fingerprint) {
    throw new CaptureError('EAGER_FINGERPRINT_MISMATCH', `eager corpus fingerprint is ${eagerFingerprint}`);
  }
  const eagerFailures = validateCorpus(eagerDir, {
    mode: 'capture',
    stage: EAGER_CORPUS.stage,
    expectedCaptureCommit: EAGER_CORPUS.captureCommit,
    expectedFixtures,
    readdirSync,
    readFileSync,
  });
  if (eagerFailures.length) throw new CaptureError('EAGER_CORPUS_INVALID', `${eagerFailures.length} failure(s)`);

  const names = [...readdirSync(dir)];
  const first = enumerateCases()[0];
  const commit = JSON.parse(readFileSync(join(dir, artifactName(first.caseId)), 'utf8')).capture_commit;
  if (typeof commit !== 'string' || !/^[0-9a-f]{40}$/.test(commit)) {
    throw new CaptureError('LAZY_COMMIT_NOT_UNIFORM', `subject corpus carries no usable capture_commit: ${commit}`);
  }
  const failures = validateCorpus(dir, {
    mode: 'capture',
    stage,
    expectedCaptureCommit: commit,
    expectedFixtures,
    readdirSync,
    readFileSync,
  });
  if (failures.length) {
    throw new CaptureError('LAZY_CORPUS_INVALID', `${failures.length} failure(s): ${failures.map((f) => f.code).join(', ')}`);
  }

  const mismatches = [];
  let compared = 0;
  for (const testCase of enumerateCases()) {
    const name = artifactName(testCase.caseId);
    const eager = JSON.parse(readFileSync(join(eagerDir, name), 'utf8'));
    const subject = JSON.parse(readFileSync(join(dir, name), 'utf8'));
    mismatches.push(...compareArtifacts(testCase.caseId, eager, subject));
    compared += 1;
  }
  if (compared !== EXPECTED_CASE_COUNT) throw new CaptureError('LAZY_CORPUS_INVALID', `compared ${compared}`);
  return {
    outcome: mismatches.length === 0 ? 'match' : 'mismatch',
    subjectStage: stage,
    comparedCases: compared,
    comparedFields: [...COMPARED_FIELDS],
    excludedFields: [...EXCLUDED_FIELDS],
    eagerFingerprint,
    eagerCaptureCommit: EAGER_CORPUS.captureCommit,
    subjectCaptureCommit: commit,
    subjectArtifactCount: names.length,
    mismatches,
  };
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const argv = process.argv.slice(2);
  const i = argv.indexOf('--stage');
  if (argv.length !== 3 || i !== 1 || !argv[0] || !argv[2]) {
    console.error('usage: compare_stage.mjs <corpus_dir> --stage <name>');
    process.exitCode = EXIT_USAGE;
  } else {
    try {
      const r = compareStageCorpus({ dir: argv[0], stage: argv[2] });
      console.log(JSON.stringify({ ...r, mismatches: r.mismatches.slice(0, 20) }, null, 2));
      console.log(`compared ${r.comparedCases} cases on ${r.comparedFields.join(', ')}; excluded ${r.excludedFields.join(', ')}`);
      if (r.outcome === 'match') {
        console.log('EXACT MATCH');
        console.log('Interpretation, fixed before any lazy trace existed: exact agreement on those fields across these 92 cases ONLY.');
        process.exitCode = EXIT_MATCH;
      } else {
        console.log(`MISMATCH: ${r.mismatches.length} compared-field difference(s)`);
        process.exitCode = EXIT_MISMATCH;
      }
    } catch (err) {
      console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
      const code = codeOfThrown(err);
      process.exitCode = code && /CORPUS|COMMIT|FINGERPRINT/.test(code) ? EXIT_CORPUS_INVALID : code === 'USAGE' ? EXIT_USAGE : EXIT_ERROR;
    }
  }
}
