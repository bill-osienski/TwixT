import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  AlphaZeroClient,
  defaultServerUrl,
} from '../assets/js/ai/alphaZeroClient.js';

test('default server URL follows the page hostname', () => {
  assert.equal(defaultServerUrl({ hostname: 'localhost' }), 'http://localhost:3001');
  assert.equal(defaultServerUrl({ hostname: '127.0.0.1' }), 'http://127.0.0.1:3001');
});

test('default server URL remains usable outside a browser', () => {
  assert.equal(defaultServerUrl(undefined), 'http://localhost:3001');
});

test('an explicit server URL still overrides the page-derived default', () => {
  const client = new AlphaZeroClient('http://example.test:4567');
  assert.equal(client.serverUrl, 'http://example.test:4567');
});

test('IPv6 page hostnames are formatted as valid URLs', () => {
  assert.equal(defaultServerUrl({ hostname: '::1' }), 'http://[::1]:3001');
  // location.hostname already brackets IPv6 in some browsers; bracketing again
  // would produce http://[[::1]]:3001, which no browser will fetch.
  assert.equal(defaultServerUrl({ hostname: '[::1]' }), 'http://[::1]:3001');
});
