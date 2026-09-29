import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const app = fs.readFileSync(path.join(root, 'js/app.js'), 'utf8');

const resolveLocal = (href) => {
  const local = path.resolve(root, href.split(/[?#]/)[0]);
  assert.ok(local.startsWith(root + path.sep), href);
  assert.ok(fs.existsSync(local), href);
};

test('every local src/href in the page exists and stays relative for the /gyroscope/ subpath', () => {
  for (const [, href] of html.matchAll(/(?:src|href|content)=["']((?:\.\/|js\/|css\/)[^"']+|og-image\.jpg)["']/g)) resolveLocal(href);
  assert.equal(/(?:src|href)=["']\//.test(html), false, 'no root-absolute paths');
});

test('the import map and module imports point at vendored files', () => {
  const map = JSON.parse(html.match(/<script type="importmap">([\s\S]*?)<\/script>/)[1]);
  resolveLocal(map.imports.three);
  for (const [, spec] of app.matchAll(/from '(\.[^']+)'/g)) resolveLocal(path.join('js', spec));
  const vendor = fs.readFileSync(path.join(root, 'js/vendor/three.module.js'), 'utf8');
  for (const [, spec] of vendor.matchAll(/from '(\.[^']+)'/g)) resolveLocal(path.join('js/vendor', spec));
});

test('every control slot and preset link in the page is backed by the app', () => {
  for (const [, key] of html.matchAll(/data-key="([^"]+)"/g)) assert.ok(app.includes(`  ${key}: {`), key);
  for (const [, id] of html.matchAll(/data-preset="([^"]+)"/g)) assert.ok(app.includes(`id: '${id}'`), id);
  for (const [, id] of app.matchAll(/getElementById\('([^']+)'\)/g)) assert.ok(html.includes(`id="${id}"`), id);
});
