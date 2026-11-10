import { build } from 'esbuild';

await build({
  entryPoints: ['browser-entry.js'],
  bundle: true,
  outfile: 'browser-bundle.js',
  format: 'iife',
  platform: 'browser',
  sourcemap: false,
  logLevel: 'info'
});
