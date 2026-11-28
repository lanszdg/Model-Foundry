import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';
const dir=fileURLToPath(new URL('.',import.meta.url));
await build({entryPoints:[dir+'entry.mjs'],bundle:true,format:'iife',minify:true,legalComments:'inline',outfile:dir+'../../src/vendor/native-hf.js'});
await build({stdin:{contents:"import * as graphql from 'graphql'; globalThis.FoundryGraphQL=graphql;",resolveDir:dir},bundle:true,format:'iife',minify:true,legalComments:'inline',outfile:dir+'../../src/vendor/native-graphql.js'});
