import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
let html=readFileSync(path.join(dir,'src/index.html'),'utf8').replace(/\r\n?/g,'\n');
for(const [key,file]of [['STYLES','style.css'],['ENGINE','engine.js'],['SYSTEMS','systems.js'],['RUNTIME','runtime.js'],['SCENE','scene.js'],['APP','app.js'],['CAMPUS','campus.js'],['OPERATIONS','operations.js']])html=html.replace('/* '+key+' */',()=>readFileSync(path.join(dir,'src',file),'utf8').replace(/\r\n?/g,'\n').replace(/<\/script/gi,'<\\/script'));
mkdirSync(path.join(dir,'dist'),{recursive:true});writeFileSync(path.join(dir,'dist/index.html'),html);writeFileSync(path.join(dir,'index.html'),html);console.log('Built index.html + dist/index.html · '+Buffer.byteLength(html)+' bytes · zero external dependencies');
