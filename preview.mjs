import http from 'node:http';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
http.createServer((req,res)=>{try{const html=readFileSync(path.join(root,'dist/index.html'));res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(req.method==='HEAD'?undefined:html);}catch{res.writeHead(500);res.end('Build the HTML first.');}}).listen(4175,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4175'));
