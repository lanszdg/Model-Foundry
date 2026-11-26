import http from 'node:http';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const qa=process.argv.includes('--qa');
http.createServer((req,res)=>{try{
 if(req.url?.startsWith('/__qa/')){
  if(!qa){res.writeHead(404);res.end('Not found');return;}
  if(req.url==='/__qa/cluster-project.json'){const file=readFileSync(path.join(root,'.test-artifacts/cluster-project.json'));res.writeHead(200,{'Content-Type':'application/json','Content-Disposition':'attachment; filename="cluster-project.json"'});res.end(file);return;}
  res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>Model Foundry browser acceptance</title><h1>浏览器验收存档</h1><a href="/__qa/cluster-project.json" download>下载集群验收工程</a></html>');return;
 }
 const html=readFileSync(path.join(root,'dist/index.html'));res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(req.method==='HEAD'?undefined:html);
 }catch{res.writeHead(500);res.end('Build the HTML or generate the QA fixture first.');}}).listen(4175,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4175'+(qa?' · QA archive enabled':'')));
