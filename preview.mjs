import http from 'node:http';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const qa=process.argv.includes('--qa');
http.createServer((req,res)=>{try{
 if(req.url?.startsWith('/__qa/')){
  if(!qa){res.writeHead(404);res.end('Not found');return;}
  if(['/__qa/cluster-project.json','/__qa/learning-project.json','/__qa/distributed-project.json','/__qa/transformer-project.json','/__qa/contract-project.json','/__qa/model-project.json','/__qa/native-project.json'].includes(req.url)){const filename=req.url.split('/').at(-1),file=readFileSync(path.join(root,'.test-artifacts',filename));res.writeHead(200,{'Content-Type':'application/json','Content-Disposition':'attachment; filename="'+filename+'"'});res.end(file);return;}
  res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>Model Foundry browser acceptance</title><h1>浏览器验收存档</h1><a href="/__qa/cluster-project.json" download>下载集群验收工程</a><br><a href="/__qa/learning-project.json" download>下载完整训练与推理工程</a><br><a href="/__qa/distributed-project.json" download>下载逐 GPU 执行工程</a><br><a href="/__qa/transformer-project.json" download>下载 Transformer 数值工程</a><br><a href="/__qa/contract-project.json" download>下载统一模型与终端验收工程</a><br><a href="/__qa/model-project.json" download>下载开放模型工程</a><br><a href="/__qa/native-project.json" download>下载原生模型与调试工程</a></html>');return;
 }
 const html=readFileSync(path.join(root,'dist/index.html'));res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(req.method==='HEAD'?undefined:html);
 }catch{res.writeHead(500);res.end('Build the HTML or generate the QA fixture first.');}}).listen(4175,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4175'+(qa?' · QA archive enabled':'')));
