const assert=require('node:assert/strict'),fs=require('node:fs');
const C=require('./src/engine.js');for(const n of ['systems','runtime','python','corpus','learning','accumulation','decoder','serving','artifacts','distributed'])require('./src/'+n+'.js');
const saved=JSON.parse(fs.readFileSync('.test-artifacts/learning-project.json','utf8')).state;
let checks=0;const check=(v,m)=>{assert.ok(v,m);checks++;},equal=(a,b,m)=>{assert.deepEqual(a,b,m);checks++;};
function fresh(){const w=new C.World(saved);w.state.started=true;w.state.speed=1;w.action('undeploy');w.state.jobs=[];w.state.checkpoints=[];w.state.models=[];w.state.distributed.debt=0;w.state.distributed.remaining=0;return w;}
function recipe(w,overrides={},nodes=4){let text=C.DEFAULT_CONFIG;for(const [k,v]of Object.entries({seq_length:1024,pipeline_parallel:4,global_batch_size:8,train_steps:3,warmup_steps:1,checkpoint_interval:2,...overrides})){const pattern=new RegExp('^'+k+':.*$','m');text=pattern.test(text)?text.replace(pattern,k+': '+v):text+k+': '+v+'\n';}w.state.files[C.CONFIG]=text;w.state.files['/workspace/atlas/jobs/pretrain.sbatch']='#!/bin/bash\n#SBATCH --nodes='+nodes+'\n#SBATCH --gpus-per-node=8\n#SBATCH --time=48:00:00\n#SBATCH --partition=compute\nsrun torchrun --nproc_per_node=8 train.py\n';}
function submit(w){const c=w.action('check');assert.ok(c.ok,c.text);const r=w.action('submit');assert.ok(r.ok,r.text);return w.state.jobs[0];}
function until(w,test,message,dt=.2,max=20000){for(let i=0;i<max&&!test();i++)w.tick(dt);check(test(),message);}
function ready(w,j){until(w,()=>['RUNNING','FAILED','OUT_OF_MEMORY','NODE_FAIL','TIMEOUT','COMPLETED'].includes(j.status),'allocation and rendezvous resolve',.01);check(j.status==='RUNNING',j.logs.at(-1)?.text);}
function finish(w,j){until(w,()=>['FAILED','COMPLETED'].includes(j.status),'job reaches terminal state');check(j.status==='COMPLETED',j.logs.at(-1)?.text);}

const normal=fresh();recipe(normal);const nj=submit(normal);ready(normal,nj);
equal(nj.ranks.length,32,'one runtime record exists per physical GPU');
equal(new Set(nj.ranks.map(r=>r.node+':'+r.gpu)).size,32,'rank mapping never duplicates a GPU');
check(nj.ranks.every(r=>r.rank===(((r.pp*nj.info.dp+r.dp)*nj.config.context_parallel+r.cp)*nj.config.tensor_parallel+r.tp)),'rank coordinates follow declared tp-cp-dp-pp order');
until(normal,()=>nj.execution?.completedMicro.length>0&&nj.step===0,'backward work precedes optimizer commit',.0005);
equal(nj.optimizer.language.t,0,'partial backward work cannot update Adam');
equal(nj.batchCursor,0,'partial backward work cannot commit the data cursor');
check(normal.action('execution-pause').ok,'engineering clock pauses independently of scene controls');const pausedAt=normal.state.distributed.now;normal.tick(10);equal(normal.state.distributed.now,pausedAt,'paused execution does not advance on normal ticks');check(normal.action('execution-step').ok,'debug step advances one event boundary');check(normal.state.distributed.now>=pausedAt&&normal.state.distributed.paused,'single event preserves paused state');normal.action('execution-resume');
const restored=new C.World(normal.state);restored.state.started=true;restored.state.speed=1;const restoredJob=restored.state.jobs[0];
finish(normal,nj);finish(restored,restoredJob);
equal(restoredJob.profile.language,nj.profile.language,'mid-step project reload replays exactly the same numerical work');
equal(restoredJob.optimizer.language,nj.optimizer.language,'mid-step reload preserves Adam state');
equal(restoredJob.rng,nj.rng,'mid-step reload preserves data RNG');
equal(nj.modelMetrics.samples,8,'each optimizer step consumes the configured global batch');
equal(nj.modelMetrics.microbatches,8,'microbatch reduction count matches GBS / MBS');
equal(nj.batchCursor,24,'three steps consume twenty-four logical samples');
equal(nj.optimizer.language.t,3,'Adam increments only at complete global barriers');
check(nj.lastExecution.tasks.every(t=>t.status==='DONE'),'final graph contains no unfinished operation');
for(const t of nj.lastExecution.tasks)for(const dep of t.deps)assert.ok(t.started+1e-7>=nj.lastExecution.tasks[dep].ended,'task begins only after all dependencies finish');
checks++;check(nj.lastExecution.tasks.some(t=>t.kind==='RecomputeForward'),'activation recomputation has explicit compute events');
check(nj.lastExecution.ranks.some(r=>r.wait>0),'pipeline fill, drain and barriers create idle time');
check(normal.latestCheckpoint().payload.program.executionVersion===C.Distributed.version,'checkpoint identifies execution protocol');
equal(normal.latestCheckpoint().payload.batchCursor,nj.batchCursor,'checkpoint binds the committed reader position');
check(normal.verifyCheckpoint(normal.latestCheckpoint()).ok,'distributed checkpoint checksums still verify');

const dataParallel=fresh();recipe(dataParallel,{pipeline_parallel:2});const dj=submit(dataParallel);finish(dataParallel,dj);
equal(dj.info.dp,2,'parallel layout creates two data replicas');
equal(dj.info.accum,4,'gradient accumulation adjusts to DP size');
equal(dj.profile.language,nj.profile.language,'parallel mapping preserves the bounded numerical objective');
equal(dj.optimizer.language,nj.optimizer.language,'DP reduction preserves the same optimizer update');
const context=fresh();recipe(context,{tensor_parallel:4,pipeline_parallel:2,context_parallel:2});const cj=submit(context);finish(context,cj);
equal(cj.info.dp,2,'TP4 / CP2 / PP2 yields DP2 on 32 GPUs');
check(cj.lastExecution.tasks.filter(t=>t.kind==='CP Attention').some(t=>t.units>0),'CP attention performs communication');
check(cj.lastExecution.tasks.filter(t=>t.kind==='DP ReduceScatter').some(t=>t.units>0),'DP / CP gradient reduction performs communication');

const slow=fresh();recipe(slow);const sj=submit(slow);slow.action('gpu-throttle',{node:'gpu-01',gpu:0,factor:.25});ready(slow,sj);
until(slow,()=>sj.ranks.some(r=>r.state.startsWith('WAIT_PEER')),'fast TP peers wait for the slowed rank',.00005);
finish(slow,sj);
check(sj.distributedHistory[0].duration>nj.distributedHistory[0].duration,'a slow GPU increases complete step duration');
equal(sj.profile.language,nj.profile.language,'slower hardware changes scheduling without changing data or updates');
check(sj.lastExecution.ranks[0].compute>sj.lastExecution.ranks[1].compute,'per-rank compute time reflects the injected slowdown');

const noRecompute=fresh();recipe(noRecompute,{activation_recompute:'false'});const nr=submit(noRecompute);finish(noRecompute,nr);
check(nr.distributedHistory[0].duration<nj.distributedHistory[0].duration,'removing recomputation reduces compute time');
check(nr.lastExecution.ranks[0].peakMemory>nj.lastExecution.ranks[0].peakMemory,'removing recomputation increases activation memory');

const lost=fresh();recipe(lost);const lj=submit(lost);ready(lost,lj);const beforeRows=JSON.stringify(lj.profile.language.rows);lost.action('cable',{id:'gpu-01',kind:'ib',connected:false});
until(lost,()=>lj.status==='FAILED','inter-node collective reaches its watchdog',.01);
check(lj.logs.at(-1).text.includes('watchdog timeout')&&lj.logs.at(-1).text.includes('endpoints='),'failure identifies the blocked collective and physical endpoints');
equal(lj.step,0,'failed collective cannot commit a partial step');
equal(lj.optimizer.language.t,0,'failed collective cannot partially update Adam');
equal(JSON.stringify(lj.profile.language.rows),beforeRows,'failed step leaves committed model state untouched');
equal(lost.collectMetrics().allocated,0,'failed ranks release GPU allocations');
check(lj.lastExecution.tasks.some(t=>t.status==='DONE'&&t.kind==='Forward'),'work can progress before reaching the broken link dependency');

const local=fresh();recipe(local,{pipeline_parallel:1,optimizer_offload:'cpu'},1);const localJob=submit(local);ready(local,localJob);local.action('cable',{id:'gpu-01',kind:'ib',connected:false});finish(local,localJob);
check(localJob.lastExecution.tasks.some(t=>t.kind==='PCIe D2H')&&localJob.lastExecution.tasks.some(t=>t.kind==='CPU Adam'),'CPU offload has PCIe and host-memory work');
check(localJob.lastExecution.tasks.filter(t=>t.groups).every(t=>t.groups.every(g=>new Set(g.map(i=>localJob.lastExecution.ranks[i].node)).size===1)),'single-node job does not require inter-node IB');

const lowRam=fresh(),hw=structuredClone(lowRam.state.hardware);hw.host.memory_GiB=128;lowRam.state.files[C.Distributed.hardwarePath]=JSON.stringify(hw);check(lowRam.action('hardware-apply').ok,'idle hardware profile applies');recipe(lowRam,{pipeline_parallel:1,optimizer_offload:'cpu'},1);const ramJob=submit(lowRam);until(lowRam,()=>ramJob.status==='FAILED','offload overcommit fails with host OOM');check(ramJob.logs.at(-1).text.includes('Host OOM'),'host memory shortage has a distinct diagnostic');

const interrupted=fresh();recipe(interrupted,{train_steps:4,checkpoint_interval:2});const ij=submit(interrupted);until(interrupted,()=>ij.pendingCheckpoint&&ij.step===2,'periodic checkpoint begins after a complete optimizer step',.02);const cp=interrupted.state.checkpoints.find(p=>p.id===ij.pendingCheckpoint);equal(cp.payload.batchCursor,16,'periodic checkpoint contains exactly two global batches');until(interrupted,()=>cp.complete,'checkpoint commits');interrupted.action('cancel',{id:ij.id});const resumed=interrupted.action('resume',{id:ij.id});check(resumed.ok,'rank protocol resumes from a committed checkpoint');const restart=interrupted.state.jobs[0];check(!restart.execution&&restart.batchCursor===16,'resume rebuilds the pipeline from committed data and optimizer state');finish(interrupted,restart);equal(restart.optimizer.language.t,4,'resumed job applies only remaining optimizer steps');

const shared=fresh(),sharedProfile=structuredClone(shared.state.hardware);sharedProfile.fabric.spine_Gbps=1;shared.state.files[C.Distributed.hardwarePath]=JSON.stringify(sharedProfile);check(shared.action('hardware-apply').ok,'fabric budget applies while idle');recipe(shared,{tensor_parallel:4,pipeline_parallel:2,optimizer_offload:'cpu',checkpoint_interval:3},2);const first=submit(shared),second=submit(shared);until(shared,()=>first.step>=1&&second.step>=1,'two jobs progress through the shared fabric',.1);equal(new Set([...first.allocation,...second.allocation].flatMap(a=>a.gpus.map(g=>a.node+':'+g))).size,32,'concurrent jobs own disjoint GPU cards');
const alone=fresh();alone.state.files[C.Distributed.hardwarePath]=JSON.stringify(sharedProfile);alone.action('hardware-apply');recipe(alone,{tensor_parallel:4,pipeline_parallel:2,optimizer_offload:'cpu',checkpoint_interval:3},2);const single=submit(alone);until(alone,()=>single.step>=1,'single fabric reference completes one step',.1);check(first.distributedHistory[0].duration>single.distributedHistory[0].duration,'shared fabric contention increases measured event duration');check(!shared.action('hardware-apply').ok,'active jobs protect the applied hardware snapshot');

const aligned=new C.World(normal.state);aligned.state.started=true;aligned.state.speed=1;check(aligned.action('sftdata').ok,'alignment dataset is prepared');recipe(aligned,{stage:'sft',global_batch_size:4,train_steps:2,checkpoint_interval:2});const sftJob=submit(aligned);finish(aligned,sftJob);check(sftJob.profile.language.steps>nj.profile.language.steps&&sftJob.modelMetrics.supervisedTokens>4,'SFT accumulates assistant Token supervision over the global batch');recipe(aligned,{stage:'dpo',global_batch_size:4,train_steps:2,checkpoint_interval:2,dpo_beta:0});const dpoJob=submit(aligned);finish(aligned,dpoJob);check(Math.abs(dpoJob.modelMetrics.loss-Math.log(2))<1e-12&&dpoJob.modelMetrics.gradNorm===0,'DPO beta zero gives log(2) loss and zero preference gradient');check(dpoJob.modelMetrics.supervisedTokens>4,'DPO reports completion Token count separately from pair weighting');
const clipped=fresh();recipe(clipped,{train_steps:2,checkpoint_interval:2,grad_clip:0});const clipJob=submit(clipped);finish(clipped,clipJob);check(clipJob.modelMetrics.clippedGradNorm===0,'gradient clipping zero is applied rather than replaced by a default');

const mapped=fresh();recipe(mapped);mapped.state.files['/workspace/atlas/project/recipe.py']=mapped.state.files['/workspace/atlas/project/recipe.py'].replace('batch = next(data_iterator)','assert state.pp_rank == 3\n    batch = next(data_iterator)');check(mapped.action('check').ok,'preflight uses the representative loss-stage rank context');
check(JSON.parse(normal.state.files['/runs/'+nj.id+'/data-manifest.json']).dataset===nj.datasetSnapshot.version,'run manifest freezes data and sampling lineage');

// Independent, hand-computable reference for token-weighted accumulation.
const original=C.Learning.sampleLoss,source=C.Accumulation.payload(nj),ctx=Object.keys(source.profile.language.rows)[0],token=Object.keys(source.profile.language.rows[ctx])[0],key=ctx+'|'+token;
source.config={...source.config,global_batch_size:2,micro_batch_size:1,grad_clip:100,weight_decay:0};source.optimizer.language={m:{},v:{},t:0};source.profile.language.rows[ctx][token]=10;source.logs=[];
try{C.Learning.sampleLoss=(world,j,runner,p)=>p.cursor%2?{loss:new C.Learning.Scalar(3,{[key]:6}),weight:3}:{loss:new C.Learning.Scalar(1,{[key]:2}),weight:1};source.batchCursor=0;const acc=C.Accumulation.create(normal,source,1);acc.run(1);acc.run(0);const result=acc.finish();equal(result.modelMetrics.loss,2.5,'weighted NLL matches (1*1+3*3)/4');equal(result.modelMetrics.gradNorm,5,'weighted gradient matches (2*1+6*3)/4');check(Math.abs(result.optimizer.language.m[key]-.5)<1e-12,'Adam first moment uses the globally reduced gradient');check(Math.abs(result.optimizer.language.v[key]-.025)<1e-12,'Adam second moment uses the globally reduced gradient');}finally{C.Learning.sampleLoss=original;}

if(process.argv.includes('--fixture')){
 const fixture=fresh();recipe(fixture,{train_steps:2000,warmup_steps:100,checkpoint_interval:100});const previewJob=submit(fixture);ready(fixture,previewJob);fixture.state.speed=1;fs.writeFileSync('.test-artifacts/distributed-project.json',JSON.stringify({format:'model-foundry',state:fixture.state}));
}
console.log('PASS: '+checks+' distributed execution, independent reduction and failure checks.');
