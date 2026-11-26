const assert=require('node:assert/strict'),fs=require('node:fs');
const C=require('./src/engine.js');require('./src/systems.js');require('./src/runtime.js');
const w=new C.World();w.state.started=true;let assertions=0;
function check(v,msg){assert.ok(v,msg);assertions++;}
function act(t,a={}){const r=w.action(t,a);assert.ok(r.ok,t+': '+r.text);return r;}
function wait(test,label,limit=4000){while(!test()&&limit-->0)w.tick(1);check(test(),label);}
function save(path,content){act('savefile',{path,content});}
function unit(name,host,verb='start'){return act('unit',{name,host,verb});}
function recipe(nodes=4,offload=false,steps=800,interval=200,wall='48:00:00'){
 let c=C.DEFAULT_CONFIG.replace('seq_length: 8192','seq_length: 1024').replace('train_steps: 2000','train_steps: '+steps).replace('checkpoint_interval: 400','checkpoint_interval: '+interval);
 if(offload)c+='optimizer_offload: cpu\n';save(C.CONFIG,c);
 save('/workspace/atlas/jobs/pretrain.sbatch','#!/bin/bash\n#SBATCH --nodes='+nodes+'\n#SBATCH --gpus-per-node=8\n#SBATCH --partition=compute\n#SBATCH --time='+wall+'\nsrun torchrun --nproc_per_node=8 train.py\n');act('check');
}
function submit(){act('submit');return w.state.jobs[0];}
for(const n of w.state.nodes){act('door',{id:n.id});for(const kind of ['power','mgmt',...(n.gpus?['ib']:[])])act('cable',{id:n.id,kind});act('node-power',{id:n.id});act('node-provision',{id:n.id});}
for(let i=0;i<6;i++)w.tick(1);
for(const [i,n]of w.state.nodes.entries()){act('node-net-save',{id:n.id,address:'10.42.0.'+(i+10),gateway:'10.42.0.1',dns:'10.42.0.10'});act('node-net-apply',{id:n.id});act('node-runtime',{id:n.id});}
check(!w.state.authenticated,'network connectivity must not create cluster authentication');
act('install',{service:'storage'});act('start',{service:'storage'});act('mount');
act('install',{service:'slurm'});check(!w.action('start',{service:'slurm'}).ok,'controller requires a running Munge socket');
act('cluster-auth');act('agents-start');act('start',{service:'slurm'});w.registerNodes();check(w.computes().every(n=>n.scheduler.registered),'nodes register individually');
for(const service of ['data','tracking','kubernetes','monitor']){act('install',{service});act('start',{service});}
const storage=w.unit('storage'),path=C.servicePath('storage'),original=w.state.files[path],pid=storage.pid;
save(path,original.replace('bandwidth_gbps: 5','bandwidth_gbps: 8'));
check(+w.serviceConfig('storage').bandwidth_gbps===5&&w.unitDirty(storage),'saving does not change applied configuration');
act('start',{service:'storage'});check(+w.serviceConfig('storage').bandwidth_gbps===5,'start is idempotent');
w.command('systemctl daemon-reload');check(+w.serviceConfig('storage').bandwidth_gbps===5,'daemon-reload does not reload application config');
act('reload',{service:'storage'});check(+w.serviceConfig('storage').bandwidth_gbps===8&&storage.pid===pid,'reload applies config without replacing PID');
save(path,original.replace('bandwidth_gbps: 5','bandwidth_gbps: broken'));check(!w.action('reload',{service:'storage'}).ok,'invalid reload rejected');check(storage.state==='active'&&+storage.config.bandwidth_gbps===8,'failed reload preserves active config');
save(path,original);act('reload',{service:'storage'});
const slurmFile=w.state.files['/etc/slurm/slurm.conf'];save('/etc/slurm/slurm.conf',slurmFile.replace('Gres=gpu:h100:8','Gres=gpu:h100:4'));w.registerNodes();check(w.computes().every(n=>n.scheduler.registered),'controller config needs reconfigure');act('reload',{service:'slurm'});check(w.computes().every(n=>n.scheduler.reason.includes('GRES mismatch')),'applied GRES mismatch rejects registration');save('/etc/slurm/slurm.conf',slurmFile);act('reload',{service:'slurm'});
const keyPath='/nodes/gpu-01/etc/munge/munge.key',key=w.state.files[keyPath];save(keyPath,'wrong-authentication-key');w.registerNodes();check(w.node('gpu-01').scheduler.registered,'key edits require Munge restart');unit('munge','gpu-01','restart');check(!w.node('gpu-01').scheduler.registered&&w.node('gpu-01').scheduler.reason.includes('credential'),'authentication mismatch has independent evidence');save(keyPath,key);unit('munge','gpu-01','restart');
act('ingest');act('clean');wait(()=>w.state.data.clean,'data clean');act('tokenize');wait(()=>w.state.data.tokenized,'tokenized');
recipe(2,true);const a=submit(),b=submit();w.tick(1);w.tick(1);w.tick(1);
check(a.status==='RUNNING'&&b.status==='RUNNING','two 16 GPU jobs run concurrently');
const owned=w.state.jobs.flatMap(j=>j.allocation.flatMap(a=>a.gpus.map(g=>a.node+':'+g)));check(new Set(owned).size===32&&owned.length===32,'no GPU is allocated twice');
recipe(4);const c=submit();w.tick(1);check(c.status==='PENDING'&&c.phase==='Resources','third job waits for actual resources');
act('cable',{id:'gpu-03',kind:'ib',connected:false});w.tick(1);check(b.status==='FAILED'&&a.status==='RUNNING','link failure affects only allocated workers');act('cable',{id:'gpu-03',kind:'ib'});act('cancel',{id:a.id});
const d=submit();act('job-control',{id:d.id,priority:500});w.tick(1);check(d.status==='CONFIGURING'&&c.status==='PENDING','priority determines next allocation');act('cancel',{id:d.id});act('job-control',{id:c.id,held:true});w.tick(1);check(c.phase==='JobHeldUser','hold prevents allocation');act('node-drain',{id:'gpu-04'});act('job-control',{id:c.id,held:false});w.tick(1);check(c.phase==='ReqNodeNotAvail','drain blocks new allocation');act('node-drain',{id:'gpu-04',drain:false});wait(()=>c.status==='RUNNING','resume drain permits scheduling');
act('node-drain',{id:'gpu-01'});w.tick(1);check(c.status==='RUNNING','draining does not kill an existing allocation');act('node-drain',{id:'gpu-01',drain:false});
wait(()=>w.state.checkpoints.filter(p=>p.jobId===c.id&&p.complete).length>=2,'two durable checkpoints');
wait(()=>w.state.checkpoints.some(p=>p.jobId===c.id&&p.status==='WRITING'),'another checkpoint writing');
const incomplete=w.state.checkpoints.find(p=>p.jobId===c.id&&p.status==='WRITING');check(!incomplete.complete&&incomplete.shards.some(s=>!s.complete),'writing checkpoint cannot be restored');
const midWrite=new C.World(JSON.parse(JSON.stringify(w.state))),savedBytes=incomplete.bytesWritten;check(midWrite.state.checkpoints.find(p=>p.id===incomplete.id).bytesWritten===savedBytes,'import freezes writer progress until tick');midWrite.tick(1);check(midWrite.state.checkpoints.find(p=>p.id===incomplete.id).bytesWritten>savedBytes,'incomplete writer continues after import');
const lostHost=new C.World(JSON.parse(JSON.stringify(w.state)));lostHost.action('node-power',{id:'gpu-01'});lostHost.tick(1);check(lostHost.state.checkpoints.find(p=>p.id===incomplete.id).status==='ABORTED','host failure cannot commit a pending checkpoint');
const lostStorageLink=new C.World(JSON.parse(JSON.stringify(w.state)));lostStorageLink.action('cable',{id:'storage-01',kind:'mgmt',connected:false});lostStorageLink.tick(1);check(lostStorageLink.unit('storage').state==='active'&&!lostStorageLink.state.services.storage&&lostStorageLink.state.jobs.find(j=>j.id===c.id).status==='FAILED','running storage process can be unreachable to clients');
act('fault',{kind:'storage'});w.tick(1);check(c.status==='FAILED'&&incomplete.status==='ABORTED','I/O failure leaves an incomplete checkpoint');
const committed=w.state.checkpoints.filter(p=>p.jobId===c.id&&p.complete),newest=committed.at(-1),fallback=committed.at(-2);act('checkpoint-corrupt',{id:newest.id});check(!w.verifyCheckpoint(newest).ok,'shard checksum detects corruption');act('start',{service:'storage'});act('resume',{id:c.id});const resumed=w.state.jobs[0];
check(resumed.step===fallback.step,'resume skips incomplete and corrupt checkpoints');check(JSON.stringify(resumed.profile)===JSON.stringify(fallback.payload.profile),'model restored from selected checkpoint');check(JSON.stringify(resumed.optimizer)===JSON.stringify(fallback.payload.optimizer),'optimizer restored');check(resumed.rng===fallback.payload.rng&&resumed.consumedTokens===fallback.payload.consumedTokens,'RNG and data cursor restored');
wait(()=>resumed.status==='COMPLETED','resumed pretraining completes');check(w.verifyCheckpoint(w.latestCheckpoint()).ok,'final checkpoint is committed before job completion');check(w.collectMetrics().allocated===0,'GPU released after final flush');
act('sftdata');for(const stage of ['sft','dpo']){recipe(4,false,300,200);save(C.CONFIG,w.state.files[C.CONFIG].replace('stage: pretrain','stage: '+stage).replace('learning_rate: 0.00015','learning_rate: '+(stage==='sft'?'0.00002':'0.000005')));act('check');const run=submit();wait(()=>run.status==='COMPLETED',stage+' completes');}
act('evaluate');wait(()=>w.state.evaluation.status==='COMPLETED','evaluation completes');check(w.state.evaluation.passed,'trained checkpoint evaluates successfully');act('export');act('deploy');wait(()=>w.state.deployment.status==='READY','serving ready');check(w.collectMetrics().allocated===8,'deployment owns eight concrete GPUs');const answer=w.infer('解释梯度累积');check(answer.tokens>0&&answer.text.includes('梯度'),'exported profile still drives inference');
act('undeploy');recipe(4,false,800,200,'00:01:00');const timed=submit();wait(()=>timed.status==='TIMEOUT','wall time limit enforced');check(timed.allocation.length===0,'timeout releases GPU allocation');
check(w.command('scontrol show nodes').text.includes('Gres='),'node inspection command');check(w.command('ps aux').text.includes('slurmd'),'process inspection reads live units');check(w.command('ss -lntp').text.includes('6817'),'socket inspection reads live units');
const restored=new C.World(JSON.parse(JSON.stringify(w.state)));check(restored.unit('storage').configHash===w.unit('storage').configHash,'applied snapshots persist');check(restored.verifyCheckpoint(restored.latestCheckpoint()).ok,'checkpoint files verify after import');
if(process.argv.includes('--fixture')){fs.mkdirSync('.test-artifacts',{recursive:true});fs.writeFileSync('.test-artifacts/cluster-project.json',JSON.stringify({format:'model-foundry',state:{...w.state,camera:{x:-4.2,z:8.9,yaw:0,pitch:-.06}}}));}
console.log('PASS: '+assertions+' v0.3 integration assertions: process snapshots, registration, resource contention, scoped failures, durable checkpoints, resume, alignment and deployment.');
