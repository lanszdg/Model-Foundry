/* Logical target-model traffic on shared links; bounded kernels are the numerical oracle. */
(function(root){'use strict';const C=root.FoundryCore,protocol='project-links-v1',copy=x=>JSON.parse(JSON.stringify(x));
function groups(j,axes){const out=new Map();for(const r of j.budget.ranks){const key=axes.map(k=>r[k]).join(':');if(!out.has(key))out.set(key,[]);out.get(key).push(r.rank);}return [...out.values()];}
function event(kind,members,bytesPerRank,latencies=1){return{kind,members,bytesPerRank,latencies,progress:0,latencyRemaining:null,elapsed:0};}
function collectives(j,kind,quantity,factor,latencies=1){const out=[];for(const [prefix,axes]of [['dense',['tp','pp']],['expert',['tp','pp','ep']]])for(const rs of groups(j,axes)){const bytes=Math.max(0,...rs.map(i=>j.budget.ranks[i].traffic?.[prefix+quantity]||0));if(bytes&&rs.length>1)out.push(event(kind,rs,bytes*(rs.length-1)/rs.length*factor,latencies*(rs.length-1)));}return out;}
function plan(j,phase){const q=j.executionPlan.parallel,z=j.executionPlan.zeroStage,ranks=j.budget.ranks,events=[],bytes=rs=>Math.max(0,...rs.map(i=>ranks[i].traffic?.weightBytes??ranks[i].weights+ranks[i].adapters)),learn=rs=>Math.max(0,...rs.map(i=>ranks[i].trainable)),word=j.recipe.dtype==='fp64'?8:4,passes=(j.stage==='dpo'?2:1)*(j.executionPlan.recompute?3:2);
 if(phase==='prefetch'){
  if(j.recipe.offload)for(const r of ranks)events.push(event('HostToDevice',[r.rank],(r.weights+r.adapters)*passes));
  if(z===3)events.push(...collectives(j,'ParameterAllGather','WeightBytes',passes,passes));
 }else if(phase==='micro'){
  events.push(...collectives(j,z>=2?'GradientReduceScatter':'GradientAllReduce','Trainable',word*(z>=2?1:2)));
  // TP/CP/PP payloads use the target geometry. This is an analytic schedule, not a GPU profiler.
  const topo=C.ModelTopology.get(j.model).topology,h=topo.text?.hidden||topo.vision?.hidden||1024,depth=topo.text?.layers||topo.vision?.layers||1,seq=j.executionPlan.sequenceLength||64,activation=j.executionPlan.microBatch*seq*h*(j.recipe.dtype==='fp64'?8:j.recipe.dtype==='fp32'?4:2),passes=j.executionPlan.recompute?3:2;
  if(q.tp>1)for(const rs of groups(j,['dp','cp','pp']))events.push(event('TensorAllReduce',rs,activation*depth/q.pp*2*(q.tp-1)/q.tp*passes,depth*passes));
  if(q.cp>1)for(const rs of groups(j,['dp','tp','pp']))events.push(event('ContextKVAllGather',rs,activation*depth/q.pp*2*(q.cp-1)/q.cp*passes,depth*passes));
  if(q.pp>1)for(const rs of groups(j,['dp','cp','tp']))for(let i=1;i<rs.length;i++)events.push(event('PipelineSendRecv',[rs[i-1],rs[i]],activation*2/q.cp,2));
 }else if(phase==='commit'){
  if(z===1||z===2)events.push(...collectives(j,'UpdatedParameterAllGather','Trainable',j.recipe.dtype==='fp64'?8:j.recipe.dtype==='fp32'?4:2));
  if(j.recipe.offload)for(const r of ranks)events.push(event('OptimizerOffload',[r.rank],r.gradients+r.weights+r.adapters));
 }
 return events.filter(e=>Number.isFinite(e.bytesPerRank)&&e.bytesPerRank>0);
}
function enqueue(j,phase,world){const events=plan(j,phase);j.networkWait={protocol,phase,events,elapsed:0,done:events.length===0,timeout:j.recipe.collectiveTimeoutSeconds??world?.state.hardware.fabric.watchdog_s??30};return j.networkWait;}
function links(world,j,e){const h=world.state.hardware,hosts=new Map();for(const i of e.members){const a=j.allocation[i];if(!a)throw Error('Transport rank has no allocated GPU');hosts.set(a.node,(hosts.get(a.node)||0)+1);}const demands={},capacities={},put=(key,bytes,capacity)=>{demands[key]=(demands[key]||0)+bytes;capacities[key]=capacity;};
 for(const [node,count]of hosts){const n=world.node(node),fabric=world.state.engineering?.fabric[node],factor=n.fabricFactor??1;
  if(['HostToDevice','OptimizerOffload'].includes(e.kind))put('pcie:'+node,e.bytesPerRank*count,h.host.pcie_GBps*1e9);
  else if(hosts.size===1)put('nvlink:'+node,e.bytesPerRank*count,h.gpu.nvlink_GBps*1e9);
  else{put('ib:'+node,e.bytesPerRank*count,Math.min(h.fabric.ib_Gbps,fabric?.bandwidthGbps??h.fabric.ib_Gbps)*1e9/8*factor*(n.cables.ib?1:0));if(count>1)put('nvlink:'+node,e.bytesPerRank*count,h.gpu.nvlink_GBps*1e9);}
 }if(hosts.size>1)put('fabric',e.bytesPerRank*e.members.length,h.fabric.spine_Gbps*1e9/8);return{demands,capacities,kind:hosts.size>1?'IB':e.kind==='HostToDevice'||e.kind==='OptimizerOffload'?'PCIe':'NVLink'};
}
function running(world){return world.state.modelLab.projects.flatMap(p=>p.runs.filter(j=>j.status==='RUNNING'&&j.networkWait&&!j.networkWait.done));}
function advance(world,seconds){if(!Number.isFinite(seconds)||seconds<0)throw Error('Invalid transport time');let left=seconds,turns=0;const lab=world.state.modelLab;lab.transport??={protocol,now:world.state.clock-seconds};lab.transport.now+=seconds;
 while(left>1e-10&&turns++<4096){const jobs=running(world),active=[];if(!jobs.length)break;for(const j of jobs)for(const e of j.networkWait.events)if(e.progress<1){const path=links(world,j,e);if(e.latencyRemaining===null)e.latencyRemaining=world.state.hardware.fabric.latency_us/1e6*e.latencies;e.linkKind=path.kind;active.push({j,e,...path});}
  const users={};for(const a of active)if(a.e.latencyRemaining<=1e-12)for(const k of Object.keys(a.demands))users[k]=(users[k]||0)+1;
  let delta=left;for(const a of active){a.rate=a.e.latencyRemaining>1e-12?0:Math.min(...Object.keys(a.demands).map(k=>a.capacities[k]/users[k]/a.demands[k]));delta=Math.min(delta,a.e.latencyRemaining>1e-12?a.e.latencyRemaining:a.rate>0?(1-a.e.progress)/a.rate:Infinity);}
  for(const j of jobs)delta=Math.min(delta,Math.max(0,j.networkWait.timeout-j.networkWait.elapsed));if(delta<1e-12)delta=Math.min(left,1e-9);
  for(const a of active){const e=a.e;e.elapsed+=delta;if(e.latencyRemaining>1e-12)e.latencyRemaining=Math.max(0,e.latencyRemaining-delta);else e.progress=Math.min(1,e.progress+a.rate*delta);if(e.progress>1-1e-10)e.progress=1;}
  for(const j of jobs){const wait=j.networkWait;wait.elapsed+=delta;wait.done=wait.events.every(e=>e.progress===1);if(!wait.done&&wait.elapsed>=wait.timeout-1e-8)wait.error='NCCL watchdog: '+wait.phase+' exceeded '+wait.timeout+' seconds';if(wait.error){j.status='FAILED';j.error=wait.error;j.logs.push(wait.error+'; update not committed');j.allocation=[];}}
  left-=delta;
 }
}
function release(j){const wait=j.networkWait;if(!wait)return;if(wait.error)throw Error(wait.error);if(!wait.done)throw Error('Communication barrier is incomplete');j.transportHistory=(j.transportHistory||[]).concat({phase:wait.phase,elapsed:wait.elapsed,events:wait.events.map(e=>({kind:e.kind,link:e.linkKind||'local',bytesPerRank:e.bytesPerRank,participants:e.members.length,elapsed:e.elapsed}))}).slice(-32);delete j.networkWait;}
function summary(j){const w=j.networkWait;if(!w)return{phase:'READY',remainingBytes:0,elapsed:0,links:[],events:0};return{phase:w.phase,done:w.done,elapsed:w.elapsed,timeout:w.timeout,remainingBytes:w.events.reduce((n,e)=>n+e.bytesPerRank*e.members.length*(1-e.progress),0),links:[...new Set(w.events.map(e=>e.linkKind||'待调度'))],events:w.events.length,error:w.error};}
function validate(s){const w=s.networkWait;if(!w)return;if(w.protocol!==protocol||!['prefetch','micro','commit'].includes(w.phase)||!Number.isFinite(w.elapsed)||w.elapsed<0||!Number.isFinite(w.timeout)||w.timeout<=0||!Array.isArray(w.events)||w.events.some(e=>!Number.isFinite(e.progress)||e.progress<0||e.progress>1||!Number.isFinite(e.bytesPerRank)||e.bytesPerRank<=0||!Array.isArray(e.members)||e.members.some(i=>!Number.isInteger(i)||i<0||i>=s.recipe.gpus)))throw Error('Corrupt checkpoint transport');}
C.NetworkTimeline={protocol,plan,enqueue,links,advance,release,summary,validate};
})(globalThis);
