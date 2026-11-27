/* Model Foundry: shared world, service, filesystem and job state. No runtime dependencies. */
(function(root){
'use strict';
const VERSION=1, CONFIG='/workspace/atlas/configs/pretrain.yaml';
const clone=x=>JSON.parse(JSON.stringify(x));
const DEFAULT_CONFIG=`model: atlas-70b\nstage: pretrain\nprecision: bf16\nseq_length: 8192\nmicro_batch_size: 1\nglobal_batch_size: 128\nlearning_rate: 0.00015\nwarmup_steps: 100\ntrain_steps: 2000\ntensor_parallel: 8\npipeline_parallel: 2\ncontext_parallel: 1\nactivation_recompute: true\ncheckpoint_interval: 400\nseed: 42\n`;
const TRAIN=`# Atlas distributed training entry point\nfrom megatron.bridge.training.pretrain import pretrain\nfrom megatron.bridge.training.finetune import finetune\nfrom project.recipe import load_recipe, forward_step\n\ncfg = load_recipe("configs/pretrain.yaml")\nentry = pretrain if cfg.stage == "pretrain" else finetune\nentry(cfg, forward_step_func=forward_step)\n`;
const initial=()=>({version:VERSION,clock:0,speed:1,budget:480000,spent:0,started:false,host:'workstation',cwd:'/workspace/atlas',
 nodes:['control-01','storage-01','gpu-01','gpu-02','gpu-03','gpu-04'].map((id,i)=>({id,role:i===0?'control':i===1?'storage':'compute',gpus:i>1?8:0,power:false,os:false,ip:'',runtime:false,health:'OK'})),
 installed:{storage:false,slurm:false,data:false,tracking:false,kubernetes:false,monitor:false},services:{storage:false,slurm:false,data:false,tracking:false,kubernetes:false,monitor:false},
 mounts:false,authenticated:false,network:false,data:{raw:false,clean:false,tokenized:false,sft:false,version:0,quality:0,docs:0,tokens:0,rules:{dedup:true,quality:true,pii:true}},
 checkedHash:'',files:{[CONFIG]:DEFAULT_CONFIG,'/workspace/atlas/train.py':TRAIN,'/workspace/atlas/jobs/pretrain.sbatch':'#!/bin/bash\n#SBATCH --job-name=atlas-70b\n#SBATCH --nodes=4\n#SBATCH --gpus-per-node=8\n#SBATCH --time=48:00:00\n#SBATCH --partition=compute\nsrun torchrun --nproc_per_node=8 train.py\n','/workspace/atlas/README.txt':'ATLAS / 70B\n工程目标：从基础设施建设到大模型交付。\n配置位于 configs/pretrain.yaml。\n先建立网络、存储、计算运行时和 Slurm。\n'},
 jobs:[],checkpoints:[],models:[],evaluation:null,deployment:null,events:[],nextJob:24001,incident:null,terminal:[{type:'muted',text:'FoundryOS 24.04 LTS · workstation\n项目目录 /workspace/atlas\n输入 help 查看工程命令。'}],history:[],camera:{x:0,z:8,yaw:0,pitch:-.06},completed:[]});
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return(h>>>0).toString(16);}
function parseConfig(text){const c={};for(const line of text.split('\n')){const l=line.replace(/\s+#.*$/,'').trim();if(!l||l.startsWith('#'))continue;const m=l.match(/^([a-z_]+):\s*(.+)$/);if(!m)throw Error('配置语法错误：'+l);c[m[1]]=m[2].trim();}for(const k of ['seq_length','micro_batch_size','global_batch_size','learning_rate','warmup_steps','train_steps','tensor_parallel','pipeline_parallel','context_parallel','checkpoint_interval','seed']){if(!(k in c)||!Number.isFinite(Number(c[k])))throw Error('缺少或无效的数值字段：'+k);c[k]=Number(c[k]);}return c;}
class World{
 constructor(saved){this.state=initial();if(saved){if(saved.version!==VERSION)throw Error('项目存档版本不兼容');if(!Array.isArray(saved.nodes)||saved.nodes.length!==6||!saved.files||!Array.isArray(saved.jobs)||!saved.services||!saved.data)throw Error('项目存档结构不完整');this.state=Object.assign(initial(),clone(saved));}this.listeners=[];if(root.FoundryCore?.upgrade)root.FoundryCore.upgrade(this.state,!!saved);}
 on(fn){this.listeners.push(fn);return()=>this.listeners=this.listeners.filter(f=>f!==fn);}
 emit(){this.listeners.forEach(f=>f(this.state));}
 event(text,level='info'){this.state.events.unshift({t:this.state.clock,text,level});this.state.events=this.state.events.slice(0,250);}
 need(ok,message){if(!ok)throw Error(message);}
 computes(){return this.state.nodes.filter(n=>n.role==='compute');}
 all(field){return this.state.nodes.every(n=>!!n[field]);}
 config(){return parseConfig(this.state.files[CONFIG]);}
 fingerprint(){return hash([CONFIG,'/workspace/atlas/train.py','/workspace/atlas/jobs/pretrain.sbatch'].map(p=>this.state.files[p]||'').join('\n---\n'));}
 configInfo(c=this.config()){
  const gpus=this.batchRequest?this.batchRequest().gpus:32;
  const world=c.tensor_parallel*c.pipeline_parallel*c.context_parallel,dp=gpus/world,accum=c.global_batch_size/(c.micro_batch_size*dp);
  const stateMemory=70/Math.max(1,c.tensor_parallel*c.pipeline_parallel)*(6+(c.optimizer_offload==='cpu'?0:12)/Math.max(1,dp*c.context_parallel));
  const activation=6*c.micro_batch_size*(c.seq_length/8192)*(c.activation_recompute==='true'?1:3.2)/Math.max(1,c.context_parallel);
  return{dp,accum,memory:stateMemory+activation+4,gpus,tokensPerStep:c.global_batch_size*c.seq_length};
 }
 validate(c=this.config()){
  const allowed=['model','stage','precision','seq_length','micro_batch_size','global_batch_size','learning_rate','warmup_steps','train_steps','tensor_parallel','pipeline_parallel','context_parallel','activation_recompute','checkpoint_interval','seed','optimizer_offload','grad_clip','weight_decay','dpo_beta'];
  for(const key of ['grad_clip','weight_decay','dpo_beta'])if(c[key]!==undefined){c[key]=Number(c[key]);this.need(Number.isFinite(c[key])&&c[key]>=0,key+' 必须为非负数');}for(const k of Object.keys(c))this.need(allowed.includes(k),'未知配置字段：'+k);
  this.need(c.model==='atlas-70b','当前工程模型为 atlas-70b');this.need(['pretrain','sft','dpo'].includes(c.stage),'stage 必须为 pretrain、sft 或 dpo');
  this.need(c.precision==='bf16','当前镜像支持 bf16 配方');this.need(['true','false'].includes(c.activation_recompute),'activation_recompute 必须是 true 或 false');
  for(const k of ['seq_length','micro_batch_size','global_batch_size','train_steps','tensor_parallel','pipeline_parallel','context_parallel','checkpoint_interval'])this.need(Number.isInteger(c[k])&&c[k]>0,k+' 必须是正整数');
  this.need(Number.isInteger(c.warmup_steps)&&c.warmup_steps>=0&&c.warmup_steps<c.train_steps,'warmup_steps 必须小于 train_steps');
  this.need(c.seq_length>=128&&c.seq_length<=131072,'seq_length 范围 128—131072');this.need(c.train_steps<=1000000,'train_steps 超出工程预算');
  this.need(c.learning_rate>0&&c.learning_rate<=.1,'learning_rate 范围 (0, 0.1]');
  this.need(!c.optimizer_offload||['none','cpu'].includes(c.optimizer_offload),'optimizer_offload 必须为 none 或 cpu');
  const info=this.configInfo(c);this.need(Number.isInteger(info.dp)&&info.dp>=1,'TP × PP × CP 必须整除 world_size='+info.gpus);
  const target=globalThis.FoundryCore.ModelSpec?globalThis.FoundryCore.ModelSpec.resolve(this).target:{num_attention_heads:64,num_hidden_layers:80};this.need(target.num_attention_heads%c.tensor_parallel===0,'attention_heads 必须能被 TP 整除');this.need(target.num_hidden_layers%c.pipeline_parallel===0,'layers 必须能被 PP 整除');
  this.need(Number.isInteger(info.accum)&&info.accum>=1,'global_batch 必须是 micro_batch × DP 的整数倍');return info;
 }
 action(type,arg={}){
  try{const s=this.state;let result='';
   if(type==='power'){const list=arg.id?s.nodes.filter(n=>n.id===arg.id):s.nodes;this.need(list.length,'未知节点');list.forEach(n=>n.power=true);result='BMC：'+list.map(n=>n.id).join(', ')+' 已上电';}
   else if(type==='provision'){this.need(this.all('power'),'BMC：存在未上电节点，请先上电');s.nodes.forEach(n=>n.os=true);result='系统镜像 ubuntu-24.04 部署完成，6 个节点可通过管理控制台访问';}
   else if(type==='network'){this.need(this.all('os'),'请先为所有节点部署操作系统');s.nodes.forEach((n,i)=>n.ip='10.42.0.'+(10+i));s.network=true;s.authenticated=true;result='管理网 / 存储网 / IB 网络连通；DNS、NTP 和节点认证已配置';}
   else if(type==='runtime'){this.need(s.network,'网络不可达，无法分发运行时');s.nodes.forEach(n=>n.runtime=true);result='运行时已分发：NVIDIA driver、CUDA、NCCL、container runtime；32 张 H100 可见';}
   else if(type==='install'){
    const service=arg.service;this.need(Object.hasOwn(s.installed,service),'未知软件包：'+service);this.need(s.network,'软件源不可达：请先配置节点网络');
    if(['data','tracking'].includes(service))this.need(s.services.storage&&s.mounts,'存储端点或 /data 挂载不可用');
    if(['slurm','kubernetes'].includes(service))this.need(this.all('runtime'),'GPU 运行时尚未就绪');
    s.installed[service]=true;result=service+' 软件包已安装；配置已写入 /etc/foundry/'+service+'.conf';s.files['/etc/foundry/'+service+'.conf']='service: '+service+'\nlisten: 10.42.0.10\n';
   }
   else if(type==='start'){
    const service=arg.service;this.need(s.installed[service],service+': unit not found，请先安装');this.need(s.network,'节点网络不可达');
    if(service==='slurm')this.need(s.authenticated&&this.all('runtime'),'节点认证或 GPU GRES 初始化失败');
    if(['data','tracking'].includes(service))this.need(s.services.storage&&s.mounts,'对象存储连接失败：ECONNREFUSED');
    s.services[service]=true;result=service==='slurm'?'slurmctld active；4 个计算节点注册成功，32 GPU IDLE':service+' active (running)';
   }
   else if(type==='stop'){this.need(Object.hasOwn(s.services,arg.service),'未知服务');s.services[arg.service]=false;result=arg.service+' 已停止';}
   else if(type==='mount'){this.need(s.services.storage,'mount: storage-01:/data: Connection refused');s.mounts=true;result='storage-01:/data mounted on /data；对象存储 s3://atlas 已连接';}
   else if(type==='unmount'){s.mounts=false;result='/data 已卸载';}
   else if(type==='ingest'){
    this.need(s.services.data&&s.services.storage&&s.mounts,'数据服务或存储未就绪');this.need(!this.runningTrain(),'当前训练已绑定数据版本，请等待结束再构建新版本');
    s.data={...s.data,raw:true,clean:false,tokenized:false,sft:false,version:s.data.version+1,quality:46,docs:12480000,tokens:42800000000};s.checkedHash='';
    s.files['/data/raw/manifest.json']=JSON.stringify({version:s.data.version,documents:12480000,tokens:42800000000,sources:['web-zh','books','code','science'],shards:512},null,2);result='接入任务完成：512 shards · 12.48M documents · 42.8B tokens';
   }
   else if(type==='clean'){
    this.need(s.services.data&&s.services.storage&&s.mounts,'数据处理 Worker 无法访问存储');this.need(s.data.raw,'请先登记并接入原始语料');this.need(!this.runningTrain(),'训练运行中，数据构建被锁定');
    const r=arg.rules||s.data.rules;s.data.rules=clone(r);const retention=(r.dedup?.82:1)*(r.quality?.88:1)*(r.pii?.98:1);
    s.data.docs=Math.round(12480000*retention);s.data.tokens=Math.round(42800000000*retention);s.data.quality=46+(r.dedup?18:0)+(r.quality?24:0)+(r.pii?6:0);
    s.data.clean=true;s.data.tokenized=false;s.data.version++;s.checkedHash='';s.files['/data/curated/report.json']=JSON.stringify({version:s.data.version,rules:r,documents:s.data.docs,tokens:s.data.tokens,quality:s.data.quality},null,2);result='清洗完成：保留 '+s.data.docs.toLocaleString()+' 篇；质量分 '+s.data.quality+'；已发布 dataset-v'+s.data.version;
   }
   else if(type==='tokenize'){
    this.need(s.data.clean,'缺少已清洗的数据版本');this.need(s.services.storage&&s.services.data&&s.mounts,'数据构建服务不可用');s.data.tokenized=true;
    s.files['/data/tokenized/atlas_text_document.idx']='MEGATRON_INDEX\nvocab=128256\nsequence_length=8192\ndataset_version='+s.data.version;
    s.files['/data/tokenized/atlas_text_document.bin']='[indexed token shards: '+s.data.tokens+' tokens]';result='Tokenizer 128256 已绑定；packing 和 .bin/.idx 分片构建完成';
   }
   else if(type==='sftdata'){this.need(s.services.data&&s.mounts&&s.data.clean,'数据服务或已清洗语料不可用');s.data.sft=true;s.files['/data/sft/training.jsonl']='{"messages":[{"role":"user","content":"解释梯度累积"},{"role":"assistant","content":"累计多个 micro batch 的梯度后执行一次优化器更新。"}]}\n';s.files['/data/preference/manifest.json']='{"pairs":180000,"schema":["prompt","chosen","rejected"]}';result='指令集 1.2M conversations / 偏好集 180K pairs 已冻结；模板与 loss mask 通过检查';}
   else if(type==='savefile'){this.need(typeof arg.path==='string'&&typeof arg.content==='string','无效文件');this.need(arg.content.length<100000,'文件超过 100 KB');s.files[arg.path]=arg.content;if([CONFIG,'/workspace/atlas/train.py','/workspace/atlas/jobs/pretrain.sbatch'].includes(arg.path))s.checkedHash='';result='已保存 '+arg.path;}
   else if(type==='check'){const c=this.config(),inf=this.validate(c),script=s.files['/workspace/atlas/train.py'];this.need(script&&/from megatron\.bridge/.test(script)&&/forward_step_func\s*=/.test(script)&&/(?:pretrain|entry)\(cfg/.test(script),'训练入口检查失败：需要 Megatron 入口、cfg 与 forward_step_func');const batch=s.files['/workspace/atlas/jobs/pretrain.sbatch'];if(this.batchRequest)this.batchRequest();else this.need(batch&&/#SBATCH\s+--nodes=4/.test(batch)&&/#SBATCH\s+--gpus-per-node=8/.test(batch),'Slurm 脚本必须与 world_size=32、4×8 GPU 布局一致');s.checkedHash=this.fingerprint();result='配置通过：world='+inf.gpus+' / TP='+c.tensor_parallel+' / PP='+c.pipeline_parallel+' / CP='+c.context_parallel+' / DP='+inf.dp+'；梯度累积='+inf.accum+'；峰值显存约 '+inf.memory.toFixed(1)+' GiB/GPU';}
   else if(type==='submit'){
    const c=this.config(),inf=this.validate(c);this.need(s.checkedHash===this.fingerprint(),'配置或代码已更改或未检查，请先执行预检');
    this.need(s.services.slurm,'sbatch: Unable to contact slurm controller');this.need(s.data.tokenized,'数据索引不存在：/data/tokenized/atlas_text_document.idx');this.need(s.mounts&&s.services.storage,'训练数据挂载不可用');
    if(!this.schedule)this.need(!s.jobs.some(j=>['PENDING','CONFIGURING','RUNNING'].includes(j.status)),'当前已有训练作业，请先完成或取消');
    if(c.stage!=='pretrain'){this.need(s.data.sft,'后训练数据尚未准备');this.need(s.checkpoints.some(p=>p.complete&&p.stage===(c.stage==='sft'?'pretrain':'sft')),'缺少 '+(c.stage==='sft'?'Base':'SFT')+' 检查点');}
    const parent=c.stage==='pretrain'?null:[...s.checkpoints].reverse().find(p=>p.complete&&p.stage===(c.stage==='sft'?'pretrain':'sft'));
    const j={id:s.nextJob++,stage:c.stage,status:'PENDING',phase:'Resources',age:0,step:0,total:c.train_steps,config:clone(c),info:inf,dataVersion:s.data.version,quality:s.data.quality,loss:0,valLoss:0,points:[],checkpoints:[],parent:parent?.id||null,gpus:inf.gpus,logs:[],started:s.clock,resumeFrom:0};
    s.jobs.unshift(j);this.jobLog(j,'Job accepted; waiting for allocation.');s.files['/workspace/atlas/runs/'+j.id+'/config.yaml']=s.files[CONFIG];result='Submitted batch job '+j.id;
   }
   else if(type==='cancel'){const j=s.jobs.find(j=>j.id===Number(arg.id))||this.runningTrain();this.need(j&&['PENDING','CONFIGURING','RUNNING'].includes(j.status),'没有可停止的作业');if(j.status==='RUNNING'&&arg.save&&s.services.storage&&s.mounts)this.checkpoint(j);j.status='CANCELLED';j.phase='User request';this.jobLog(j,'Job cancelled; resources released.');result='作业 '+j.id+' 已停止，资源已释放';}
   else if(type==='resume'){
    const old=s.jobs.find(j=>j.id===Number(arg.id));this.need(old&&['FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY'].includes(old.status),'请选择已结束的作业');
    const cp=[...s.checkpoints].reverse().find(p=>p.jobId===old.id&&p.complete);this.need(cp,'没有完整检查点，需重新提交');this.need(!s.jobs.some(j=>['PENDING','CONFIGURING','RUNNING'].includes(j.status)),'其他作业正在运行');this.need(s.services.slurm&&s.services.storage&&s.mounts,'调度或检查点存储不可用');
    const j=clone(old);j.id=s.nextJob++;j.status='PENDING';j.phase='Resources';j.age=0;j.step=cp.step;j.resumeFrom=cp.step;j.logs=[];j.started=s.clock;j.checkpoints=[];s.jobs.unshift(j);this.jobLog(j,'Resume from '+cp.id+'; optimizer, RNG and DataLoader state restored.');result='Submitted batch job '+j.id+'，从 step '+cp.step+' 恢复';
   }
   else if(type==='fault'){
    s.incident=arg.kind;if(arg.kind==='network'){s.nodes.find(n=>n.id==='gpu-03').health='NETWORK';result='gpu-03 的 IB 链路中断';}
    else if(arg.kind==='storage'){s.services.storage=false;result='storage 服务已中断';}
    else{s.nodes.find(n=>n.id==='gpu-02').health='DOWN';result='gpu-02 节点失联';}
   }
   else if(type==='repair'){s.nodes.forEach(n=>n.health='OK');if(s.installed.storage)s.services.storage=true;s.incident=null;result='链路、节点和存储检查完成，故障已排除；失败作业需要重新提交或恢复';}
   else if(type==='evaluate'){
    const cp=this.latestCheckpoint();this.need(cp,'没有可评测的完整检查点');this.need(s.services.tracking,'实验追踪服务未启动');this.need(s.services.storage&&s.mounts,'模型存储不可用');
    this.need(!s.evaluation||s.evaluation.status!=='RUNNING','评测运行中');s.evaluation={status:'RUNNING',age:0,checkpoint:cp.id,stage:cp.stage,quality:cp.quality,config:clone(cp.config)};result='评测任务已提交：能力 / 指令跟随 / 长上下文 / 风险';
   }
   else if(type==='export'){
    const cp=this.latestCheckpoint();this.need(cp,'没有完整检查点');this.need(s.services.storage&&s.mounts,'模型存储不可用');const ev=s.evaluation;this.need(ev?.status==='COMPLETED'&&ev.checkpoint===cp.id,'请先完成当前检查点的评测');this.need(ev.passed,'评测未通过发布门槛');
    let m=s.models.find(m=>m.checkpoint===cp.id);if(!m){m={id:'atlas-70b-v'+(s.models.length+1),checkpoint:cp.id,stage:cp.stage,score:ev.overall,quality:cp.quality,created:s.clock,format:'safetensors',dataset:cp.dataVersion};s.models.push(m);s.files['/models/'+m.id+'/config.json']=JSON.stringify({model_type:'llama',hidden_size:8192,num_hidden_layers:80,torch_dtype:'bfloat16',vocab_size:128256},null,2);s.files['/models/'+m.id+'/model.safetensors.index.json']='{"metadata":{"total_size":140000000000},"shards":32}';}result='HF 权重转换与一致性检查完成；注册 '+m.id;
   }
   else if(type==='deploy'){
    const m=s.models.at(-1);this.need(m,'没有可部署的模型版本');this.need(s.services.kubernetes,'Kubernetes control plane 不可用');this.need(!this.runningTrain(),'GPU 正被训练作业占用');this.need(this.computes().every(n=>n.health==='OK'),'节点健康检查失败');this.need(s.services.storage&&s.mounts,'权重存储不可达');
    s.deployment={model:m.id,status:'LOADING',age:0,replicas:1,gpus:8,requests:0,tokens:0,queue:0,ttft:0,config:{maxLen:8192,tp:8}};result='Deployment atlas-api 创建；正在拉取镜像和加载权重';
   }
   else if(type==='undeploy'){this.need(s.deployment,'服务不存在');s.deployment.status='STOPPED';result='atlas-api 已下线；8 GPU 已释放';}
   else if(type==='loadtest'){this.need(s.deployment?.status==='READY','推理服务未就绪');const concurrency=Number(arg.concurrency)||16;this.need(concurrency>=1&&concurrency<=256,'并发范围 1—256');const d=s.deployment;d.queue=Math.max(0,concurrency-24);d.ttft=90+concurrency*9+d.queue*12;d.load={concurrency,throughput:Math.round(Math.min(4200,concurrency*180)/(1+d.queue*.008)),p95:Math.round(380+concurrency*26+d.queue*65),error:concurrency>80?((concurrency-80)*.13).toFixed(1):'0.0'};d.requests+=concurrency;result='压测完成：并发 '+concurrency+'，P95 '+d.load.p95+'ms，失败率 '+d.load.error+'%';}
   else throw Error('未知操作：'+type);
   s.clock+=30;this.event(result);this.emit();return{ok:true,text:result};
  }catch(e){this.event(e.message,'error');this.emit();return{ok:false,text:e.message};}
 }
 runningTrain(){return this.state.jobs.find(j=>j.status==='RUNNING');}
 latestCheckpoint(){return[...this.state.checkpoints].reverse().find(p=>p.complete&&p.final);}
 jobLog(j,text){j.logs.push({t:this.state.clock,text});j.logs=j.logs.slice(-180);this.state.files['/workspace/atlas/runs/'+j.id+'/train.log']=j.logs.map(l=>'['+this.time(l.t)+'] '+l.text).join('\n');}
 checkpoint(j,final=false){if(this.state.checkpoints.some(p=>p.jobId===j.id&&p.step===j.step))return;const cp={id:'ckpt-'+j.id+'-'+String(j.step).padStart(7,'0'),jobId:j.id,step:j.step,stage:j.stage,dataVersion:j.dataVersion,quality:j.quality,complete:true,final,config:clone(j.config),loss:j.loss};this.state.checkpoints.push(cp);j.checkpoints.push(cp.id);this.state.files['/checkpoints/'+cp.id+'/metadata.json']=JSON.stringify({step:cp.step,optimizer:true,rng:true,dataloader:true,shards:32,complete:true,stage:cp.stage},null,2);this.jobLog(j,'Checkpoint committed: '+cp.id+' (32/32 shards, optimizer + RNG + data cursor)');}
 tick(dt=1){
  const s=this.state;if(!s.started)return;const elapsed=dt*s.speed;s.clock+=elapsed*120;
  for(const j of s.jobs){if(!['PENDING','CONFIGURING','RUNNING'].includes(j.status))continue;j.age+=elapsed;
   if(j.status==='PENDING'){
    if(!s.services.slurm){j.phase='ControllerUnavailable';continue;}if(this.computes().some(n=>n.health!=='OK')){j.phase='ReqNodeNotAvail';continue;}
    if(s.deployment&&['READY','LOADING'].includes(s.deployment.status)){j.phase='Resources';continue;}
    if(j.age>=2){j.status='CONFIGURING';j.phase='Container / Rendezvous / NCCL';j.age=0;this.jobLog(j,'Allocated gpu-[01-04], 32x H100; container environment ready.');}
   }else if(j.status==='CONFIGURING'&&j.age>=2){
    if(!s.mounts||!s.services.storage){this.fail(j,'FAILED','DataLoader: /data unavailable');continue;}
    if(j.info.memory>78){this.fail(j,'OUT_OF_MEMORY','CUDA out of memory: required '+j.info.memory.toFixed(1)+' GiB, available 78 GiB');continue;}
    j.status='RUNNING';j.phase='Training';j.age=0;this.jobLog(j,'NCCL initialized; TP='+j.config.tensor_parallel+' PP='+j.config.pipeline_parallel+' DP='+j.info.dp);this.event('作业 '+j.id+' 开始 '+j.stage);
   }else if(j.status==='RUNNING'){
    const bad=this.computes().find(n=>n.health!=='OK');if(bad){this.fail(j,bad.health==='DOWN'?'NODE_FAIL':'FAILED',bad.health==='DOWN'?'Worker lost: '+bad.id:'NCCL watchdog timeout: '+bad.id+' ib0 unavailable');continue;}
    if(!s.services.storage||!s.mounts){this.fail(j,'FAILED','DataLoader I/O error: storage endpoint unreachable');continue;}
    const advance=Math.max(1,Math.round(elapsed*(j.stage==='pretrain'?42:65)));j.step=Math.min(j.total,j.step+advance);
    const p=j.step/j.total,lr=j.config.learning_rate,instability=Math.max(0,Math.log10(lr/.0003));const floor=(100-j.quality)/80+(j.stage==='pretrain'?1.35:j.stage==='sft'?.65:.4);
    j.loss=floor+(j.stage==='pretrain'?3.6:1.8)*Math.exp(-p*5)+Math.sin(j.step*.17+j.config.seed)*.035+instability*p*8;
    j.valLoss=j.loss+.12+Math.max(0,p-.65)*(100-j.quality)/35;j.points.push({step:j.step,loss:j.loss,val:j.valLoss});if(j.points.length>240)j.points.shift();
    this.jobLog(j,'step '+j.step+'/'+j.total+' | loss '+j.loss.toFixed(4)+' | val '+j.valLoss.toFixed(4)+' | lr '+(lr*Math.min(1,j.step/Math.max(1,j.config.warmup_steps))*(.1+.9*(1+Math.cos(p*Math.PI))/2)).toExponential(2)+' | '+Math.round(j.info.tokensPerStep/2.7).toLocaleString()+' tok/s');
    s.spent+=elapsed*32*120/3600*3.2;
    if(instability>1&&p>.2){this.fail(j,'FAILED','Non-finite gradient norm; optimizer step aborted. Check learning rate.');continue;}
    const last=s.checkpoints.filter(c=>c.jobId===j.id).at(-1);if(j.step-(last?.step||j.resumeFrom)>=j.config.checkpoint_interval&&j.step<j.total)this.checkpoint(j);
    if(j.step>=j.total){this.checkpoint(j,true);j.status='COMPLETED';j.phase='ExitCode=0';this.jobLog(j,'Training completed; all ranks exited with code 0.');this.event(j.stage+' 完成：作业 '+j.id);}
   }
  }
  if(s.evaluation?.status==='RUNNING'){
   const e=s.evaluation;e.age+=elapsed;if(!s.services.tracking||!s.services.storage){e.status='FAILED';this.event('评测中断：追踪服务或模型存储不可用','error');}
   else if(e.age>5){const bonus=e.stage==='pretrain'?0:e.stage==='sft'?9:14;e.scores=[{name:'通用知识',score:Math.min(92,54+e.quality*.18+bonus*.25)},{name:'指令跟随',score:Math.min(96,41+e.quality*.2+bonus*1.7)},{name:'代码与推理',score:Math.min(94,46+e.quality*.2+bonus*.7)},{name:'长上下文',score:Math.min(91,50+e.quality*.22+bonus*.3)}];e.overall=e.scores.reduce((a,b)=>a+b.score,0)/4;e.passed=e.overall>=65&&e.config.learning_rate<=.001;e.status='COMPLETED';s.files['/evaluation/'+e.checkpoint+'/results.json']=JSON.stringify(e,null,2);this.event('评测完成：'+e.overall.toFixed(1)+' / 100，'+(e.passed?'通过':'未通过')+'发布门槛');}
  }
  if(s.deployment&&['LOADING','READY'].includes(s.deployment.status)){
   const d=s.deployment;d.age+=elapsed;if(!s.services.kubernetes||!s.services.storage||!s.mounts||this.computes().some(n=>n.health!=='OK')){d.status='FAILED';this.event('atlas-api 就绪检查失败','error');}
   else if(d.status==='LOADING'&&d.age>4){d.status='READY';d.ttft=145;this.event('atlas-api READY · /v1/chat/completions');}
  }this.emit();
 }
 fail(j,status,message){j.status=status;j.phase=message;this.jobLog(j,message);this.event('作业 '+j.id+' '+status+': '+message,'error');}
 time(t=this.state.clock){return'D'+String(1+Math.floor(t/86400)).padStart(2,'0')+' '+String(Math.floor(t/3600)%24+0).padStart(2,'0')+':'+String(Math.floor(t/60)%60).padStart(2,'0');}
 tasks(){const s=this.state;return[
  {id:'infra',title:'节点上电与操作系统部署',done:this.all('os'),app:'infra',command:'labctl power --all → labctl provision'},
  {id:'net',title:'建立网络与 GPU 运行时',done:s.network&&this.all('runtime'),app:'infra',command:'labctl network apply → labctl runtime install'},
  {id:'storage',title:'建设存储并挂载 /data',done:s.services.storage&&s.mounts,app:'infra',command:'安装 storage → 启动 → 挂载'},
  {id:'scheduler',title:'建立 Slurm 调度集群',done:s.services.slurm,app:'infra',command:'安装 slurm → 启动 slurmctld'},
  {id:'platform',title:'启动数据与实验平台',done:s.services.data&&s.services.tracking,app:'infra',command:'安装并启动 data / tracking'},
  {id:'dataset',title:'语料治理、分词与分片',done:s.data.tokenized,app:'data',command:'接入语料 → 清洗 → 构建训练分片'},
  {id:'train',title:'完成 Atlas 70B 预训练',done:s.checkpoints.some(c=>c.stage==='pretrain'&&c.final),app:'jobs',command:'检查配方 → sbatch jobs/pretrain.sbatch'},
  {id:'alignment',title:'指令微调与偏好对齐',done:s.checkpoints.some(c=>c.stage==='dpo'&&c.final),app:'code',command:'准备后训练集 → stage: sft → stage: dpo'},
  {id:'model',title:'评测、导出与注册模型',done:s.models.length>0,app:'models',command:'评测候选 → 转换并注册'},
  {id:'serve',title:'部署推理并通过压力测试',done:s.deployment?.status==='READY'&&!!s.deployment.load,app:'serve',command:'建立服务集群 → 发布模型 → 压测'}];}
 infer(prompt,maxTokens=256){
  const s=this.state,d=s.deployment;this.need(d?.status==='READY','503 Service Unavailable: model not ready');this.need(prompt.trim(),'请输入消息');this.need(prompt.length<12000,'400 context_length_exceeded');
  const model=s.models.find(m=>m.id===d.model),stage=model?.stage||'pretrain';let answer;
  if(/梯度|batch|批次/i.test(prompt))answer='梯度累积会把多个 micro batch 的梯度累加，再执行一次优化器更新。有效批次 = micro batch × 数据并行数 × 累积步数。例如当前 32 GPU、TP=8、PP=2 时，DP=2；micro batch=1、累积 64 步，对应 global batch=128。增大累积步数通常不会像增大 micro batch 那样增加激活显存。';
  else if(/显存|OOM|内存/i.test(prompt))answer='排查显存不足时，先区分模型状态、激活和运行时缓冲。可以减小 micro batch、缩短序列、启用激活重计算，或调整 TP/PP 与优化器分片。修改后先做短程验证，并查看每卡峰值显存。梯度累积可帮助保持有效批次，但不能直接消除权重占用。';
  else if(/NCCL|网络|通信/i.test(prompt))answer='先检查每个 rank 的首个报错，再核对节点连通性、IB 网卡、rendezvous 地址和分配的 GPU。某个节点退出时，其他 rank 可能只表现为 NCCL 超时。应先修复故障节点或链路，再从最后完整检查点恢复。';
  else if(/你|介绍|你好|hello/i.test(prompt))answer='你好，我是 Atlas 70B。当前服务版本 '+d.model+'，已完成'+(stage==='dpo'?'预训练、指令微调与偏好对齐':stage==='sft'?'预训练与指令微调':'基础预训练')+'。你可以测试模型工程、分布式训练、数据治理和部署运维问题。';
  else if(/数据|清洗|去重/i.test(prompt))answer='数据治理需要先登记来源、校验分片并建立质量画像，再执行规范化、精确与近似去重、敏感字段处理和评测去污染。按文档簇或来源切分后冻结数据版本，绑定 Tokenizer，完成 packing 与索引构建。训练应绑定这一份不可变的数据快照。';
  else answer='围绕“'+prompt.trim().slice(0,80)+'”，我会先明确目标与验收条件，然后检查数据、模型配置和资源约束；用固定版本的评测集比较实验，保留代码、数据和检查点血缘。涉及变更时，先做验证，再逐步发布，并准备恢复路径。';
  if(stage==='pretrain')answer=answer.slice(0,Math.round(answer.length*.68))+'…';answer=Array.from(answer).slice(0,Math.max(16,Math.min(1024,Number(maxTokens)||256))).join('');
  d.requests++;d.tokens+=Array.from(answer).length;d.ttft=120+prompt.length*2+d.queue*15;this.event('POST /v1/chat/completions · 200 · '+d.model);this.emit();return{text:answer,model:d.model,tokens:Array.from(answer).length,ttft:d.ttft};
 }
 command(line){
  const s=this.state;line=line.trim();if(!line)return{ok:true,text:''};s.history.push(line);s.history=s.history.slice(-100);let out;
  try{const a=line.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)||[];const cmd=a[0],path=x=>x?.startsWith('/')?x:(s.cwd+'/'+(x||'')).replace(/\/\.\//g,'/');
   if(cmd==='help')out={ok:true,text:'工程命令\n  labctl status | power --all | provision | network apply | runtime install\n  labctl install <storage|slurm|data|tracking|kubernetes|monitor>\n  systemctl start|stop|status <service>\n  mount storage-01:/data /data\n  datactl ingest | clean | tokenize | posttrain\n  python train.py --check\n  sbatch jobs/pretrain.sbatch | squeue | sacct | scancel <jobid>\n  nvidia-smi | sinfo | ssh <node> | exit\n  ls [path] | cd <path> | pwd | cat <file>\n  modelctl evaluate | export | deploy | stop\n  kubectl get pods | logs atlas-api | rollout status\n  curl http://atlas-api/v1/models\n  clear'};
   else if(cmd==='clear'){s.terminal=[];return{ok:true,text:''};}
   else if(cmd==='pwd')out={ok:true,text:s.cwd};
   else if(cmd==='cd'){const p=path(a[1]);const normalized=p.split('/').reduce((acc,x)=>{if(x==='..')acc.pop();else if(x&&x!=='.')acc.push(x);return acc;},[]);const dest='/'+normalized.join('/');this.need(Object.keys(s.files).some(f=>f.startsWith(dest.replace(/\/$/,'')+'/')),'cd: no such directory: '+dest);s.cwd=dest;out={ok:true,text:''};}
   else if(cmd==='ls'){const p=path(a[1]).replace(/\/$/,'')+'/';const list=[...new Set(Object.keys(s.files).filter(f=>f.startsWith(p)).map(f=>f.slice(p.length).split('/')[0]))];this.need(list.length,'ls: cannot access '+p);out={ok:true,text:list.join('   ')};}
   else if(cmd==='cat'){const p=path(a[1]);this.need(Object.hasOwn(s.files,p),'cat: '+p+': No such file');out={ok:true,text:s.files[p]};}
   else if(cmd==='labctl'){if(a[1]==='status')out={ok:true,text:s.nodes.map(n=>n.id.padEnd(13)+' '+(n.power?'ON ':'OFF')+' '+(n.os?'Ubuntu24.04':'NO OS').padEnd(12)+' '+(n.ip||'NO NETWORK').padEnd(14)+' '+n.health).join('\n')};else if(a[1]==='install')out=this.action('install',{service:a[2]});else out=this.action(({power:'power',provision:'provision',network:'network',runtime:'runtime'})[a[1]]||'unknown');}
   else if(cmd==='systemctl'){const service=({slurmctld:'slurm',slurmd:'slurm',nfs:'storage','nfs-server':'storage',containerd:'kubernetes','data-api':'data',mlflow:'tracking',prometheus:'monitor'})[a.at(-1)]||a.at(-1);if(a[1]==='status')out={ok:!!s.services[service],text:service+' '+(s.services[service]?'active (running)':'inactive (dead)')};else out=this.action(a.includes('stop')?'stop':'start',{service});}
   else if(cmd==='mount')out=this.action('mount');else if(cmd==='umount')out=this.action('unmount');
   else if(cmd==='datactl')out=this.action(({ingest:'ingest',clean:'clean',tokenize:'tokenize',posttrain:'sftdata'})[a[1]]||'unknown');
   else if(cmd==='python'&&a.includes('--check'))out=this.action('check');
   else if(cmd==='sbatch')out=this.action('submit');else if(cmd==='scancel')out=this.action('cancel',{id:a[1]});
   else if(cmd==='squeue'||cmd==='sacct')out={ok:true,text:'JOBID  NAME        STATE           STEP       REASON\n'+s.jobs.filter(j=>cmd==='sacct'||['RUNNING','PENDING','CONFIGURING'].includes(j.status)).map(j=>j.id+'  atlas-'+j.stage.padEnd(7)+' '+j.status.padEnd(15)+' '+String(j.step).padEnd(10)+' '+j.phase).join('\n')};
   else if(cmd==='sinfo'){this.need(s.services.slurm,'Unable to contact slurm controller');out={ok:true,text:'PARTITION AVAIL NODES STATE NODELIST\ncompute*  up    4     '+(this.computes().some(n=>n.health!=='OK')?'drain':'idle')+'  gpu-[01-04]'};}
   else if(cmd==='nvidia-smi'){this.need(this.all('runtime'),'nvidia-smi: NVIDIA driver is not installed');const j=this.runningTrain(),d=s.deployment;out={ok:true,text:'NVIDIA-SMI  /  CUDA  /  H100 SXM 80GB\n'+this.computes().map(n=>n.id+'  8 x H100  '+(j?j.info.memory.toFixed(1):d?.status==='READY'?'22.5':'0.7')+' / 80 GiB  '+(j?'96%':d?.status==='READY'?'38%':'0%')+'  '+n.health).join('\n')};}
   else if(cmd==='ssh'){const n=s.nodes.find(n=>n.id===a[1]);this.need(n&&n.os&&n.ip,'ssh: connect to host '+a[1]+': Network is unreachable');s.host=n.id;out={ok:true,text:'Connected to '+n.id+' ('+n.ip+')'};}
   else if(cmd==='exit'){s.host='workstation';out={ok:true,text:'Connection closed.'};}
   else if(cmd==='modelctl')out=this.action(({evaluate:'evaluate',export:'export',deploy:'deploy',stop:'undeploy'})[a[1]]||'unknown');
   else if(cmd==='kubectl'){this.need(s.services.kubernetes,'The connection to the server 10.42.0.10:6443 was refused');const d=s.deployment;out={ok:true,text:a[1]==='logs'?(d?'[vLLM] '+d.status+' '+d.model:'No resources found'):a[1]==='rollout'?(d?.status==='READY'?'deployment atlas-api successfully rolled out':'Waiting for model readiness...'):'NAME          READY  STATUS\n'+(d?'atlas-api-0   '+(d.status==='READY'?'1/1':'0/1')+'    '+d.status:'No resources found')};}
   else if(cmd==='curl'){this.need(s.deployment?.status==='READY','curl: (7) Failed to connect to atlas-api');out={ok:true,text:JSON.stringify({object:'list',data:[{id:s.deployment.model,object:'model'}]},null,2)};}
   else throw Error(cmd+': command not found. 输入 help 查看命令。');
  }catch(e){out={ok:false,text:e.message};}
  s.terminal.push({type:'command',text:line,host:s.host,cwd:s.cwd},{type:out.ok?'output':'error',text:out.text});s.terminal=s.terminal.slice(-250);this.emit();return out;
 }
}
root.FoundryCore={World,initial,parseConfig,hash,CONFIG,DEFAULT_CONFIG};if(typeof module!=='undefined')module.exports=root.FoundryCore;
})(globalThis);
