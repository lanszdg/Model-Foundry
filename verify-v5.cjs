const assert=require('node:assert/strict'),fs=require('node:fs');
const C=require('./src/engine.js');
for(const name of ['systems','runtime','python','corpus','learning','decoder','serving','artifacts'])require('./src/'+name+'.js');
const saved=JSON.parse(fs.readFileSync('.test-artifacts/learning-project.json','utf8')).state;
let checks=0;
function check(value,message){assert.ok(value,message);checks++;}
function equal(a,b,message){assert.deepEqual(a,b,message);checks++;}
function clone(state=saved){const w=new C.World(state);w.state.started=true;w.state.speed=1;return w;}
function wait(w,test,message){for(let i=0;i<2000&&!test();i++)w.tick(.05);check(test(),message);}
function loaded(w){return w.state.deployment.loadedModels[w.state.deployment.model];}
const prompt='decode-check',w=clone(),tok=loaded(w).tokenizer,x=C.encode('x',tok)[0],y=C.encode('y',tok)[0];
equal(C.decode([x,y],tok),'xy','test fixture has independently decodable ASCII tokens');
function synthetic(world,{alternate=false}={}){
 const m=loaded(world),start=C.encode(prompt,m.tokenizer).at(-1)+',2';
 // Controlled logits exercise the real sampler and scheduler; no output plan is injected.
 m.profile.language.rows={[start]:{[x]:4,[y]:alternate?-80:3,2:-80},['*,'+x]:{[x]:alternate?-80:4,[y]:alternate?4:3,2:-80},['*,'+y]:{[x]:4,[y]:alternate?-80:3,2:-80}};
 return m;
}
const m=synthetic(w);
const f1=C.Learning.characterF1;
equal(f1('aaaaaa','a'),2/7,'duplicate characters cannot inflate reference matches');
equal(f1('aab','ab'),.8,'character overlap uses multiset counts');
equal(f1('😀😀','😀'),2/3,'Unicode characters are counted as code points');
equal(f1('',''),1,'two empty answers are equal');
equal(f1('','a'),0,'empty prediction against nonempty reference scores zero');
for(const a of ['aaaa','ababa','测试测试','😀a',''])for(const b of ['a','ab','测试','😀',''])check(f1(a,b)>=0&&f1(a,b)<=1,'F1 always lies in [0, 1]');

let calls=0;const actualStep=w.decodeStep;
w.decodeStep=function(...args){calls++;return actualStep.apply(this,args);};
const r=w.enqueue(prompt,{maxTokens:40,seed:123,timeoutMs:60000});
equal(calls,0,'enqueue performs no token sampling');
check(!Object.hasOwn(r,'plan')&&r.decoder.ids.length===0,'request has decoder state and no future answer');
const initialRng=r.decoder.rng;
w.advanceServing(.001);
equal(calls,0,'prefill consumes no decode sample');
equal(r.decoder.rng,initialRng,'prefill preserves sampling RNG');
wait(w,()=>r.outputIds.length>=3,'tokens are generated during decode');
equal(r.decoder.draws,r.outputIds.length,'one RNG draw occurs per sampled non-EOS token');
check(w.action('request-pause',{id:r.id}).ok,'client pauses active request');
const paused=JSON.stringify(r.decoder),partial=[...r.outputIds];
equal(r.lease,0,'pause releases physical KV lease');
w.advanceServing(.5);
equal(JSON.stringify(r.decoder),paused,'paused request does not advance decoder');
equal(r.status,'PAUSED','scheduler does not silently resume client pause');
const restored=clone(w.state),restoredRequest=restored.state.inference.requests.find(p=>p.id===r.id);
equal(restoredRequest.outputIds,partial,'project reload retains partial output');
equal(restoredRequest.decoder.rng,r.decoder.rng,'project reload retains exact RNG');
check(restored.action('request-resume',{id:r.id}).ok,'saved request resumes');
wait(restored,()=>restoredRequest.status==='COMPLETED','resumed request finishes');
equal(restoredRequest.outputIds,restored.generate(loaded(restored),prompt,{maxTokens:40,seed:123}).ids,'pause and reload preserve uninterrupted sampling result');
check(restoredRequest.events.filter(e=>e.type==='PREFILL_COMPLETE').length===2,'resume recomputes KV before decoding');
equal(restoredRequest.decoder.steps,40,'length-limited request performs exactly forty decode steps');
equal(restoredRequest.finishReason,'length','length completion is recorded');

const cancelled=w.enqueue(prompt,{maxTokens:20});const beforeCalls=calls;
check(w.action('request-cancel',{id:cancelled.id}).ok,'queued request cancels');
w.advanceServing(.2);
equal(calls,beforeCalls,'queued cancellation performs no hidden generation');
equal(cancelled.outputIds.length,0,'cancelled queued request has no sampled tokens');
const timed=w.enqueue(prompt,{timeoutMs:1,maxTokens:20});const beforeTimeout=calls;
w.advanceServing(.1);
equal(timed.status,'TIMED_OUT','deadline expires before generation');
equal(calls,beforeTimeout,'expired request performs no sampling');

const direct=clone(),dm=synthetic(direct,{alternate:true});
dm.profile.memory=[{text:'THIS MUST NOT SEED OUTPUT'}];
const stopped=direct.createDecoder(dm,prompt,{temperature:0,maxTokens:8,stop:['xy']});
equal(direct.decodeStep(dm,stopped).text,'','partial stop prefix is withheld');
const stopResult=direct.decodeStep(dm,stopped);
check(stopResult.done&&stopResult.text===''&&stopped.matchedStop==='xy','stop string spanning token steps is excluded');
equal(stopped.ids,[x,y],'sampling usage retains tokens that matched stop');
equal(stopped.draws,0,'greedy temperature zero does not consume RNG');
const ended=JSON.stringify(stopped);direct.decodeStep(dm,stopped);equal(JSON.stringify(stopped),ended,'finished decoder is immutable');
const greedy=direct.generate(dm,prompt,{temperature:0,maxTokens:3});
equal(greedy.text,'xyx','generation follows prompt-conditioned logits without corpus retrieval');
equal(greedy.ids.length,3,'byte fallback tokens are allowed in sampling');
const eosModel=synthetic(direct);eosModel.profile.language.rows[C.encode(prompt,tok).at(-1)+',2']={2:100};
const eos=direct.enqueue(prompt,{temperature:0});wait(direct,()=>eos.status==='COMPLETED','EOS completes request');
equal(eos.outputIds.length,0,'EOS is not rendered as a completion token');
equal(eos.firstToken,null,'empty completion does not invent a first token');
equal(eos.finishReason,'stop','EOS termination is reported');

const cached=clone();synthetic(cached);const longPrompt=prompt;
const ca=cached.enqueue(longPrompt,{maxTokens:20,seed:7});wait(cached,()=>ca.status==='COMPLETED','uncached incremental request finishes');
const cb=cached.enqueue(longPrompt,{maxTokens:20,seed:7});wait(cached,()=>cb.status==='COMPLETED','second incremental request finishes');
equal(ca.outputIds,cb.outputIds,'cache scheduling cannot alter RNG result');

const legacy=structuredClone(saved);delete legacy.evaluation.protocolVersion;delete legacy.evaluation.id;delete legacy.nextEvaluation;legacy.evaluationHistory=[];legacy.evaluation.passed=true;
for(const model of legacy.models)delete model.evaluation.protocolVersion;
const old=clone(legacy);
check(old.state.evaluationHistory.some(e=>e.protocolVersion==='evaluation-v1'&&e.passed),'historical qualification is archived before protocol invalidation');
check(old.state.evaluation.outdated&&!old.state.evaluation.passed,'old evaluation cannot silently qualify under new protocol');
check(!old.action('export').ok,'outdated evaluation blocks export');
let refused=false;try{old.modelAssets(old.state.models[0].id);}catch{refused=true;}
check(refused,'old report blocks new cold deployment until re-evaluated');
const keepServing=old.enqueue('existing resident request',{maxTokens:1});wait(old,()=>keepServing.status==='COMPLETED','loaded old version can finish while reports are re-evaluated');
check(old.action('evaluate',{checkpoint:old.state.models[0].checkpoint}).ok,'old checkpoint can be re-evaluated');
wait(old,()=>old.state.evaluation.status==='COMPLETED','new protocol completes evaluation');
check(!old.state.evaluation.outdated&&old.state.models[0].evaluation.protocolVersion===C.Learning.evaluationVersion,'re-evaluation refreshes matching registry version');
check(old.state.evaluationHistory.some(e=>e.protocolVersion==='evaluation-v1'&&e.passed),'re-evaluation retains the original historical report');
check(old.state.evaluationHistory.every(e=>JSON.parse(old.state.files[e.path]).id===e.id),'history entries reference durable versioned report files');
check(old.state.evaluationHistory.filter(e=>e.protocolVersion==='evaluation-v1').every(e=>JSON.parse(old.state.files[e.path]).passed===true),'re-evaluation cannot rewrite historical qualification in archived files');
check(old.state.evaluation.samples.every(s=>s.tokenF1>=0&&s.tokenF1<=1),'stored per-task scores have valid bounds');
equal(JSON.parse(old.state.files['/evaluation/'+old.state.evaluation.checkpoint+'/results.json']).status,'COMPLETED','report file records completed state');

const migrating=structuredClone(legacy);migrating.evaluation.status='RUNNING';migrating.evaluation.samples=migrating.evaluation.samples.slice(0,1);migrating.conversions[0].status='RUNNING';delete migrating.conversions[0].evaluation.protocolVersion;
const migration=clone(migrating),modelsBeforeMigration=migration.state.models.length;
check(migration.state.evaluation.status==='FAILED'&&migration.state.evaluation.outdated,'inflight old evaluation stops before mixing protocols');
check(migration.state.conversions[0].status==='FAILED','inflight old conversion cannot register under an obsolete report');
migration.tick(4);
check(migration.state.evaluation.status==='FAILED'&&migration.state.models.length===modelsBeforeMigration,'old inflight work does not silently resume after upgrade');
check(migration.state.evaluationHistory.some(e=>e.status==='FAILED'&&e.error),'interrupted old evaluation retains failure evidence');

const training=clone();training.action('undeploy');training.state.jobs=[];training.state.checkpoints=[];
const recipe=training.state.files['/workspace/atlas/project/recipe.py'];
training.state.files['/workspace/atlas/project/recipe.py']='import torch\n'+recipe.replace('return output_tensor,','output_tensor = torch.nn.functional.cross_entropy(output_tensor, batch["labels"])\n    return output_tensor,');
const invalid=training.action('check');
check(!invalid.ok&&invalid.text.includes('NotImplementedError')&&invalid.text.includes('recipe.py'),'unsupported framework interface reports a source error');
training.state.files['/workspace/atlas/project/recipe.py']=recipe;
training.state.files[C.CONFIG]=C.DEFAULT_CONFIG.replace('seq_length: 8192','seq_length: 1024').replace('train_steps: 2000','train_steps: 2').replace('warmup_steps: 100','warmup_steps: 1').replace('checkpoint_interval: 400','checkpoint_interval: 2');
check(training.action('check').ok&&training.action('submit').ok,'new optimizer recipe submits');
const job=training.state.jobs[0];wait(training,()=>job.status==='COMPLETED','new optimizer run finishes');
equal(job.program.optimizerVersion,'adam-v2','checkpoint source records optimizer semantics');
check(Math.abs(job.modelMetrics.learningRate-job.config.learning_rate*.1)<1e-12,'applied learning rate equals recipe cosine schedule without stage multiplier');
console.log('PASS: '+checks+' incremental execution and independent metric checks.');
