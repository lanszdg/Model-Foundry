/* Stage transitions bind numerical state, evaluation policy and release artifacts. */
(function(root){'use strict';const C=root.FoundryCore,copy=x=>JSON.parse(JSON.stringify(x));
const digest=x=>C.hash(JSON.stringify(x));
function numericModel(j){return j.transformer||j.network;}
function optimizerBinding(j){const model=numericModel(j);return{model:digest(model?.architecture||model?.config),...(C.ModelProtocol?{trainablePolicy:model?.stagePolicy||null}:{}),dtype:j.transformer?.execution?.dtype||j.recipe.dtype,method:j.transformer?.adapter?.method||j.recipe.method||'full',parameters:digest(model?.parameters||model?.weights),adapter:j.transformer?.adapter?digest(j.transformer.adapter.tensors.map(p=>({name:p.name,A:p.A,B:p.B}))):null};}
function prepare(j,parent){if(!numericModel(j))throw Error('Numerical checkpoint missing');const policy=j.recipe.optimizerPolicy||'reset';if(!['reset','continue'].includes(policy))throw Error('optimizerPolicy must be reset or continue');const current=optimizerBinding(j),previous=parent.optimizerBinding;
 if(policy==='continue'){
  if(!previous||JSON.stringify(previous)!==JSON.stringify(current))throw Error('Optimizer continuation rejected: weights, architecture, precision or adaptation changed');
  const opt=j.optimizer?.transformer;if(j.transformer&&!j.transformer.adapter&&opt?.master){if(opt.master.length!==j.transformer.parameters.length||opt.master.some((v,i)=>C.RankNumerics.cast(v,current.dtype)!==j.transformer.parameters[i]))throw Error('Optimizer master does not describe the restored parameters');}
  if(j.transformer?.execution&&parent.transformer?.execution)j.transformer.execution.lossScale=parent.transformer.execution.lossScale||128;
  j.optimizerTransition={policy:'continue',from:previous.parameters};
 }else{
  j.optimizer={};if(j.transformer?.execution){j.transformer.execution.lossScale=j.recipe.dtype==='fp16'?128:1;j.transformer.execution.foundInf=false;}if(j.network?.optimizer){j.network.optimizer={m:j.network.parameters.map(()=>0),v:j.network.parameters.map(()=>0),t:0};}if(j.network?.master){delete j.network.master;j.network.moments=j.network.weights.map(w=>({m:w.map(()=>0),v:w.map(()=>0)}));j.network.step=0;}
  if(j.transformer?.adapter)for(const p of j.transformer.adapter.tensors){p.mA=p.A.map(()=>0);p.vA=p.A.map(()=>0);p.mB=p.B.map(()=>0);p.vB=p.B.map(()=>0);}if(j.transformer?.adapter)j.transformer.adapter.step=0;
  if(j.transformer?.adapter)j.transformer.adapter.amp={scale:j.recipe.dtype==='fp16'?128:1,good:0,skipped:0};
  j.optimizerTransition={policy:'reset',reason:previous&&JSON.stringify(previous)!==JSON.stringify(current)?'weights / method / precision changed':'new stage optimizer'};
 }
 j.optimizerBinding=current;return j;
}
function evaluationSpec(j){const model=numericModel(j);return{schema:1,...(C.ModelProtocol?{inferenceContract:C.ModelProtocol.inferenceSpec(j)}:{}),protocol:j.network?.kind?.startsWith('domain-')?'domain-eval-v3':'domain-eval-v2',gates:{maxValidationLoss:j.recipe.maxValidationLoss??10,maxRolloutDistance:j.recipe.maxRolloutDistance??.5,requireTaskSuccess:!!j.recipe.requireTaskSuccess,maxObjectDistance:j.recipe.maxObjectDistance??.08,maxCollisions:j.recipe.maxCollisions??0},dtype:j.transformer?.execution?.dtype||j.recipe.dtype,architecture:model?.architecture||model?.config||null,processor:model?.processor||{type:j.transformer?.architecture.native?.vision?'native-rgb-prefix':'byte-text-v1'},normalization:model?.normalization||null,tokenizer:j.transformer?.tokenizerHash||null,quantization:j.transformer?.quantization||null,executionRole:j.network?.family||'causal-decoder',inputProtocol:j.executionPlan?.protocol||'legacy-project',evaluationSequenceLength:j.executionPlan?.sequenceLength??null};}
function binding(j,base){return{...base,capacitySpecDigest:j.budget?.specDigest||null,evaluationSpecDigest:digest(evaluationSpec(j))};}
function assertRelease(j,report,expected){if(!report?.passed)throw Error('Candidate did not pass its frozen evaluation policy');if(JSON.stringify(report.binding)!==JSON.stringify(expected))throw Error('Evaluation policy, execution precision, processor, model or data changed; evaluate this candidate again');}
C.ProjectContract={prepare,optimizerBinding,evaluationSpec,binding,assertRelease};
})(globalThis);
