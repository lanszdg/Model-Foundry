/* Serializable, one-step token sampling shared by evaluation and serving. */
(function(root){'use strict';
const C=root.FoundryCore,W=C.World,VERSION='decode-v2';
function visibleText(state,tok,final=false){
 let text=C.decode(state.ids,tok),end=text.length,stop=null;
 for(const value of state.stop){const at=text.indexOf(value);if(at>=0&&at<end){end=at;stop=value;}}
 if(stop!==null){state.done=true;state.finishReason='stop';state.matchedStop=stop;return text.slice(0,end);}
 if(!final)for(const value of state.stop)for(let n=1;n<value.length&&n<=text.length;n++)if(text.endsWith(value.slice(0,n)))end=Math.min(end,text.length-n);
 return text.slice(0,end);
}
W.prototype.createDecoder=function(model,prompt,options={}){
 this.need(model.profile?.language&&model.tokenizer?.type==='unicode-bpe','Model has no trained Token assets');
 const maxTokens=Number(options.maxTokens??256),temperature=Number(options.temperature??.7),topP=Number(options.topP??.9),penalty=Number(options.repetitionPenalty??1),stop=typeof options.stop==='string'?[options.stop]:options.stop||[];
 this.need(Number.isInteger(maxTokens)&&maxTokens>=1&&maxTokens<=1024,'max_tokens range 1—1024');
 this.need(Number.isFinite(temperature)&&temperature>=0&&temperature<=2,'temperature range 0—2');
 this.need(Number.isFinite(topP)&&topP>0&&topP<=1,'top_p range (0, 1]');
 this.need(Number.isFinite(penalty)&&penalty>=1&&penalty<=2,'repetition_penalty range 1—2');
 this.need(Array.isArray(stop)&&stop.length<=4&&stop.every(x=>typeof x==='string'&&x.length>0&&x.length<=200),'stop requires 1—4 non-empty strings');
 const promptIds=[3,...C.encode(String(prompt),model.tokenizer),2];
 return{version:VERSION,promptIds,context:promptIds.slice(-2),ids:[],counts:{},rng:Number(options.seed??42)>>>0,seed:Number(options.seed??42)>>>0,draws:0,steps:0,maxTokens,temperature,topP,repetitionPenalty:penalty,stop:[...stop],done:false,finishReason:null,text:''};
};
W.prototype.decodeStep=function(model,state){
 this.need(state?.version===VERSION,'Unsupported decoder state');
 if(state.done)return{id:null,done:true,text:state.text,finishReason:state.finishReason};
 const rows=model.profile.language.rows,key=state.context.join(','),row=rows[key]||rows['*,'+state.context.at(-1)]||{2:0};
 const candidates=Object.keys(row).map(Number).filter(id=>id===2||id>=4).map(id=>{let logit=Number(row[id]);if(state.counts[id])logit=logit>0?logit/state.repetitionPenalty:logit*state.repetitionPenalty;return{id,logit};}).filter(x=>Number.isFinite(x.logit)).sort((a,b)=>b.logit-a.logit||a.id-b.id);
 this.need(candidates.length>0,'No finite token logits');
 let id=candidates[0].id;
 if(state.temperature>0){const peak=candidates[0].logit;for(const c of candidates)c.mass=Math.exp(Math.max(-80,(c.logit-peak)/state.temperature));const sum=candidates.reduce((a,c)=>a+c.mass,0);let acc=0;const keep=[];for(const c of candidates){keep.push(c);acc+=c.mass/sum;if(acc>=state.topP)break;}
  state.rng=(Math.imul(1664525,state.rng)+1013904223)>>>0;state.draws++;let pick=state.rng/4294967296*keep.reduce((a,c)=>a+c.mass,0);id=keep.at(-1).id;for(const c of keep){pick-=c.mass;if(pick<=0){id=c.id;break;}}
 }
 state.steps++;
 if(id===2){state.done=true;state.finishReason='stop';}
 else{state.ids.push(id);state.counts[id]=(state.counts[id]||0)+1;state.context=[state.context.at(-1),id];if(state.ids.length>=state.maxTokens){state.done=true;state.finishReason='length';}}
 state.text=visibleText(state,model.tokenizer,state.done);
 return{id:id===2?null:id,done:state.done,text:state.text,finishReason:state.finishReason};
};
W.prototype.generate=function(model,prompt,options={}){const state=this.createDecoder(model,prompt,options);while(!state.done)this.decodeStep(model,state);return{text:state.text,ids:[...state.ids],finishReason:state.finishReason,seed:state.seed,decodeSteps:state.steps};};
C.Decoder={version:VERSION};
})(globalThis);
