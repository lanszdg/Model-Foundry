from pathlib import Path
import sys,json,ast,types,math
sys.path.insert(0,str(Path('.test-artifacts/v17-python').resolve()))
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from sympy import isprime
sys.path.insert(0,str(Path('.test-artifacts/v16-python').resolve()))
from tokenizers import Tokenizer
root=Path('.test-artifacts/v12-evidence/deepseek-v4.1-flash-1017/inference')
evidence=json.loads(Path('.test-artifacts/deepseek-reference-v17.json').read_text('utf8'));a=evidence['architecture'];n=a['native'];d=a['hidden_size'];h=a['num_attention_heads'];T=len(evidence['ids']);dtype=torch.float64
flat=torch.tensor(evidence['parameters'],dtype=dtype,requires_grad=True);P={p['name']:flat[p['offset']:p['offset']+p['length']].reshape(p['shape']) for p in evidence['spec']}
class LiftFloat64(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=='float' and not node.args else node
source=ast.parse((root/'model.py').read_text('utf8'));environment=dict(torch=torch,F=F,math=math,linear=lambda x,w,bias=None:F.linear(x,w,bias))
def method(cls,name):
 c=next(c for c in source.body if isinstance(c,ast.ClassDef) and c.name==cls);f=next(f for f in c.body if isinstance(f,ast.FunctionDef) and f.name==name);f=LiftFloat64().visit(f);ast.fix_missing_locations(f);exec(compile(ast.Module(body=[f],type_ignores=[]),'official-model.py','exec'),environment);return environment[name]
def sink(mixes,scale,base,hc,iters,eps):
 pre=torch.sigmoid(mixes[...,:hc]*scale[0]+base[:hc])+eps;post=2*torch.sigmoid(mixes[...,hc:2*hc]*scale[1]+base[hc:2*hc]);comb=(mixes[...,2*hc:]*scale[2]+base[2*hc:]).reshape(*mixes.shape[:-1],hc,hc).softmax(-1)+eps;comb=comb/(comb.sum(-2,keepdim=True)+eps)
 for _ in range(iters-1):comb=comb/(comb.sum(-1,keepdim=True)+eps);comb=comb/(comb.sum(-2,keepdim=True)+eps)
 return pre,post,comb
hc_mixes=method('Block','hc_mixes');hc_pre=method('Block','hc_pre');hc_post=method('Block','hc_post');official_block=method('Block','forward');official_engram=method('Engram','forward');official_gate=method('Gate','forward');official_expert=method('Expert','forward');environment['hc_split_sinkhorn']=sink
hash_env=dict(np=np,torch=torch,nn=nn,dataclass=dataclass,isprime=isprime)
engram_ast=ast.parse((root/'engram.py').read_text('utf8'));definitions=[x for x in engram_ast.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))];exec(compile(ast.Module(body=definitions,type_ignores=[]),'official-engram.py','exec'),hash_env)
backend=Tokenizer.from_file('.test-artifacts/v16-evidence/deepseek-v4.1-flash-1017/tokenizer.json')
class Wrapper:
 backend_tokenizer=backend
 def __len__(self):return backend.get_vocab_size(with_added_tokens=True)
args=types.SimpleNamespace(engram_layer_ids=[1,14],engram_max_ngram_size=4,engram_n_heads=8,engram_vocab_size=16000000,engram_num_embeddings=[384006168,384016682],engram_head_dim=256,engram_compressed_vocab_size=99092,engram_pad_id=2,max_batch_size=1,max_seq_len=64)
layout=hash_env['EngramLayout'].from_args(args);state=hash_env['NgramHashState'](args,layout,Wrapper());hashes=state(torch.tensor([evidence['ids']]),0)[0]
for l in n['engramLayers']:
 src=n['engramSourceLayers'][n['engramLayers'].index(l)];actual=evidence['trace']['layers'][l]['engram']['rows'];assert hashes[:,[1,14].index(src)].tolist()==actual,'independent official n-gram hashes'
def linear(x,key):return x@P[key]
def rms(x,w,eps=None):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+(eps or a['rms_norm_eps']))*w
def rope(x,pos,heads,compressed=False,inverse=False):
 shape=x.shape;x=x.reshape(T if len(pos)==T else len(pos),heads,-1);hd=x.shape[-1];w=n['rotary'];theta=n['compressedTheta'] if compressed else a['rope_theta'];freq=theta**(-torch.arange(0,w,2,dtype=dtype)/w)
 if compressed and n['originalContext']>0:
  corrected=lambda beta:w*math.log(n['originalContext']/(beta*2*math.pi))/(2*math.log(theta));low=max(math.floor(corrected(n['betaFast'])),0);high=min(math.ceil(corrected(n['betaSlow'])),w-1);smooth=1-((torch.arange(w//2,dtype=dtype)-low)/max(high-low,1e-3)).clamp(0,1);freq=freq/n['ropeFactor']*(1-smooth)+freq*smooth
 angles=torch.tensor(pos,dtype=dtype)[:,None]*freq[None,:]*(-1 if inverse else 1);rot=torch.view_as_real(torch.view_as_complex(x[...,-w:].reshape(len(pos),heads,w//2,2).contiguous())*torch.polar(torch.ones_like(angles),angles)[:,None,:]).flatten(-2);return torch.cat([x[...,:hd-w],rot],-1).reshape(shape)
shared={};positions=list(range(T));mask=[i!=0 for i in evidence['ids']]
def attention(x,p,l):
 ratio=n['ratios'][l];qr=rms(linear(x,p+'wq_a.weight'),P[p+'q_norm.weight']);q=rope(linear(qr,p+'wq_b.weight'),positions,h,ratio>0).reshape(T,h,n['headDim']);kv=rope(rms(linear(x,p+'wkv.weight'),P[p+'kv_norm.weight']),positions,1,ratio>0)
 if ratio and l in n['kvSources']:
  value=linear(x,p+'compressor.wkv.weight');count=T//ratio
  if ratio>1:
   gate=linear(x,p+'compressor.wgate.weight');value=(value[:count*ratio].reshape(count,ratio,-1)*gate[:count*ratio].reshape(count,ratio,-1).softmax(1)).sum(1)
  latent=rms(value,P[p+'compressor.norm.weight']);gp=list(range(0,count*ratio,ratio));shared['kv']=rope(latent,gp,1,True);shared['keys']=rope(rms(linear(latent,p+'indexer.wk.weight'),P[p+'indexer.k_norm.weight']),gp,1,True)
 if ratio and l in n['indexSources']:
  iq=rope(linear(qr,p+'indexer.wq_b.weight'),positions,n['indexHeads'],True).reshape(T,n['indexHeads'],n['indexDim']);weights=linear(x,p+'indexer.weights_proj.weight')/math.sqrt(n['indexDim']*n['indexHeads']);score=torch.einsum('thd,jd->thj',iq,shared['keys']).relu();score=(score*weights[:,:,None]).sum(1);choices=[];candidates=[]
  for t in range(T):
   visible=(t+1)//ratio;valid=list(range(visible));values=score[t,:visible].detach().tolist()
   if l==n['candidateSource']:
    block={}
    for j,v in enumerate(values):block[j//n['candidateBlock']]=max(block.get(j//n['candidateBlock'],-math.inf),v)
    if visible:block[(visible-1)//n['candidateBlock']]=math.inf
    keep=sorted(block,key=lambda b:(-block[b],b))[:n['candidateTopBlocks']];candidates.append([j for j in valid if j//n['candidateBlock'] in keep])
   if l>n['candidateSource']:valid=[j for j in valid if j in shared['candidates'][t]]
   choices.append(sorted(sorted(valid,key=lambda j:(-values[j],j))[:n['indexTopK']]))
  shared['choices']=choices
  if l==n['candidateSource']:shared['candidates']=candidates
 if ratio:kv=torch.cat([kv,shared['kv']],0)
 outputs=[]
 for t in range(T):
  selected=[j for j in range(max(0,t-n['window']+1),t+1) if mask[j]]+([T+j for j in shared['choices'][t]] if ratio else [])
  assert selected==evidence['trace']['layers'][l]['selected'][t],'independent sparse selection'
  logits=torch.einsum('hd,jd->hj',q[t],kv[selected])/math.sqrt(n['headDim']);prob=torch.cat([logits,P[p+'attn_sink'][:,None]],-1).softmax(-1)[:,:-1];outputs.append(prob@kv[selected])
 output=rope(torch.stack(outputs).reshape(T,-1),positions,h,ratio>0,True);parts=[];width=output.shape[-1]//n['oGroups']
 for group in range(n['oGroups']):parts.append(output[:,group*width:(group+1)*width]@P[p+'wo_a.weight'][:,group*n['oRank']:(group+1)*n['oRank']])
 return torch.cat(parts,-1)@P[p+'wo_b.weight']
def expert(x,p,weights=None):
 self=types.SimpleNamespace(w1=lambda x:linear(x,p+'w1.weight'),w2=lambda x:linear(x,p+'w2.weight'),w3=lambda x:linear(x,p+'w3.weight'),swiglu_limit=n['swigluLimit']);return official_expert(self,x,weights)
def moe(x,p):
 self=types.SimpleNamespace(weight=P[p+'gate.weight'].T,gate_temp=n['gateTemperature'],score_func='sqrtsoftplus',bias=P[p+'gate.bias'],bias_vl=P[p+'gate.bias_vl'],topk=n['topK'],norm_topk_prob=True,route_scale=n['routeScale']);weights,ids=official_gate(self,x);out=expert(x,p+'shared_experts.')
 for t in range(T):
  for k,e in enumerate(ids[t]):
   add=expert(x[t:t+1],p+'experts.'+str(int(e))+'.',weights[t:t+1,k:k+1]);out=out+F.pad(add,(0,0,t,T-t-1))
 return out
x=F.embedding(torch.tensor(evidence['ids']),P['model.embed_tokens.weight'])[:,None,:].repeat(1,n['hc'],1);pre=torch.zeros(T,n['hc'],dtype=dtype);pre[:,0]=1
for l in range(a['num_hidden_layers']):
 p=f'model.layers.{l}.'
 if l in n['engramLayers']:
  src=n['engramSourceLayers'][n['engramLayers'].index(l)];ids=hashes[:,[1,14].index(src)]%n['engramRows'];self=types.SimpleNamespace(wkv=lambda y:linear(y,p+'engram.wkv.weight'),embed=lambda z:F.embedding(z,P[p+'engram.embed.weight']),q_weight=P[p+'engram.q_weight'],k_weight=P[p+'engram.k_weight'],eps=a['rms_norm_eps'],dim=d,hc_mult=n['hc'],clamp_value=1e-6);x=official_engram(self,x.unsqueeze(0),ids.unsqueeze(0)).squeeze(0)
 self=types.SimpleNamespace(norm_eps=a['rms_norm_eps'],hc_mult=n['hc'],hc_sinkhorn_iters=n['sinkhorn'],hc_eps=n['hcEps'],hc_attn_fn=P[p+'hc_attn_fn'].T,hc_attn_scale=P[p+'hc_attn_scale'],hc_attn_base=P[p+'hc_attn_base'],hc_ffn_fn=P[p+'hc_ffn_fn'].T,hc_ffn_scale=P[p+'hc_ffn_scale'],hc_ffn_base=P[p+'hc_ffn_base'],attn_norm=lambda y:rms(y,P[p+'attn_norm.weight']),ffn_norm=lambda y:rms(y,P[p+'ffn_norm.weight']),attn=lambda y,start:attention(y.squeeze(0),p+'attn.',l).unsqueeze(0),ffn=lambda y,im:moe(y.squeeze(0),p+'ffn.').unsqueeze(0))
 self.hc_mixes=types.MethodType(hc_mixes,self);self.hc_pre=types.MethodType(hc_pre,self);self.hc_post=types.MethodType(hc_post,self);x,pre=official_block(self,x.unsqueeze(0),0,pre.unsqueeze(0),None);x=x.squeeze(0);pre=pre.squeeze(0)
x=(x*pre[:,:,None]).sum(1);logits=rms(x,P['model.norm.weight'])@P['lm_head.weight'];loss=F.cross_entropy(logits,torch.tensor(evidence['labels']),reduction='mean');loss.backward();np.testing.assert_allclose(logits.detach().numpy().flatten(),evidence['logits'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(flat.grad.numpy(),evidence['gradient'],rtol=2e-8,atol=2e-9)
print(json.dumps({'reference':'official methods lifted to float64 + PyTorch CPU autograd','logitMaxError':float(np.max(np.abs(logits.detach().numpy().flatten()-np.array(evidence['logits'])))),'gradientMaxError':float(np.max(np.abs(flat.grad.numpy()-np.array(evidence['gradient'])))),'parameters':flat.numel(),'foldedVocabulary':99092}))

Path(".test-artifacts/deepseek-torch-v17.json").write_text(json.dumps({"logits":logits.detach().flatten().tolist(),"gradient":flat.grad.tolist()}),encoding="utf8")
