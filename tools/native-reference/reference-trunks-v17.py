from pathlib import Path
import sys,ast,json,types,math
sys.path.insert(0,str(Path('.test-artifacts/v17-python').resolve()))
import torch,numpy as np
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
base=Path('.test-artifacts/v12-evidence/sources/transformers/src/transformers/models')
class Lift(ast.NodeTransformer):
 def visit_Attribute(self,node):
  node=self.generic_visit(node)
  if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float32':node.attr='float64'
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=='float' and not node.args else node
def load(source,cls,name,env):
 tree=ast.parse(source.read_text('utf8'));body=tree.body if cls is None else next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls).body
 f=next(x for x in body if isinstance(x,ast.FunctionDef) and x.name==name);f.decorator_list=[];f=Lift().visit(f);ast.fix_missing_locations(f);exec(compile(ast.Module(body=[f],type_ignores=[]),'official-'+source.name,'exec'),env);return env[name]
for modelid in ['glm-5.3','qwen3.8-27b']:
 evidence=json.loads(Path('.test-artifacts/'+modelid+'-reference-v17.json').read_text('utf8'));a=evidence['architecture'];n=a['native'];d=a['hidden_size'];heads=a['num_attention_heads'];hd=n['headDim'];ts=len(evidence['ids']);glm=modelid=='glm-5.3';flat=torch.tensor(evidence['parameters'],requires_grad=True);P={p['name']:flat[p['offset']:p['offset']+p['length']].reshape(p['shape']) for p in evidence['spec']};env={'torch':torch,'F':F,'nn':torch.nn,'math':math}
 qsrc=base/'qwen3_5/modeling_qwen3_5.py';gsrc=base/'glm_moe_dsa/modeling_glm_moe_dsa.py'
 load(qsrc,None,'l2norm',env);delta=load(qsrc,None,'torch_recurrent_gated_delta_rule',env);router=load(gsrc,'GlmMoeDsaTopkRouter','forward',env)
 def linear(x,p):return x@P[p]
 def rms(x,w,eps=1e-6,offset=False):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*(w+1 if offset else w)
 def rope(x,width,split=False,offset=0):
  freq=a['rope_theta']**(-torch.arange(0,width,2)/width);angle=torch.arange(ts)[:,None]*freq;co=angle.cos()[:,None,:];si=angle.sin()[:,None,:];r=x[...,offset:offset+width]
  if split:left,right=r.chunk(2,-1);r=torch.cat([left*co-right*si,left*si+right*co],-1)
  else:left,right=r[...,::2],r[...,1::2];r=torch.stack([left*co-right*si,left*si+right*co],-1).flatten(-2)
  return torch.cat([x[...,:offset],r,x[...,offset+width:]],-1)
 def ff(x,p):return (F.silu(linear(x,p+'gate_proj.weight'))*linear(x,p+'up_proj.weight'))@P[p+'down_proj.weight']
 def attend(q,k,v,selected=None):
  score=torch.einsum('thd,jhd->htj',q,k)/math.sqrt(hd);mask=torch.ones(ts,ts,dtype=torch.bool).tril()
  if selected is not None:
   sparse=torch.zeros_like(mask)
   for t,chosen in enumerate(selected):sparse[t,chosen]=True
   mask &=sparse
  return torch.einsum('htj,jhd->thd',score.masked_fill(~mask[None],-torch.inf).softmax(-1),v).flatten(-2)
 x=F.embedding(torch.tensor(evidence['ids']),P['model.embed_tokens.weight']);selected=None
 for l in range(a['num_hidden_layers']):
  p=f'model.layers.{l}.';value=rms(x,P[p+'input_layernorm.weight'],a['rms_norm_eps'],not glm)
  if glm:
   ap=p+'self_attn.';low=rms(linear(value,ap+'q_a_proj.weight'),P[ap+'q_a_layernorm.weight']);q=rope(linear(low,ap+'q_b_proj.weight').reshape(ts,heads,hd),n['rotary'],offset=hd-n['rotary']);kv=linear(value,ap+'kv_a_proj_with_mqa.weight');latent=rms(kv[:,:n['kvRank']],P[ap+'kv_a_layernorm.weight']);kr=rope(kv[:,None,n['kvRank']:],n['rotary']);expanded=linear(latent,ap+'kv_b_proj.weight').reshape(ts,heads,2*hd-n['rotary']);kn,v=expanded.split([hd-n['rotary'],hd],-1);k=torch.cat([kn,kr.repeat(1,heads,1)],-1)
   if n['indexerTypes'][l]=='full':
    with torch.no_grad():
     iq=rope(linear(low,ap+'indexer.wq_b.weight').reshape(ts,n['indexHeads'],n['indexDim']),n['rotary']);ik=rope(F.layer_norm(linear(value,ap+'indexer.wk.weight'),[n['indexDim']],P[ap+'indexer.k_norm.weight'],P[ap+'indexer.k_norm.bias'],1e-6)[:,None,:],n['rotary'])[:,0,:];iw=linear(value,ap+'indexer.weights_proj.weight')/math.sqrt(n['indexHeads']*n['indexDim']);score=(torch.einsum('thd,jd->thj',iq,ik).relu()*iw[:,:,None]).sum(1);selected=[sorted(range(t+1),key=lambda j:(-float(score[t,j]),j))[:n['indexTopK']] for t in range(ts)]
   out=attend(q,k,v,selected)@P[ap+'o_proj.weight']
  elif (l+1)%n['fullAttentionInterval']==0:
   ap=p+'self_attn.';both=linear(value,ap+'q_proj.weight').reshape(ts,heads,2*hd);q,gate=both.chunk(2,-1);q=rope(rms(q,P[ap+'q_norm.weight'],a['rms_norm_eps'],True),n['rotary'],True);k=rope(rms(linear(value,ap+'k_proj.weight').reshape(ts,a['num_key_value_heads'],hd),P[ap+'k_norm.weight'],a['rms_norm_eps'],True),n['rotary'],True);v=linear(value,ap+'v_proj.weight').reshape(ts,a['num_key_value_heads'],hd);out=(attend(q,k.repeat_interleave(heads//a['num_key_value_heads'],1),v.repeat_interleave(heads//a['num_key_value_heads'],1))*gate.flatten(-2).sigmoid())@P[ap+'o_proj.weight']
  else:
   ap=p+'linear_attn.';kh=n['linearKeyHeads'];vh=n['linearValueHeads'];width=n['linearHeadDim'];mixed=linear(value,ap+'in_proj_qkv.weight');mixed=F.silu(F.conv1d(mixed.T[None],P[ap+'conv1d.weight'][:,None,:],padding=n['conv']-1,groups=mixed.shape[-1])[0,:,:ts].T);q,k,v=mixed.split([kh*width,kh*width,vh*width],-1);decay=-P[ap+'A_log'].exp()*F.softplus(linear(value,ap+'in_proj_a.weight')+P[ap+'dt_bias']);beta=linear(value,ap+'in_proj_b.weight').sigmoid();out,state=delta(q.reshape(1,ts,kh,width),k.reshape(1,ts,kh,width),v.reshape(1,ts,vh,width),decay[None],beta[None],output_final_state=True,use_qk_l2norm_in_kernel=True);out=(rms(out[0],P[ap+'norm.weight'],a['rms_norm_eps'])*F.silu(linear(value,ap+'in_proj_z.weight').reshape(ts,vh,width))).flatten(-2)@P[ap+'out_proj.weight']
  x=x+out;value=rms(x,P[p+'post_attention_layernorm.weight'],a['rms_norm_eps'],not glm)
  if glm and l>=n['firstDense']:
   mp=p+'mlp.';self=types.SimpleNamespace(hidden_dim=d,weight=P[mp+'gate.weight'].T,e_score_correction_bias=P[mp+'gate.e_score_correction_bias'],num_group=n['groups'],num_experts=n['experts'],topk_group=n['topGroups'],top_k=n['topK'],norm_topk_prob=True,routed_scaling_factor=n['routedScaling']);_,weights,choices=router(self,value);out=ff(value,mp+'shared_experts.')
   for t in range(ts):
    for i,e in enumerate(choices[t]):out=out+F.pad(ff(value[t:t+1],mp+'experts.'+str(int(e))+'.')*weights[t,i],(0,0,t,ts-t-1))
  else:out=ff(value,p+'mlp.')
  x=x+out
 logits=rms(x,P['model.norm.weight'],a['rms_norm_eps'],not glm)@P['lm_head.weight'];loss=F.cross_entropy(logits,torch.tensor(evidence['labels']));loss.backward();err=float((logits.detach().flatten()-torch.tensor(evidence['logits'])).abs().max());ge=float((flat.grad-torch.tensor(evidence['gradient'])).abs().max());print(json.dumps({'model':modelid,'logitMaxError':err,'gradientMaxError':ge,'parameters':flat.numel()}),flush=True);np.testing.assert_allclose(logits.detach().flatten(),evidence['logits'],rtol=1e-8,atol=1e-8);np.testing.assert_allclose(flat.grad,evidence['gradient'],rtol=1e-6,atol=1e-7)

 Path(".test-artifacts/"+modelid+"-torch-v17.json").write_text(json.dumps({"logits":logits.detach().flatten().tolist(),"gradient":flat.grad.tolist()}),encoding="utf8")
