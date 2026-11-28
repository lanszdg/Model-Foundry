from pathlib import Path
import json,sys,ast,math
sys.path.insert(0,str(Path('.test-artifacts/v17-python').resolve()))
import torch,numpy as np
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
for family in ['qwen','deepseek']:
 e=json.loads(Path('.test-artifacts/'+family+'-vision-v17.json').read_text('utf8'));a=e['architecture'];c=a['native'].get('visionSpec') or dict(family='deepseek',layers=32,hidden=64,heads=16,intermediate=176,patch=2,tubelet=1,grid=6,merge=3,positions=0,theta=10000);deep=family=='deepseek';d=c['hidden'];P={p['name']:torch.tensor(p['values'],dtype=torch.float64,requires_grad=True).reshape(p['shape']) for p in e['spec']}
 for v in P.values():v.retain_grad()
 def linear(x,p):return F.linear(x,P[p+'.weight'].T,P.get(p+'.bias'))
 def norm(x,p):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*P[p+'.weight'] if deep else F.layer_norm(x,[d],P[p+'.weight'],P[p+'.bias'],1e-6)
 coords=torch.tensor(e['raw']['coordinates']);raw=torch.tensor(e['raw']['raw']).reshape(-1,3*c['tubelet']*c['patch']**2);x=linear(raw,'visual.patch_embed.proj');hd=d//c['heads'];half=hd//2
 if c['positions']:
  side=int(math.sqrt(c['positions']));table=P['visual.pos_embed.weight'].reshape(side,side,d).permute(2,0,1)[None];interp=F.interpolate(table,size=[c['grid'],c['grid']],mode='bilinear',align_corners=True)[0].permute(1,2,0);x=x+interp[coords[:,1],coords[:,2]]
 def rope(x):
  x=x.reshape(-1,c['heads'],hd);phase=torch.cat([coords[:,1:2]*c['theta']**(-2*torch.arange(half//2)/half),coords[:,2:3]*c['theta']**(-2*torch.arange(half//2)/half)],-1)[:,None,:];left,right=x.chunk(2,-1);return torch.cat([left*phase.cos()-right*phase.sin(),left*phase.sin()+right*phase.cos()],-1)
 for l in range(c['layers']):
  p=f'visual.blocks.{l}.';q,k,v=linear(norm(x,p+'norm1'),p+'attn.qkv').chunk(3,-1);q,k=rope(q),rope(k);v=v.reshape(-1,c['heads'],hd);mask=coords[:,0,None]==coords[None,:,0];att=torch.einsum('thd,jhd->htj',q,k)/math.sqrt(hd);att=torch.einsum('htj,jhd->thd',att.masked_fill(~mask[None],-torch.inf).softmax(-1),v).flatten(-2);x=x+linear(att,p+'attn.proj');up=linear(norm(x,p+'norm2'),p+'mlp.fc1')
  if deep:gate,values=up.chunk(2,-1);up=F.silu(gate)*values
  else:up=F.gelu(up,approximate='tanh')
  x=x+linear(up,p+'mlp.fc2')
 x=norm(x,'visual.norm' if deep else 'visual.merger.norm')
 if deep:x=F.unfold(x.reshape(c['grid'],c['grid'],d).permute(2,0,1)[None],c['merge'],stride=c['merge'])[0].T
 else:x=x.reshape(-1,d*c['merge']**2)
 out=linear(F.gelu(linear(x,'visual.merger.fc1')),'visual.merger.fc2');seed=torch.tensor([(i%7-3)/11 for i in range(out.numel())]).reshape(out.shape);(out*seed).sum().backward();error=float((out.detach().flatten()-torch.tensor(e['output'])).abs().max());ge=max(float((P[p['name']].grad.flatten()-torch.tensor(p['gradient'])).abs().max()) for p in e['spec'] if P[p['name']].grad is not None);print(json.dumps({'family':family,'layers':c['layers'],'outputMaxError':error,'gradientMaxError':ge}),flush=True);np.testing.assert_allclose(out.detach().flatten(),e['output'],rtol=1e-8,atol=1e-9)
 for p in e['spec']:
  if P[p['name']].grad is not None:np.testing.assert_allclose(P[p['name']].grad.flatten(),p['gradient'],rtol=1e-6,atol=1e-8,err_msg=p['name'])


 Path(".test-artifacts/"+family+"-vision-torch-v17.json").write_text(json.dumps({"output":out.detach().flatten().tolist(),"gradient":{name:(v.grad.flatten().tolist() if v.grad is not None else [0]*v.numel()) for name,v in P.items()}}),encoding="utf8")
