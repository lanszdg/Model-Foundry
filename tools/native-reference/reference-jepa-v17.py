from pathlib import Path
import sys,json,math
sys.path.insert(0,str(Path('.test-artifacts/v17-python').resolve()))
import torch,numpy as np
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
e=json.loads(Path('.test-artifacts/jepa-reference-v17.json').read_text('utf8'));c=e['config'];flat=torch.tensor(e['parameters'],requires_grad=True);teacher=torch.tensor(e['targetParameters']);P={p['name']:flat[p['offset']:p['offset']+p['length']].reshape(p['shape']) for p in e['spec']};TP={p['name']:teacher[p['offset']:p['offset']+p['length']].reshape(p['shape']) for p in e['spec']};grid=c['image']//c['patch'];frames=e['row']['frames'][:c['frames']];pixels=torch.tensor([f['pixels'] for f in frames]).reshape(len(frames),frames[0]['height'],frames[0]['width'],3).permute(3,0,1,2)[None]*2-1
def linear(x,p,P):return F.linear(x,P[p+'.weight'].T,P[p+'.bias'])
def norm(x,p,P):return F.layer_norm(x,[x.shape[-1]],P[p+'.weight'],P[p+'.bias'],1e-6)
def rope(x,heads,ids,interpolate):
 hd=x.shape[-1]//heads;axis_width=2*((hd//3)//2);x=x.reshape(len(ids),heads,hd);parts=[];coords=torch.tensor([[i//c['tokens'],i%c['tokens']//grid,i%grid] for i in ids])
 for axis in range(3):
  pos=coords[:,axis]*(15/(grid-1) if interpolate and axis else 1);angle=pos[:,None]*10000**(-torch.arange(0,axis_width,2)/axis_width);r=x[...,axis*axis_width:(axis+1)*axis_width].reshape(len(ids),heads,-1,2);left,right=r.unbind(-1);parts.append(torch.stack([left*angle.cos()[:,None]-right*angle.sin()[:,None],left*angle.sin()[:,None]+right*angle.cos()[:,None]],-1).flatten(-2))
 return torch.cat([*parts,x[...,3*axis_width:]],-1)
def block(x,p,P,heads,ids,interp):
 d=x.shape[-1];q,k,v=linear(norm(x,p+'norm1',P),p+'attn.qkv',P).chunk(3,-1);q,k=rope(q,heads,ids,interp),rope(k,heads,ids,interp);v=v.reshape(len(ids),heads,d//heads);out=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None])[0].transpose(0,1).flatten(-2);x=x+linear(out,p+'attn.proj',P);return x+linear(F.gelu(linear(norm(x,p+'norm2',P),p+'mlp.fc1',P)),p+'mlp.fc2',P)
def encode(P,ids):
 weight=P['encoder.patch_embed.proj.weight'].T.reshape(c['hidden'],3,c['tubelet'],c['patch'],c['patch']);x=F.conv3d(pixels,weight,P['encoder.patch_embed.proj.bias'],stride=(c['tubelet'],c['patch'],c['patch'])).flatten(2).transpose(1,2)[0,ids]+P['encoder.video_mod_embed'];levels=[]
 for l in range(c['encoderLayers']):
  x=block(x,f'encoder.blocks.{l}.',P,c['heads'],ids,True)
  if l in c['levels']:levels.append(norm(x,'encoder.norms_block.'+str(c['levels'].index(l)),P))
 return levels
visible=e['visible'];targets=e['targets'];online=encode(P,visible)
with torch.no_grad():target=encode(TP,list(range(c['tokens']*c['frames']//c['tubelet'])))
x=linear(F.gelu(linear(torch.cat(online,-1),'world.predictor_embed.0',P)),'world.predictor_embed.2',P);masks=P['world.mask_tokens'][1].repeat(len(targets),1);ids=visible+targets;order=np.argsort(ids).tolist();ordered=[ids[i] for i in order];x=torch.cat([x,masks],0)[order]+P['world.video_mod_embed']
for l in range(c['predictorLayers']):x=block(x,f'world.predictor_blocks.{l}.',P,c['predictorHeads'],ordered,False)
x=norm(x,'world.predictor_norm',P);prediction=linear(x[[ordered.index(i) for i in targets]],'world.predictor_proj',P);context=linear(x[[ordered.index(i) for i in visible]],'world.predictor_proj_context',P);loss=F.l1_loss(prediction,torch.cat(target,-1)[targets])+.25*F.l1_loss(context,torch.cat(target,-1)[visible]);loss.backward();err=float((prediction.detach().flatten()-torch.tensor(e['output']['latents'])).abs().max());ge=float((flat.grad-torch.tensor(e['gradient'])).abs().max());print(json.dumps({'model':'V-JEPA2.1','loss':float(loss.detach()),'lossError':abs(float(loss.detach())-e['loss']),'outputMaxError':err,'gradientMaxError':ge}));np.testing.assert_allclose(prediction.detach().flatten(),e['output']['latents'],rtol=1e-8,atol=1e-9);np.testing.assert_allclose(flat.grad,e['gradient'],rtol=1e-6,atol=1e-8)

Path(".test-artifacts/jepa-torch-v17.json").write_text(json.dumps({"output":{"latents":prediction.detach().flatten().tolist()},"gradient":flat.grad.tolist(),"loss":float(loss.detach())}),encoding="utf8")
