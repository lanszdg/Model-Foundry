"""Independent NumPy forward implementation; does not import runtime JS."""
import json, math
from pathlib import Path
import numpy as np

fixtures=json.loads(Path('.test-artifacts/native-reference.json').read_text('utf8'))
def sigmoid(x): return 1/(1+np.exp(-x))
def silu(x): return x*sigmoid(x)
def norm(x,w,eps=1e-5,zero=False): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*(w+1 if zero else w)
def ln(x,w,b):
    z=x-x.mean(axis=-1,keepdims=True)
    return z/np.sqrt((z*z).mean(axis=-1,keepdims=True)+1e-6)*w+b
def rotary(x,h,pos,theta,width,offset=0,axes=None):
    out=x.copy().reshape(len(x),h,-1)
    for t in range(len(x)):
        for j in range(0,width,2):
            position=axes[t,(j//2)%3] if axes is not None else pos[t]
            angle=position/theta**(j/width)
            co,si=np.cos(angle),np.sin(angle)
            left,right=out[t,:,offset+j].copy(),out[t,:,offset+j+1].copy()
            out[t,:,offset+j]=left*co-right*si
            out[t,:,offset+j+1]=left*si+right*co
    return out.reshape(x.shape)
def attention(q,k,v,h,chosen=None,bidir=False):
    length=len(q);q=q.reshape(length,h,-1);k=k.reshape(length,h,-1);v=v.reshape(length,h,-1)
    result=np.zeros_like(v)
    for t in range(length):
        allowed=np.arange(length) if bidir else np.arange(t+1)
        if chosen is not None: allowed=np.array(chosen[t])
        for head in range(h):
            score=k[allowed,head]@q[t,head]/np.sqrt(q.shape[-1]);p=np.exp(score-score.max());p/=p.sum()
            result[t,head]=p@v[allowed,head]
    return result.reshape(length,-1)
def recurrent(q,k,v,decay,beta,h,normalize=False):
    length=len(q);d=q.shape[-1]//h
    q=q.reshape(length,h,d);k=k.reshape(length,h,d);v=v.reshape(length,h,d)
    if normalize:
        q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-6)
        k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
    state=np.zeros((h,d,d));out=[]
    for t in range(length):
        state=state*decay[t,:,None,None]
        residual=v[t]-np.einsum('hi,hij->hj',k[t],state)
        state+=beta[t,:,None,None]*k[t,:, :,None]*residual[:,None,:]
        out.append(np.einsum('hi,hij->hj',q[t]/np.sqrt(d),state))
    return np.array(out).reshape(length,-1),state

for fixture in fixtures:
    if fixture['family']=='delta':
        q=np.array(fixture['q']).reshape(3,2);k=np.array(fixture['k']).reshape(3,2);v=np.array(fixture['v']).reshape(3,2)
        out,state=recurrent(q,k,v,np.array(fixture['decay']).reshape(3,1),np.array(fixture['beta']).reshape(3,1),1)
        np.testing.assert_allclose(out.ravel(),fixture['output'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(state.ravel(),fixture['state'],rtol=1e-12,atol=1e-12)
        continue
    a=fixture['model']['architecture'];n=a['native'];d=a['hidden_size'];h=a['num_attention_heads'];hd=d//h;r=n['rotary'];theta=a['rope_theta']
    weights={k:np.array(v['data']).reshape(v['shape']) for k,v in fixture['state_dict'].items()}
    def linear(x,name): return x@weights[name]
    def ffn(x,p): return linear(silu(linear(x,p+'gate_proj.weight'))*linear(x,p+'up_proj.weight'),p+'down_proj.weight')
    ids=fixture['tokens'];x=weights['model.embed_tokens.weight'][ids].copy();axes=None;pos=np.arange(len(ids))
    if fixture['media'] is not None:
        images=[]
        for image in fixture['media'][:2]:
            rgb=np.array(image['pixels']).reshape(image['height'],image['width'],3)
            yi=np.minimum(image['height']-1,((np.arange(8)+.5)*image['height']/8).astype(int));xi=np.minimum(image['width']-1,((np.arange(8)+.5)*image['width']/8).astype(int))
            images.append((rgb[yi[:,None],xi[None,:]]-.5)/.5)
        if len(images)==1: images.append(images[0])
        patches=[]
        for yy in range(4):
            for xx in range(4): patches.append(np.concatenate([im[yy*2:yy*2+2,xx*2:xx*2+2].ravel() for im in images]))
        v=linear(np.array(patches),'visual.patch_embed.proj.weight')+weights['visual.pos_embed.weight']
        z=ln(v,weights['visual.blocks.0.norm1.weight'],weights['visual.blocks.0.norm1.bias'])
        v+=linear(attention(linear(z,'visual.blocks.0.attn.q_proj.weight'),linear(z,'visual.blocks.0.attn.k_proj.weight'),linear(z,'visual.blocks.0.attn.v_proj.weight'),h,bidir=True),'visual.blocks.0.attn.proj.weight')
        z=ln(v,weights['visual.blocks.0.norm2.weight'],weights['visual.blocks.0.norm2.bias']);z=linear(z,'visual.blocks.0.mlp.fc1.weight')
        z=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)));v+=linear(z,'visual.blocks.0.mlp.fc2.weight')
        v=ln(v,weights['visual.merger.ln_q.weight'],weights['visual.merger.ln_q.bias']).reshape(4,4,d)
        merged=np.array([v[yy*2:yy*2+2,xx*2:xx*2+2].ravel() for yy in range(2) for xx in range(2)])
        x[:4]=linear(silu(linear(merged,'visual.merger.fc1.weight')),'visual.merger.fc2.weight')
        axes=np.array([[0,t//2,t%2] if t<4 else [t-2]*3 for t in range(len(ids))])
    for layer in range(a['num_hidden_layers']):
        p=f'model.layers.{layer}.';z=norm(x,weights[p+'input_layernorm.weight'],a['rms_norm_eps'],n['family']=='qwen3_5')
        if n['family']=='glm_moe_dsa':
            qlow=norm(linear(z,p+'self_attn.q_a_proj.weight'),weights[p+'self_attn.q_a_layernorm.weight'],a['rms_norm_eps'])
            q=rotary(linear(qlow,p+'self_attn.q_b_proj.weight'),h,pos,theta,r,hd-r)
            kv=linear(z,p+'self_attn.kv_a_proj_with_mqa.weight');latent=norm(kv[:,:n['latent']],weights[p+'self_attn.kv_a_layernorm.weight'],a['rms_norm_eps'])
            expanded=linear(latent,p+'self_attn.kv_b_proj.weight').reshape(len(ids),h,2*hd-r)
            kr=rotary(kv[:,n['latent']:],1,pos,theta,r)
            key=np.concatenate([expanded[:,:,:hd-r],np.repeat(kr[:,None,:],h,axis=1)],axis=-1).reshape(len(ids),d)
            value=expanded[:,:,hd-r:].reshape(len(ids),d)
            iq=rotary(linear(qlow,p+'self_attn.indexer.wq_b.weight'),h,pos,theta,r).reshape(len(ids),h,hd)
            ik=rotary(ln(linear(z,p+'self_attn.indexer.wk.weight'),weights[p+'self_attn.indexer.k_norm.weight'],weights[p+'self_attn.indexer.k_norm.bias']),1,pos,theta,r)
            iw=linear(z,p+'self_attn.indexer.weights_proj.weight');chosen=[]
            for t in range(len(ids)):
                score=np.maximum(iq[t]@ik[:t+1].T/np.sqrt(hd),0);score=iw[t]@score/np.sqrt(h)
                chosen.append(sorted(range(t+1),key=lambda j:(-score[j],j))[:n['sparseTopK']])
            output=linear(attention(q,key,value,h,chosen),p+'self_attn.o_proj.weight')
        elif (layer+1)%4:
            qkv=linear(z,p+'linear_attn.in_proj_qkv.weight');conv=np.zeros_like(qkv);kernel=weights[p+'linear_attn.conv1d.weight']
            for t in range(len(ids)):
                for j in range(n['conv']):
                    previous=t-n['conv']+1+j
                    if previous>=0: conv[t]+=qkv[previous]*kernel[:,j]
            q,k,value=np.split(silu(conv),3,axis=1);beta=sigmoid(linear(z,p+'linear_attn.in_proj_b.weight'));alpha=linear(z,p+'linear_attn.in_proj_a.weight')+weights[p+'linear_attn.dt_bias']
            decay=np.exp(-np.exp(weights[p+'linear_attn.A_log'])*np.logaddexp(0,alpha))
            y,_=recurrent(q,k,value,decay,beta,h,True);y=y.reshape(len(ids),h,hd);y=norm(y,weights[p+'linear_attn.norm.weight'],1e-6).reshape(len(ids),d)
            output=linear(y*silu(linear(z,p+'linear_attn.in_proj_z.weight')),p+'linear_attn.out_proj.weight')
        else:
            both=linear(z,p+'self_attn.q_proj.weight').reshape(len(ids),h,2*hd);q=both[:,:,:hd];gate=both[:,:,hd:].reshape(len(ids),d)
            q=norm(q,weights[p+'self_attn.q_norm.weight'],1e-6,True).reshape(len(ids),d)
            k=norm(linear(z,p+'self_attn.k_proj.weight').reshape(len(ids),h,hd),weights[p+'self_attn.k_norm.weight'],1e-6,True).reshape(len(ids),d)
            q=rotary(q,h,pos,theta,r,axes=axes);k=rotary(k,h,pos,theta,r,axes=axes)
            output=linear(attention(q,k,linear(z,p+'self_attn.v_proj.weight'),h)*sigmoid(gate),p+'self_attn.o_proj.weight')
        x+=output;z=norm(x,weights[p+'post_attention_layernorm.weight'],a['rms_norm_eps'],n['family']=='qwen3_5')
        if n['family']=='glm_moe_dsa' and layer>=n['firstDense']:
            scores=sigmoid(linear(z,p+'mlp.gate.weight'));bias=weights[p+'mlp.gate.e_score_correction_bias'];out=np.zeros_like(x)
            for t in range(len(ids)):
                experts=sorted(range(n['experts']),key=lambda e:(-scores[t,e]-bias[e],e))[:n['topK']];prob=scores[t,experts]/scores[t,experts].sum()*n['routedScaling']
                for e,w in zip(experts,prob): out[t]+=w*ffn(z[t:t+1],p+f'mlp.experts.{e}.')[0]
            x+=out+ffn(z,p+'mlp.shared_experts.')
        else: x+=ffn(z,p+'mlp.')
    logits=linear(norm(x,weights['model.norm.weight'],a['rms_norm_eps'],n['family']=='qwen3_5'),'lm_head.weight')
    np.testing.assert_allclose(logits.ravel(),fixture['logits'],rtol=2e-8,atol=2e-8)
    print('PASS: independent NumPy '+fixture['family']+' full forward, max error '+str(np.max(np.abs(logits.ravel()-fixture['logits']))))
print('PASS: independent recurrent delta state and outputs.')
