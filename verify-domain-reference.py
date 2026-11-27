"""Independent NumPy reference for the bounded domain architectures; no JS imports."""
import json,math
from pathlib import Path
import numpy as np

def ln(x):
    z=x-x.mean(-1,keepdims=True)
    return z/np.sqrt((z*z).mean(-1,keepdims=True)+1e-6)
def attention(q,k,v):
    q=q.reshape(len(q),2,8);k=k.reshape(len(k),2,8);v=v.reshape(len(v),2,8)
    score=np.einsum('thd,shd->hts',q,k)/np.sqrt(8)
    p=np.exp(score-score.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    return np.einsum('hts,shd->thd',p,v).reshape(len(q),16)
def patches(frame):
    im=np.array(frame['pixels']).reshape(frame['height'],frame['width'],3)
    resized=im[(np.arange(8)+.5).astype(int)[:,None]*frame['height']//8,(np.arange(8)+.5).astype(int)[None,:]*frame['width']//8]
    return resized.reshape(4,2,4,2,3).transpose(0,2,1,3,4).reshape(16,12)*2-1
def main(f):
    model=f['net'];row=f['row'];P={}
    for p in f['layout']:
        P[p['name']]=np.array(model['parameters'][p['offset']:p['offset']+p['length']]).reshape(p['shape'])
    target={p['name']:np.array(model['targetParameters'][p['offset']:p['offset']+p['length']]).reshape(p['shape']) for p in f['layout']}
    def encode(frames,weights=P,visible=None):
        patch=np.concatenate([patches(im) for im in frames]);indices=np.arange(len(patch)) if visible is None else np.array(visible)
        x=patch[indices]@weights['encoder.patch.weight']+weights['encoder.position.weight'][indices];levels=[]
        for layer in range(2):
            p='encoder.'+str(layer)+'.';n=ln(x)
            x=x+attention(n@weights[p+'q'],n@weights[p+'k'],n@weights[p+'v'])@weights[p+'out']
            x=x+np.tanh(ln(x)@weights[p+'up'])@weights[p+'down'];levels.append(ln(x))
        return x,levels
    family=model['family'];loss=0;outputs={}
    if family=='jepa':
        visible=list(range(0,16,2));online,_=encode(row['frames'][:2],visible=visible);_,levels=encode(row['frames'][:2],target)
        pred=attention(P['world.query']@P['world.q'],online@P['world.k'],online@P['world.v'])@P['world.out']
        mask=np.ones(32,dtype=bool);mask[visible]=False
        for layer in range(2):
            y=pred@P['world.level'+str(layer)]
            loss+=.5*np.mean((y[mask]-levels[layer][mask])**2)+.125*np.mean((y[~mask]-levels[layer][~mask])**2)
        outputs['latents']=pred.ravel()
    else:
        encoded,_=encode([row.get('image') or row['frames'][0]])
        chars=list(row['instruction'].encode('utf8'))[:48];text=P['text.embed'][chars].mean(0,keepdims=True)
        state=np.array((row['state']+row['goal']+[0]*9)[:9])[None,:]@P['state.proj']
        context=np.concatenate([encoded,text,state]);width=model['config']['actionDimensions'];horizon=model['config']['chunk']
        if family=='video-flow':target_values,_=encode([row['frames'][-1]],target);target_values=target_values.ravel()
        else:target_values=np.tile(np.array(row['actions']),horizon)
        noise=np.array(f['noise']);time=f['time'];noisy=(1-time)*noise+time*target_values;velocity=target_values-noise
        x=noisy.reshape(horizon,width)@P['flow.input']+P['flow.position']
        modulation=np.array([[math.sin(math.pi*time),math.cos(math.pi*time)]])@P['flow.time']
        modulated=ln(x)*(1+modulation[:,:16])+modulation[:,16:]
        x=x+attention(modulated@P['flow.sq'],modulated@P['flow.sk'],modulated@P['flow.sv'])@P['flow.so']
        x=x+attention(ln(x)@P['flow.cq'],context@P['flow.ck'],context@P['flow.cv'])@P['flow.co']
        x=x+np.tanh(ln(x)@P['flow.up'])@P['flow.down'];y=ln(x)@P['flow.output']
        loss=np.mean((y.ravel()-velocity)**2);outputs['velocity']=y.ravel()
        if family=='joint-action-world':
            memory=np.concatenate([context,x]);pred=attention(P['world.query'][:16]@P['world.q'],memory@P['world.k'],memory@P['world.v'])@P['world.out']@P['world.level1']
            _,levels=encode([row['frames'][-1]],target);loss+=.1*np.mean((pred-levels[1])**2)
        if family=='video-flow':
            z,_=encode([row['frames'][-1]],target);pixels=z@P['video.decoder'];loss+=.1*np.mean((pixels-patches(row['frames'][-1]))**2)
    assert abs(loss-f['loss'])<1e-10,(family,loss,f['loss'])
    for name,values in outputs.items():
        error=float(np.max(np.abs(values-np.array(f[name]))));assert error<1e-10,(family,name,error)
    print('PASS: independent NumPy',family,'patch attention / target / flow, loss error',abs(loss-f['loss']))
for fixture in json.loads(Path('.test-artifacts/domain-reference.json').read_text('utf8')):main(fixture)
