"""Independent NumPy forward reference. Run node verify-v7.cjs first."""
import json
from pathlib import Path
import numpy as np

r = json.loads(Path('.test-artifacts/transformer-reference.json').read_text(encoding='utf-8'))
m, ids, labels = r['model'], r['tokens'], r['labels']
a = m['architecture']
weights = np.asarray(m['parameters'], dtype=np.float64)
p = {s['name']: weights[s['offset']:s['offset']+s['length']].reshape(s['shape']) for s in r['layout']}
d, h, kh = a['hidden_size'], a['num_attention_heads'], a['num_key_value_heads']
hd, count = d // h, len(ids)

def norm(x, name):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+a['rms_norm_eps']) * p[name]

def rope(x, heads):
    z = x.reshape(count, heads, hd//2, 2)
    z = z[..., 0] + 1j*z[..., 1]
    angles = np.arange(count)[:, None, None] * a['rope_theta'] ** (-np.arange(0, hd, 2)/hd)[None, None, :]
    z = z * np.exp(1j*angles)
    return np.stack([z.real, z.imag], axis=-1).reshape(count, heads, hd)

x = p['model.embed_tokens.weight'][ids]
for layer in range(a['num_hidden_layers']):
    key = f'model.layers.{layer}.'
    z = norm(x, key+'input_layernorm.weight')
    q = rope(z @ p[key+'self_attn.q_proj.weight'], h)
    k = np.repeat(rope(z @ p[key+'self_attn.k_proj.weight'], kh), h//kh, axis=1)
    v = np.repeat((z @ p[key+'self_attn.v_proj.weight']).reshape(count, kh, hd), h//kh, axis=1)
    scores = np.einsum('thd,shd->hts', q, k)/np.sqrt(hd)
    scores[:, np.triu_indices(count, 1)[0], np.triu_indices(count, 1)[1]] = -np.inf
    probabilities = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    x = x + np.einsum('hts,shd->thd', probabilities, v).reshape(count, d) @ p[key+'self_attn.o_proj.weight']
    z = norm(x, key+'post_attention_layernorm.weight')
    gate, up = z @ p[key+'mlp.gate_proj.weight'], z @ p[key+'mlp.up_proj.weight']
    x = x + (gate/(1+np.exp(-gate))*up) @ p[key+'mlp.down_proj.weight']

logits = norm(x, 'model.norm.weight') @ p['lm_head.weight']
maximum = logits.max(axis=-1)
losses = maximum + np.log(np.exp(logits-maximum[:, None]).sum(axis=-1)) - logits[np.arange(count), labels]
np.testing.assert_allclose(logits.reshape(-1), r['logits'], rtol=1e-11, atol=1e-11)
np.testing.assert_allclose(losses, r['losses'], rtol=1e-11, atol=1e-11)
print('PASS: independent NumPy RMSNorm / complex RoPE / causal GQA / SwiGLU / logits / NLL reference.')
