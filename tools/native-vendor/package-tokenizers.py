from pathlib import Path
import json,gzip,base64,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path('.test-artifacts/v16-evidence');inventory=json.loads((root/'tokenizer-inventory.json').read_text('utf8'));blobs={};models={}
for m in inventory:
 if not m['files']:continue
 for prefix,branch in [('', 'text'),('text_tokenizer/','generator')]:
  local=root/m['id']/prefix
  if not (local/'tokenizer.json').exists():continue
  tok=json.loads((local/'tokenizer.json').read_text('utf8'));cfg=json.loads((local/'tokenizer_config.json').read_text('utf8'))
  template=(local/'chat_template.jinja').read_text('utf8') if (local/'chat_template.jinja').exists() else json.loads((local/'chat_template.json').read_text('utf8')).get('chat_template',cfg.get('chat_template','')) if (local/'chat_template.json').exists() else cfg.get('chat_template','')
  payload=json.dumps({'tokenizer':tok,'config':cfg},ensure_ascii=False,separators=(',',':')).encode();sha=hashlib.sha256(payload).hexdigest()
  if sha not in blobs:blobs[sha]=base64.b64encode(gzip.compress(payload,mtime=0)).decode()
  models[m['id']+(':'+branch if branch!='text' else '')]={'id':m['id'],'branch':branch,'repository':m['repository'],'revision':m['revision'],'blob':sha,'template':template,'vocabSize':max(list(tok['model']['vocab'].values())+[x['id'] for x in tok['added_tokens']])+1,'files':[f for f in m['files'] if f['path'].startswith(prefix)],'source':m['files'][0]['source']}
out=Path('src/native-tokenizer-assets.js');out.write_text('/* Pinned public tokenizer assets; source digests and equivalent-blob evidence are retained. */\n(function(root){const C=root.FoundryCore,models='+json.dumps(models,ensure_ascii=False,separators=(',',':'))+',blobs='+json.dumps(blobs,separators=(',',':'))+';C.NativeTokenizerAssets={models,blobs};})(globalThis);\n','utf8');Path('docs/原生Tokenizer来源.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),'utf8');print(json.dumps({'tokenizerAssets':len(models),'uniqueBlobs':len(blobs),'inlineBytes':out.stat().st_size}))
