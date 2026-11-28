from pathlib import Path
import json,urllib.request,hashlib,shutil
root=Path('.test-artifacts/v16-evidence');inventory=json.loads((root/'tokenizer-inventory.json').read_text('utf8'));g=next(x for x in inventory if x['id']=='groot-n1.7');public=next(x for x in inventory if x['id']=='vla-jepa')
private_tree=json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{g['repository']}/tree/{g['revision']}",timeout=30))
for entry in g['files']:
    filename=entry['path'];source=next(x for x in public['files'] if x['path']==filename);data=(root/'vla-jepa'/filename).read_bytes();oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();expected=next(x['oid'] for x in private_tree if x['path']==filename)
    assert oid==expected,(filename,oid,expected)
    shutil.copy2(root/'vla-jepa'/filename,root/'groot-n1.7'/filename)
    entry.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),gitBlob=oid,equivalentPublicSource=source['source'],resolution='public asset is byte-identical to the official dependency git blob');entry.pop('error',None)
(root/'tokenizer-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),'utf8');print('GR00T: all 3 processor files match the official Cosmos-Reason2-2B Git blobs')
