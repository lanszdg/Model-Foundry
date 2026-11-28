"""Download only the committed, pinned source inventory; reject changed bytes."""
from pathlib import Path
import concurrent.futures, hashlib, json, sys, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
root=Path('.test-artifacts/v16-evidence');root.mkdir(parents=True,exist_ok=True)
inventory=json.loads(Path('docs/原生Tokenizer来源.json').read_text('utf8'))
def fetch(model):
    for entry in model['files']:
        target=root/model['id']/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
        source=entry.get('equivalentPublicSource') or entry['source']
        data=target.read_bytes() if target.exists() else urllib.request.urlopen(source,timeout=60).read()
        assert hashlib.sha256(data).hexdigest()==entry['sha256'],target
        if entry.get('gitBlob'):
            actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert actual==entry['gitBlob'],(target,'Git blob changed')
        target.write_bytes(data)
    print(model['id'],len(model['files']),'verified files',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(fetch,inventory))
(root/'tokenizer-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),'utf8')
