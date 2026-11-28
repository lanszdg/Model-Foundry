from pathlib import Path
import json,gzip,base64,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path('.test-artifacts/v12-external/.venv/Lib/site-packages/mlflow/server/js/build');files={};total=0
for p in root.rglob('*'):
 if not p.is_file() or p.suffix=='.map':continue
 data=p.read_bytes();rel=p.relative_to(root).as_posix();files[rel]={'data':base64.b64encode(gzip.compress(data,mtime=0)).decode(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};total+=len(data)
payload={'version':'3.0.1','source':'https://github.com/mlflow/mlflow/tree/v3.0.1/mlflow/server/js','files':files}
target=Path('src/vendor/mlflow-ui-assets.js');target.write_text('/* MLflow 3.0.1 unmodified distribution assets. Apache-2.0; bundled license files retained. */\nglobalThis.FoundryMLflowAssets='+json.dumps(payload,separators=(',',':'))+';\n','utf8');print(json.dumps({'files':len(files),'originalBytes':total,'inlineBytes':target.stat().st_size}))
main=(root/'static/js/main.c25f73e6.js').read_text('utf8');parts=[]
for marker in ['.p=','ajax-api','/experiments/search','graphql']:
 i=main.find(marker);parts.append({'marker':marker,'context':main[max(0,i-60):i+150]})
Path('.test-artifacts/mlflow-bootstrap-v16.json').write_text(json.dumps(parts),'utf8')
