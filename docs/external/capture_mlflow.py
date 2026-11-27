"""Run the actual pinned MLflow server against private local fixtures and capture responses."""
import datetime,hashlib,json,os,pathlib,platform,socket,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
OUT=ROOT/('capture-'+STAMP);OUT.mkdir()
SERVER_DIR=OUT/'server';SERVER_DIR.mkdir()
ARTIFACTS=SERVER_DIR/'artifacts';ARTIFACTS.mkdir()
PYTHON=ROOT/'.venv/Scripts/python.exe'
ENV={k:v for k,v in os.environ.items() if not k.startswith(('MLFLOW_','DATABRICKS_'))}
ENV.update({'PYTHONIOENCODING':'utf-8','MLFLOW_ENABLE_ASYNC_LOGGING':'false','MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING':'false','MLFLOW_DISABLE_TELEMETRY':'true','MLFLOW_HTTP_REQUEST_MAX_RETRIES':'0','MLFLOW_HTTP_REQUEST_TIMEOUT':'2','MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR':'0'})
RESULT={'schema':'external-software-reference-v1','software':'MLflow','requestedVersion':'3.0.1','actualRun':True,'scope':'Tracking HTTP API, artifacts, CLI and SQLite restart; no model training or UI parity claimed','createdAt':STAMP,'python':sys.version,'platform':platform.platform(),'cases':[],'serverLifecycle':[],'cleanup':[]}
PROCESS=None;LOG_HANDLE=None

def save():
    text=json.dumps(RESULT,ensure_ascii=False,indent=2)
    (OUT/'reference.json').write_text(text,encoding='utf8')
    (ROOT/'mlflow-reference.json').write_text(text,encoding='utf8')

def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.4):return True
    except PermissionError as error:raise RuntimeError('Local connection probe is blocked by execution permissions') from error
    except OSError:return False

def http_case(name,method,path,payload=None,expected=None):
    import requests
    s=requests.Session();s.trust_env=False
    started=time.perf_counter()
    response=s.request(method,BASE+path,json=payload,timeout=10)
    try:body=response.json()
    except ValueError:body=response.text
    case={'name':name,'method':method,'path':path,'request':payload,'status':response.status_code,'body':body,'contentType':response.headers.get('Content-Type'),'elapsedMs':round((time.perf_counter()-started)*1000,3)}
    if expected is not None:
        expected=[expected] if isinstance(expected,int) else list(expected)
        case['expectedStatus']=expected;case['passed']=response.status_code in expected
    RESULT['cases'].append(case);save()
    if case.get('passed') is False:raise AssertionError(name+': unexpected HTTP '+str(response.status_code))
    return body

def cli_case(name,args,expect):
    env=dict(ENV);env['MLFLOW_TRACKING_URI']=BASE
    run=subprocess.run([str(PYTHON),'-m','mlflow',*args],cwd=SERVER_DIR,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=35,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (OUT/(name+'.stdout.txt')).write_text(run.stdout,encoding='utf8')
    (OUT/(name+'.stderr.txt')).write_text(run.stderr,encoding='utf8')
    RESULT['cases'].append({'name':name,'kind':'official-cli','argv':['python','-m','mlflow',*args],'exitCode':run.returncode,'expectedExitCode':expect,'passed':run.returncode==expect,'stdout':run.stdout[-5000:],'stderr':run.stderr[-5000:]});save()
    if run.returncode!=expect:raise AssertionError(name+': unexpected exit '+str(run.returncode))

def start_server(label):
    global PROCESS,LOG_HANDLE
    if port_open(PORT):raise RuntimeError('Selected port already in use; refusing to touch existing service')
    log=OUT/(label+'.server.log');LOG_HANDLE=log.open('ab')
    argv=[str(PYTHON),'-m','mlflow','server','--host','127.0.0.1','--port',str(PORT),'--backend-store-uri','sqlite:///'+str(SERVER_DIR/'tracking.db').replace('\\','/'),'--serve-artifacts','--artifacts-destination',ARTIFACTS.as_uri()]
    PROCESS=subprocess.Popen(argv,cwd=SERVER_DIR,env=ENV,stdout=LOG_HANDLE,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    RESULT['serverLifecycle'].append({'label':label,'pid':PROCESS.pid,'host':'127.0.0.1','port':PORT,'argv':argv,'log':str(log),'cleanupCommand':['taskkill','/PID',str(PROCESS.pid),'/T','/F']});save()
    for _ in range(120):
        if PROCESS.poll() is not None:raise RuntimeError('Server exited '+str(PROCESS.returncode)+'; see '+str(log))
        try:
            import requests
            s=requests.Session();s.trust_env=False
            response=s.get(BASE+'/health',timeout=.5)
            if response.status_code==200:return
        except Exception:pass
        time.sleep(.25)
    raise RuntimeError('Server did not become ready')

def stop_server(label):
    global PROCESS,LOG_HANDLE
    if PROCESS is None:return
    record={'label':label,'pid':PROCESS.pid,'wasRunning':PROCESS.poll() is None}
    if PROCESS.poll() is None:
        if os.name=='nt':
            killed=subprocess.run(['taskkill','/PID',str(PROCESS.pid),'/T','/F'],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            record.update({'method':'owned PID tree only','taskkillExit':killed.returncode,'stdout':killed.stdout,'stderr':killed.stderr})
        else:PROCESS.terminate()
        try:PROCESS.wait(timeout=15)
        except subprocess.TimeoutExpired:PROCESS.kill();PROCESS.wait(timeout=5)
    record['serverProcessExitCode']=PROCESS.returncode
    if LOG_HANDLE:LOG_HANDLE.close();LOG_HANDLE=None
    for _ in range(20):
        if not port_open(PORT):break
        time.sleep(.1)
    record['portClosed']=not port_open(PORT)
    RESULT['cleanup'].append(record);PROCESS=None;save()
    if not record['portClosed']:raise RuntimeError('Owned server port is still open after cleanup')

try:
    import mlflow
    from mlflow import MlflowClient
    RESULT['actualVersion']=mlflow.__version__
    if mlflow.__version__!='3.0.1':raise RuntimeError('Wrong MLflow version')
    freeze=subprocess.run([str(PYTHON),'-m','pip','freeze'],capture_output=True,text=True,encoding='utf8',check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (OUT/'installed-requirements.txt').write_text(freeze.stdout,encoding='utf8')
    RESULT['installedRequirementsSha256']=hashlib.sha256(freeze.stdout.encode()).hexdigest()
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));PORT=probe.getsockname()[1]
    BASE='http://127.0.0.1:'+str(PORT);RESULT['baseUrl']=BASE
    start_server('initial')
    http_case('health','GET','/health',expected=200)
    original='foundry-reference-original'
    exp=http_case('experiment-create','POST','/api/2.0/mlflow/experiments/create',{'name':original},200)['experiment_id']
    http_case('experiment-duplicate-error','POST','/api/2.0/mlflow/experiments/create',{'name':original},400)
    http_case('experiment-rename-repair','POST','/api/2.0/mlflow/experiments/update',{'experiment_id':exp,'new_name':'foundry-reference-renamed'},200)
    http_case('experiment-create-after-repair','POST','/api/2.0/mlflow/experiments/create',{'name':original},200)
    run=http_case('run-create','POST','/api/2.0/mlflow/runs/create',{'experiment_id':exp,'start_time':int(time.time()*1000),'tags':[{'key':'mlflow.runName','value':'source-contract-fixture'},{'key':'foundry.fixture','value':'manually-logged-reference'}]},200)['run']['info']['run_id']
    RESULT.update({'experimentId':exp,'runId':run})
    http_case('parameter-first-value','POST','/api/2.0/mlflow/runs/log-parameter',{'run_id':run,'key':'learning_rate','value':'0.003'},200)
    http_case('parameter-same-value-idempotent','POST','/api/2.0/mlflow/runs/log-parameter',{'run_id':run,'key':'learning_rate','value':'0.003'},200)
    http_case('parameter-overwrite-error','POST','/api/2.0/mlflow/runs/log-parameter',{'run_id':run,'key':'learning_rate','value':'0.009'},400)
    for step,value in enumerate([1.25,.75]):http_case('metric-step-'+str(step),'POST','/api/2.0/mlflow/runs/log-metric',{'run_id':run,'key':'loss','value':value,'timestamp':int(time.time()*1000),'step':step},200)
    history=http_case('metric-history','GET','/api/2.0/mlflow/metrics/get-history?run_id='+run+'&metric_key=loss',expected=200)
    assert [x['value'] for x in history['metrics']]==[1.25,.75]
    searched=http_case('search-real-filter','POST','/api/2.0/mlflow/runs/search',{'experiment_ids':[exp],'filter':"metrics.loss < 1 AND params.learning_rate = '0.003'",'order_by':['metrics.loss ASC'],'max_results':10},200)
    assert searched['runs'][0]['info']['run_id']==run
    http_case('search-invalid-filter-error','POST','/api/2.0/mlflow/runs/search',{'experiment_ids':[exp],'filter':'metrics.loss ~~ 1'},400)
    http_case('search-recovered-filter','POST','/api/2.0/mlflow/runs/search',{'experiment_ids':[exp],'filter':'metrics.loss < 1'},200)
    http_case('missing-run-error','GET','/api/2.0/mlflow/runs/get?run_id='+'0'*32,expected=404)
    fixture=OUT/'recipe.json';fixture.write_text(json.dumps({'fixture':True,'trainingPerformed':False,'learning_rate':.003,'dataset':'three-local-records'},sort_keys=True),encoding='utf8')
    mlflow.set_tracking_uri(BASE);client=MlflowClient(tracking_uri=BASE)
    client.log_artifact(run,str(fixture),artifact_path='configs')
    listing=http_case('artifact-list','GET','/api/2.0/mlflow/artifacts/list?run_id='+run+'&path=configs',expected=200)
    assert listing['files'][0]['path']=='configs/recipe.json'
    downloads=OUT/'downloads';downloads.mkdir()
    downloaded=pathlib.Path(client.download_artifacts(run,'configs/recipe.json',str(downloads)))
    digest=hashlib.sha256(fixture.read_bytes()).hexdigest();assert downloaded.read_bytes()==fixture.read_bytes()
    RESULT['cases'].append({'name':'artifact-download-exact-bytes','kind':'official-sdk','path':'configs/recipe.json','sha256':digest,'downloadSha256':hashlib.sha256(downloaded.read_bytes()).hexdigest(),'passed':True});save()
    cli_case('cli-version',['--version'],0)
    cli_case('cli-describe-existing',['runs','describe','--run-id',run],0)
    cli_case('cli-describe-missing',['runs','describe','--run-id','0'*32],1)
    http_case('run-finish','POST','/api/2.0/mlflow/runs/update',{'run_id':run,'status':'FINISHED','end_time':int(time.time()*1000)},200)
    stop_server('controlled-stop')
    cli_case('cli-server-unavailable',['runs','describe','--run-id',run],1)
    start_server('restart-same-sqlite')
    recovered=http_case('run-get-after-restart','GET','/api/2.0/mlflow/runs/get?run_id='+run,expected=200)
    assert recovered['run']['info']['status']=='FINISHED'
    assert recovered['run']['data']['params'][0]['value']=='0.003'
    assert recovered['run']['data']['metrics'][0]['value']==.75
    cli_case('cli-describe-after-restart',['runs','describe','--run-id',run],0)
    RESULT['passed']=True
except Exception as error:
    RESULT.update({'passed':False,'fatalError':str(error),'traceback':traceback.format_exc()})
finally:
    try:stop_server('final-cleanup')
    except Exception as error:RESULT['cleanupError']=str(error);RESULT['passed']=False
    save()
    print(json.dumps({'output':str(OUT),'version':RESULT.get('actualVersion'),'cases':len(RESULT['cases']),'passed':RESULT.get('passed'),'fatalError':RESULT.get('fatalError'),'cleanupError':RESULT.get('cleanupError')},ensure_ascii=True))
sys.exit(0 if RESULT.get('passed') else 1)
