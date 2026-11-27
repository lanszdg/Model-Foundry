"""Actual Grafana 12 + Prometheus 3.3; controlled exporter fixture, no user services."""
import copy,datetime,hashlib,http.server,json,os,pathlib,secrets,socket,subprocess,sys,threading,time,traceback,urllib.parse
import requests
ROOT=pathlib.Path(__file__).resolve().parent
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
OUT=ROOT/('monitoring-'+STAMP);OUT.mkdir()
RESULT={'schema':'external-monitoring-reference-v1','actualRun':True,'createdAt':STAMP,'scope':'Actual Grafana/Prometheus HTTP APIs and evaluation using a controlled exporter fixture; not real GPU performance or UI parity','cases':[],'lifecycle':[],'cleanup':[],'stateSamples':[]}
PROCS={};LOGS={};EXPORTER=None;FIXTURE={'value':10.0,'empty':False};AUTH=('foundry-reference',secrets.token_hex(16));HTTP=requests.Session();HTTP.trust_env=False
ENV={k:v for k,v in os.environ.items() if not k.startswith('GF_')}
ENV.update({'GOMEMLIMIT':'512MiB','GOMAXPROCS':'2'})
RESULT['isolatedProcessLimits']={'GOMEMLIMIT':'512MiB','GOMAXPROCS':'2','scope':'Grafana and Prometheus child environments only; not system settings','memoryLimitKind':'Go runtime soft memory limit, not a hard RSS cap'}

def save():
    text=json.dumps(RESULT,ensure_ascii=False,indent=2)
    (OUT/'reference.json').write_text(text,encoding='utf8');(ROOT/'monitoring-reference.json').write_text(text,encoding='utf8')

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def open_port(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.4):return True
    except PermissionError as error:raise RuntimeError('Local connection probe is blocked by execution permissions') from error
    except OSError:return False

def call(name,method,base,path,payload=None,expected=None,params=None,auth=False,record=True):
    started=time.perf_counter();r=HTTP.request(method,base+path,json=payload,params=params,auth=AUTH if auth else None,timeout=12)
    try:body=r.json()
    except ValueError:body=r.text
    row={'name':name,'method':method,'path':path,'request':payload,'params':params,'status':r.status_code,'body':body,'elapsedMs':round((time.perf_counter()-started)*1000,3)}
    if expected is not None:
        expected=[expected] if isinstance(expected,int) else list(expected);row.update({'expectedStatus':expected,'passed':r.status_code in expected})
    if record:RESULT['cases'].append(row);save()
    if row.get('passed') is False:raise AssertionError(name+': unexpected HTTP '+str(r.status_code))
    return body

def start(name,argv,port,health,env=None):
    if open_port(port):raise RuntimeError('Port is occupied; refusing to touch existing service')
    log=(OUT/(name+'-'+str(len(RESULT['lifecycle']))+'.log')).open('ab');LOGS[name]=log
    p=subprocess.Popen(argv,cwd=OUT,env=env or ENV,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));PROCS[name]=(p,port)
    RESULT['lifecycle'].append({'name':name,'pid':p.pid,'argv':argv,'host':'127.0.0.1','port':port,'cleanupCommand':['taskkill','/PID',str(p.pid),'/T','/F']});save()
    for _ in range(480):
        if p.poll() is not None:raise RuntimeError(name+' exited '+str(p.returncode))
        try:
            if HTTP.get('http://127.0.0.1:'+str(port)+health,timeout=.5).status_code==200:return
        except Exception:pass
        time.sleep(.25)
    raise RuntimeError(name+' readiness timeout')

def stop(name):
    entry=PROCS.pop(name,None)
    if not entry:return
    p,port=entry;row={'name':name,'pid':p.pid,'wasRunning':p.poll() is None}
    if p.poll() is None:
        if os.name=='nt':
            killed=subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));row.update({'taskkillExit':killed.returncode,'stdout':killed.stdout,'stderr':killed.stderr})
        else:p.terminate()
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
    if LOGS.get(name):LOGS.pop(name).close()
    for _ in range(20):
        if not open_port(port):break
        time.sleep(.1)
    row.update({'exitCode':p.returncode,'portClosed':not open_port(port)});RESULT['cleanup'].append(row);save()
    if not row['portClosed']:raise RuntimeError(name+' port remains open')

class Exporter(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!='/metrics':self.send_error(404);return
        body='' if FIXTURE['empty'] else '# HELP foundry_gpu_utilization_percent Controlled reference fixture, not GPU telemetry.\n# TYPE foundry_gpu_utilization_percent gauge\nfoundry_gpu_utilization_percent{node="gpu-01",gpu="0"} '+str(FIXTURE['value'])+'\n'
        data=body.encode();self.send_response(200);self.send_header('Content-Type','text/plain; version=0.0.4');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*a):pass

def query(expr,name,expected=200):return call(name,'GET',PROM,'/api/v1/query',params={'query':expr},expected=expected)
def grafana_query(expr,name,expected=(200,400,500,502)):
    now=int(time.time()*1000)
    return call(name,'POST',GRAF,'/api/ds/query',{'from':str(now-60000),'to':str(now),'queries':[{'refId':'A','datasource':{'type':'prometheus','uid':DS},'expr':expr,'instant':True,'range':False,'intervalMs':1000,'maxDataPoints':1000}]},expected,auth=True)

def wait_metric(value,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        b=call('metric-poll','GET',PROM,'/api/v1/query',params={'query':'foundry_gpu_utilization_percent'},expected=200,record=False)
        v=b.get('data',{}).get('result',[])
        if v and float(v[0]['value'][1])==value:return
        time.sleep(.4)
    raise RuntimeError('Prometheus did not scrape fixture value '+str(value))

def sample_states():
    b=call('alert-state-poll','GET',GRAF,'/api/prometheus/grafana/api/v1/rules',expected=200,auth=True,record=False)
    states=[]
    for group in b.get('data',{}).get('groups',[]):
        for rule in group.get('rules',[]):
            if rule.get('name')=='Controlled fixture utilization':
                states.append({'state':rule.get('state'),'health':rule.get('health'),'lastError':rule.get('lastError'),'alerts':rule.get('alerts',[]),'lastEvaluation':rule.get('lastEvaluation')})
    RESULT['stateSamples'].append({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixture':dict(FIXTURE),'states':states,'rawResponse':b});save();return states

def wait_state(wanted,timeout=45):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        states=sample_states()
        normal=lambda value:str(value).lower().split(' ',1)[0].split('(',1)[0]
        observed={normal(s.get('state','')) for s in states}
        observed|={normal(a.get('state','')) for s in states for a in s.get('alerts',[])}
        if observed.intersection(wanted):return observed
        time.sleep(1)
    raise RuntimeError('Did not observe requested state '+str(wanted))

def update_rule(rule,name):return call(name,'PUT',GRAF,'/api/v1/provisioning/alert-rules/'+RULE,rule,200,auth=True)

try:
    g=json.loads((ROOT/'grafana-installation.json').read_text('utf8'));p=json.loads((ROOT/'prometheus-installation.json').read_text('utf8'))
    RESULT['installations']={'grafana':g,'prometheus':p}
    grafana_binary=str(pathlib.Path(g['serverExecutable']).with_name('grafana.exe'))
    for name,binary,args in [('grafana',grafana_binary,['--version']),('prometheus',p['serverExecutable'],['--version'])]:
        r=subprocess.run([binary,*args],env=ENV,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));RESULT['cases'].append({'name':name+'-version','kind':'official-cli','exitCode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'passed':r.returncode==0})
    EXPORTER=http.server.ThreadingHTTPServer(('127.0.0.1',0),Exporter);exporter_port=EXPORTER.server_port
    threading.Thread(target=EXPORTER.serve_forever,daemon=True).start();RESULT['exporter']={'host':'127.0.0.1','port':exporter_port,'kind':'controlled gauge fixture','actualGPU':False}
    PROM_PORT=free_port();GRAF_PORT=free_port();PROM='http://127.0.0.1:'+str(PROM_PORT);GRAF='http://127.0.0.1:'+str(GRAF_PORT)
    RESULT.update({'prometheusUrl':PROM,'grafanaUrl':GRAF})
    conf=OUT/'prometheus.yml';conf.write_text('global:\n  scrape_interval: 1s\n  evaluation_interval: 1s\nscrape_configs:\n  - job_name: foundry_reference\n    static_configs:\n      - targets: ["127.0.0.1:'+str(exporter_port)+'"]\n',encoding='utf8')
    prom_argv=[p['serverExecutable'],'--config.file='+str(conf),'--storage.tsdb.path='+str(OUT/'prometheus-data'),'--web.listen-address=127.0.0.1:'+str(PROM_PORT),'--storage.tsdb.retention.time=1h']
    start('prometheus',prom_argv,PROM_PORT,'/-/ready')
    for directory in ['grafana-data','grafana-logs','grafana-plugins']:(OUT/directory).mkdir()
    genv=dict(ENV);genv.update({'GF_SERVER_HTTP_ADDR':'127.0.0.1','GF_SERVER_HTTP_PORT':str(GRAF_PORT),'GF_PATHS_DATA':str(OUT/'grafana-data'),'GF_PATHS_LOGS':str(OUT/'grafana-logs'),'GF_PATHS_PLUGINS':str(OUT/'grafana-plugins'),'GF_SECURITY_ADMIN_USER':AUTH[0],'GF_SECURITY_ADMIN_PASSWORD':AUTH[1],'GF_ANALYTICS_REPORTING_ENABLED':'false','GF_ANALYTICS_CHECK_FOR_UPDATES':'false','GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES':'false','GF_PLUGINS_PREINSTALL_DISABLED':'true','GF_NEWS_NEWS_FEED_ENABLED':'false','GF_USERS_ALLOW_SIGN_UP':'false','GF_UNIFIED_ALERTING_MIN_INTERVAL':'10s'})
    grafana_argv=[grafana_binary,'server','--homepath',g['homepath']]
    start('grafana',grafana_argv,GRAF_PORT,'/api/health',genv)
    health=call('grafana-health','GET',GRAF,'/api/health',expected=200);assert health['version']=='12.0.0'
    build=call('prometheus-buildinfo','GET',PROM,'/api/v1/status/buildinfo',expected=200);assert build['data']['version']=='3.3.0'
    wait_metric(10.0)
    query('foundry_gpu_utilization_percent{node="gpu-01",gpu="0"}','promql-exact-label-query')
    query('sum(foundry_gpu_utilization_percent)','promql-aggregate-query')
    query('foundry_gpu_utilization_percent{','promql-parse-error',400)
    query('foundry_gpu_utilization_percent','promql-query-recovered')
    DS='foundry-reference-prom';ds=call('datasource-create','POST',GRAF,'/api/datasources',{'name':'Foundry Reference Prometheus','uid':DS,'type':'prometheus','access':'proxy','url':PROM,'jsonData':{'httpMethod':'POST','prometheusType':'Prometheus','prometheusVersion':'3.3.0','timeInterval':'1s'}},200,auth=True)
    call('datasource-health','GET',GRAF,'/api/datasources/uid/'+DS+'/health',expected=200,auth=True)
    good=grafana_query('foundry_gpu_utilization_percent','grafana-real-promql-query',200)
    assert good.get('results',{}).get('A',{}).get('frames')
    bad=grafana_query('foundry_gpu_utilization_percent{','grafana-promql-parse-error')
    assert 'error' in json.dumps(bad).lower()
    grafana_query('foundry_gpu_utilization_percent','grafana-query-recovered',200)
    dashboard={'id':None,'uid':'foundry-reference-dashboard','title':'Foundry Reference Dashboard','schemaVersion':41,'version':0,'panels':[{'id':1,'type':'timeseries','title':'Controlled fixture utilization','gridPos':{'x':0,'y':0,'w':24,'h':10},'datasource':{'type':'prometheus','uid':DS},'targets':[{'refId':'A','expr':'foundry_gpu_utilization_percent','datasource':{'type':'prometheus','uid':DS}}]}]}
    call('dashboard-create','POST',GRAF,'/api/dashboards/db',{'dashboard':dashboard,'overwrite':False},200,auth=True)
    current=call('dashboard-get','GET',GRAF,'/api/dashboards/uid/'+dashboard['uid'],expected=200,auth=True)
    call('dashboard-version-conflict','POST',GRAF,'/api/dashboards/db',{'dashboard':dashboard,'overwrite':False},412,auth=True)
    current['dashboard']['title']='Foundry Reference Dashboard repaired'
    call('dashboard-save-after-repair','POST',GRAF,'/api/dashboards/db',{'dashboard':current['dashboard'],'overwrite':True},200,auth=True)
    folder='foundry-reference-folder';RULE='foundry-reference-alert'
    call('alert-folder-create','POST',GRAF,'/api/folders',{'uid':folder,'title':'Foundry Reference'},200,auth=True)
    condition={'refId':'B','queryType':'','relativeTimeRange':{'from':0,'to':0},'datasourceUid':'-100','model':{'conditions':[{'evaluator':{'params':[50],'type':'gt'},'operator':{'type':'and'},'query':{'params':['A']},'reducer':{'params':[],'type':'last'},'type':'query'}],'datasource':{'type':'__expr__','uid':'-100'},'intervalMs':1000,'maxDataPoints':43200,'refId':'B','type':'classic_conditions'}}
    rule={'uid':RULE,'title':'Controlled fixture utilization','orgID':1,'folderUID':folder,'ruleGroup':'reference-10s','condition':'B','for':'10s','keep_firing_for':'10s','noDataState':'NoData','execErrState':'Error','isPaused':False,'labels':{'scope':'controlled-local-fixture'},'annotations':{'summary':'Reference API acceptance; no real hardware'},'data':[{'refId':'A','queryType':'','relativeTimeRange':{'from':30,'to':0},'datasourceUid':DS,'model':{'expr':'foundry_gpu_utilization_percent','instant':True,'range':False,'intervalMs':1000,'maxDataPoints':43200,'refId':'A'}},condition]}
    invalid=copy.deepcopy(rule);invalid['condition']='MISSING';invalid['uid']='foundry-reference-bad-condition'
    call('alert-rule-unresolved-condition-accepted','POST',GRAF,'/api/v1/provisioning/alert-rules',invalid,201,auth=True)
    call('alert-rule-remove-bad-condition','DELETE',GRAF,'/api/v1/provisioning/alert-rules/'+invalid['uid'],expected=204,auth=True)
    invalid_duration=copy.deepcopy(rule);invalid_duration['for']='not-a-duration';invalid_duration['uid']='foundry-reference-bad-duration'
    call('alert-rule-invalid-duration','POST',GRAF,'/api/v1/provisioning/alert-rules',invalid_duration,400,auth=True)
    created=call('alert-rule-create','POST',GRAF,'/api/v1/provisioning/alert-rules',rule,201,auth=True)
    assert created.get('keep_firing_for')=='10s'
    group=call('alert-group-get','GET',GRAF,'/api/v1/provisioning/folder/'+folder+'/rule-groups/reference-10s',expected=200,auth=True)
    group['interval']=10;call('alert-group-interval','PUT',GRAF,'/api/v1/provisioning/folder/'+folder+'/rule-groups/reference-10s',group,200,auth=True)
    wait_state({'inactive','normal'})
    FIXTURE['value']=80.0;wait_metric(80.0);wait_state({'pending'});wait_state({'firing','alerting'})
    FIXTURE['value']=10.0;wait_metric(10.0);wait_state({'recovering'});wait_state({'inactive','normal'})
    FIXTURE['empty']=True
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        b=call('empty-poll','GET',PROM,'/api/v1/query',params={'query':'foundry_gpu_utilization_percent'},record=False)
        if not b.get('data',{}).get('result'):break
        time.sleep(.5)
    query('foundry_gpu_utilization_percent','promql-empty-data')
    wait_state({'nodata','no_data'})
    FIXTURE['empty']=False;wait_metric(10.0);wait_state({'inactive','normal'})
    invalid_query=copy.deepcopy(rule);invalid_query['data'][0]['model']['expr']='foundry_gpu_utilization_percent{'
    update_rule(invalid_query,'alert-query-change-to-error');wait_state({'error'})
    update_rule(rule,'alert-query-repaired');wait_state({'inactive','normal'})
    stop('prometheus')
    failed=grafana_query('foundry_gpu_utilization_percent','datasource-unavailable-error')
    assert 'error' in json.dumps(failed).lower()
    start('prometheus',prom_argv,PROM_PORT,'/-/ready');wait_metric(10.0)
    grafana_query('foundry_gpu_utilization_percent','datasource-query-after-restart',200)
    RESULT['passed']=True
except Exception as error:RESULT.update({'passed':False,'fatalError':str(error),'traceback':traceback.format_exc()})
finally:
    for name in ['grafana','prometheus']:
        try:stop(name)
        except Exception as error:RESULT.setdefault('cleanupErrors',[]).append(str(error));RESULT['passed']=False
    if EXPORTER:
        exporter_port=EXPORTER.server_port;EXPORTER.shutdown();EXPORTER.server_close();RESULT['cleanup'].append({'name':'controlled-exporter','port':exporter_port,'portClosed':not open_port(exporter_port)})
    save();print(json.dumps({'output':str(OUT),'cases':len(RESULT['cases']),'stateSamples':len(RESULT['stateSamples']),'passed':RESULT.get('passed'),'fatalError':RESULT.get('fatalError'),'cleanupErrors':RESULT.get('cleanupErrors')},ensure_ascii=True))
sys.exit(0 if RESULT.get('passed') else 1)
