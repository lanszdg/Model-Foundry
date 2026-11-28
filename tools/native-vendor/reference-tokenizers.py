from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('.test-artifacts/v16-python').resolve()));sys.stdout.reconfigure(encoding='utf-8')
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template
root=Path('.test-artifacts/v16-evidence');sources=json.loads((root/'tokenizer-inventory.json').read_text('utf8'))
strings=['你好，Model Foundry! 123\n','Hello world','\r\n\n\t a  b\u00a0c\u2028d','é e\u0301 हिन्दी العربية 日本語 한글','👨‍👩‍👧‍👦🧑🏽‍💻🇨🇳','１2٣4 1234567890','<|im_start|>assistant\n<｜Assistant｜>[gMASK]<sop>','\\n\\t\\\\\"\'()[]{} <>&','\x00\x01\x02\x7f\x80\xff','ß SS K K Å Å İ i Σ σ ς']
strings += [f'Line {i}: 中文{i*137} abc_{i} += {i/7:.8f}\n' for i in range(25)]
chats=[{'messages':[{'role':'user','content':'你好 Hello?'}],'options':{'add_generation_prompt':True,'enable_thinking':False,'thinking':False}}, {'messages':[{'role':'user','content':'你好 Hello?'},{'role':'assistant','content':'世界 World!'}],'options':{'enable_thinking':False,'thinking':False}}, {'messages':[{'role':'system','content':'Be concise.'},{'role':'user','content':'A?'},{'role':'assistant','content':'B.'},{'role':'user','content':'C?'}],'options':{'add_generation_prompt':True}}, {'messages':[{'role':'user','content':'Why?'},{'role':'assistant','reasoning_content':'Think carefully.','content':'Because.'}],'options':{'enable_thinking':True,'thinking':True}}, {'messages':[{'role':'user','content':'read weather'},{'role':'assistant','content':'','tool_calls':[{'id':'call_1','type':'function','function':{'name':'weather','arguments':{'city':'北京'}}}]},{'role':'tool','tool_call_id':'call_1','name':'weather','content':'sunny'},{'role':'assistant','content':'Sunny.'}],'options':{'tools':[{'type':'function','function':{'name':'weather','description':'weather','parameters':{'type':'object','properties':{'city':{'type':'string'}}}}}],'enable_thinking':False,'thinking':False}}]
results=[]
for source in sources:
 if not source['files']:continue
 for prefix,branch in [('', 'text'),('text_tokenizer/','generator')]:
  p=root/source['id']/prefix
  if not (p/'tokenizer.json').exists():continue
  ident=source['id']+(':'+branch if branch!='text' else '');t=Tokenizer.from_file(str(p/'tokenizer.json'));cfg=json.loads((p/'tokenizer_config.json').read_text('utf8'));template=(p/'chat_template.jinja').read_text('utf8') if (p/'chat_template.jinja').exists() else json.loads((p/'chat_template.json').read_text('utf8')).get('chat_template',cfg.get('chat_template','')) if (p/'chat_template.json').exists() else cfg.get('chat_template','')
  compiled=_compile_jinja_template(template);renders=[]
  for chat in chats:
   kwargs={**cfg,'add_generation_prompt':False,**chat['options']}
   try:
    rendered=compiled.render(messages=chat['messages'],**kwargs);renders.append({**chat,'text':rendered,'ids':t.encode(rendered,add_special_tokens=False).ids})
   except Exception as e:renders.append({**chat,'error':str(e)})
  results.append({'id':ident,'strings':[{'text':s,'ids':t.encode(s,add_special_tokens=False).ids,'decoded':t.decode(t.encode(s,add_special_tokens=False).ids,skip_special_tokens=False)} for s in strings],'chats':renders})
  print(ident,len(strings),'Rust encodings',len(renders),'Python Jinja cases',flush=True)
Path('.test-artifacts/native-tokenizer-reference-v16.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf8')
