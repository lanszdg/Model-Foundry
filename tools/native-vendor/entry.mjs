import {Tokenizer} from '@huggingface/tokenizers';
import {Template,Interpreter} from '@huggingface/jinja';
const applyFilter=Interpreter.prototype.applyFilter;
Interpreter.prototype.applyFilter=function(operand,filter,environment){
 if(operand.type==='ArrayValue'&&filter.type==='CallExpression'&&filter.callee?.value==='unique'){
  const [args,kwargs]=this.evaluateArguments(filter.args,environment),sensitive=(args[0]??kwargs.get('case_sensitive'))?.value??false,attribute=(args[1]??kwargs.get('attribute'))?.value,seen=new Set(),out=[];
  for(const item of operand.value){let selected=item;if(attribute!==undefined)for(const part of String(attribute).split('.'))selected=selected?.value instanceof Map?selected.value.get(part):selected?.value?.[Number(part)];let key=selected?.value;if(typeof key==='string'&&!sensitive)key=key.toLowerCase();if(!seen.has(key)){seen.add(key);out.push(item);}}return new operand.constructor(out);
 }
 return applyFilter.call(this,operand,filter,environment);
};
globalThis.FoundryNativeHF={Tokenizer,Template};
