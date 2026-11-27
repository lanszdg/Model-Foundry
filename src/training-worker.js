/* Loaded only inside the inline training Worker. */
const sessions=new Map();
self.onmessage=event=>{const m=event.data;try{
 if(m.type==='begin'){const world=new FoundryCore.World();sessions.set(m.key,{world,acc:FoundryCore.Accumulation.create(world,m.job,m.step)});}
 else if(m.type==='micro')sessions.get(m.key).acc.run(m.micro);
 else if(m.type==='finish'){const result=sessions.get(m.key).acc.finish();sessions.delete(m.key);self.postMessage({key:m.key,result});}
 else if(m.type==='drop')sessions.delete(m.key);
}catch(e){sessions.delete(m.key);self.postMessage({key:m.key,micro:m.micro,error:e.trace||e.message});}};
