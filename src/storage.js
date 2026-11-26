/* Atomic browser project persistence with legacy save migration. */
(function(root){'use strict';
let db=null,pending=null,pump=null,epoch=0;
const api={initial:null,mode:'memory'};
function transaction(value,remove=false){return new Promise((resolve,reject)=>{const tx=db.transaction('projects','readwrite'),store=tx.objectStore('projects');if(remove)store.delete('active');else store.put(value,'active');tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error||Error('Project transaction failed'));tx.onabort=()=>reject(tx.error||Error('Project transaction aborted'));});}
api.open=async()=>{let legacy=null;try{legacy=JSON.parse(localStorage.getItem('model-foundry-v1')||'null');}catch{}
 try{db=await new Promise((resolve,reject)=>{const req=indexedDB.open('model-foundry-projects',1);req.onupgradeneeded=()=>req.result.createObjectStore('projects');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);req.onblocked=()=>reject(Error('Database upgrade blocked'));});const saved=await new Promise((resolve,reject)=>{const req=db.transaction('projects','readonly').objectStore('projects').get('active');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});api.initial=saved?JSON.parse(saved):legacy;api.mode='IndexedDB';if(!saved&&legacy){await transaction(JSON.stringify(legacy));try{localStorage.removeItem('model-foundry-v1');}catch{}}}
 catch{api.initial=legacy;api.mode='localStorage';}
 return api.initial;
};
api.write=state=>{const serialized=JSON.stringify(state),mine=epoch;pending={serialized,mine};if(!pump)pump=(async()=>{while(pending){const item=pending;pending=null;if(item.mine!==epoch)continue;if(db)await transaction(item.serialized);else localStorage.setItem('model-foundry-v1',item.serialized);}})().finally(()=>{pump=null;});return pump;};
api.replace=async state=>{epoch++;pending=null;if(pump)await pump.catch(()=>{});if(db){if(state===null)await transaction(null,true);else await transaction(JSON.stringify(state));try{localStorage.removeItem('model-foundry-v1');}catch{}}else if(state===null)localStorage.removeItem('model-foundry-v1');else localStorage.setItem('model-foundry-v1',JSON.stringify(state));};
api.flush=()=>pump||Promise.resolve();
root.FoundryStorage=api;
})(globalThis);
