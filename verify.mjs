import {spawnSync} from 'node:child_process';
for(const args of [['verify-v3.cjs','--fixture'],['verify-v4.cjs','--fixture'],['verify-v4-boundaries.cjs'],['verify-v5.cjs'],['verify-v6.cjs','--fixture'],['verify-v7.cjs','--fixture'],['verify-v8.cjs','--fixture'],['verify-v9.cjs','--fixture'],['verify-v10.cjs','--fixture'],['verify-v11.cjs','--fixture'],['verify-v12.cjs'],['verify-v12-contracts.cjs'],['verify-v13.cjs'],['verify-v14.cjs'],['verify-v15.cjs'],['verify-v16.cjs'],['verify-v17.cjs'],['verify-v17-depth.cjs'],['verify-v17-processors.cjs'],['verify-v17-adaptation.cjs'],['verify-v17-async.cjs'],['verify-archive-v17.cjs'],['verify-v2.cjs']]){
 const result=spawnSync(process.execPath,args,{stdio:'inherit',cwd:new URL('.',import.meta.url),windowsHide:true});
 if(result.status!==0)process.exit(result.status||1);
}
