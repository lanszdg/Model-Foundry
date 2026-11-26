(function(){
'use strict';
const canvas=document.getElementById('world'),gl=canvas.getContext('webgl',{antialias:true,alpha:false});
const state={x:0,z:9,yaw:-.23,pitch:-.055,active:false,drag:false,keys:{},last:0,seated:false,bob:0,waypoint:null};
const targets=[{id:'station',p:[-4.2,1.9,6.8],type:'desk',name:'工程工位',sub:'坐下并使用电脑',stand:[-4.2,8.9]},...['gpu-01','gpu-02','gpu-03','gpu-04','control-01','storage-01'].map((id,i)=>({id,p:[[2.7,4.7,6.7,8.7,-7,-4.8][i],1.8,-3.45],type:'rack',name:id,sub:i<4?'H100 计算节点':'管理 / 存储设备'})),{id:'network',p:[-9.5,1.8,-.8],type:'network',name:'交换与配电面板',sub:'链路 / 端口 / 现场拓扑'},{id:'observer',p:[4.2,1.9,6.8],type:'monitor',name:'观测工位',sub:'GPU / 作业 / 请求监控'}];
window.Scene={state,targets,setActive:v=>state.active=v,canUseDesk(){return !gl||state.seated||Math.hypot(state.x+4.2,state.z-8.9)<3.8||Math.hypot(state.x-4.2,state.z-8.9)<3.8;},goDesk(){if(!this.canUseDesk()){this.guide('station');return false;}this.sit(true);return true;},sit(v,desk='station'){if(v&&!state.seated){state.beforeSeat={x:state.x,z:state.z,yaw:state.yaw,pitch:state.pitch};state.deskX=desk==='observer'?4.2:-4.2;state.transition={x:state.deskX,z:8.65,yaw:0,pitch:-.015};}else if(!v&&state.seated){state.transition=state.beforeSeat||{x:-4.2,z:9,yaw:0,pitch:0};}state.seated=v;state.keys={};},guide(id){state.waypoint=targets.find(t=>t.id===id)||targets[0];window.showToast?.('已标记 '+state.waypoint.name+'，请移动到设备前操作');},interact(t){if(!t)return;if(Math.hypot(t.p[0]-state.x,t.p[2]-state.z)>3.8){this.guide(t.id);return;}if(t.type==='desk'||t.type==='monitor'){this.sit(true,t.id);window.openDesktop?.(t.type==='monitor'?'monitor':'workspace');}else window.openDevice?.(t);}};
if(!gl){document.getElementById('worldFallback').hidden=false;return;}
const vs=`attribute vec3 p;attribute vec3 n;attribute vec3 c;uniform mat4 vp;varying vec3 col;varying float depth;void main(){vec4 pos=vp*vec4(p,1.);gl_Position=pos;float l=.46+.40*max(dot(n,normalize(vec3(-.35,1.,.5))),0.);col=c*l;depth=pos.w;}`;
const fs=`precision mediump float;varying vec3 col;varying float depth;void main(){float fog=smoothstep(12.,43.,depth);gl_FragColor=vec4(mix(col,vec3(.025,.055,.066),fog*.75),1.);}`;
function shader(t,s){const v=gl.createShader(t);gl.shaderSource(v,s);gl.compileShader(v);if(!gl.getShaderParameter(v,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(v));return v;}
const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,vs));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);gl.useProgram(prog);
const attrs=['p','n','c'].map(k=>gl.getAttribLocation(prog,k)),vpLoc=gl.getUniformLocation(prog,'vp');
let verts=[];function color(hex){return[parseInt(hex.slice(1,3),16)/255,parseInt(hex.slice(3,5),16)/255,parseInt(hex.slice(5,7),16)/255];}
function box(x,y,z,w,h,d,col){const C=Array.isArray(col)?col:color(col),a=w/2,b=h/2,c=d/2;
const faces=[[[1,0,0],[[a,-b,-c],[a,b,-c],[a,b,c],[a,-b,c]]],[[-1,0,0],[[-a,-b,c],[-a,b,c],[-a,b,-c],[-a,-b,-c]]],[[0,1,0],[[-a,b,-c],[-a,b,c],[a,b,c],[a,b,-c]]],[[0,-1,0],[[-a,-b,c],[-a,-b,-c],[a,-b,-c],[a,-b,c]]],[[0,0,1],[[a,-b,c],[a,b,c],[-a,b,c],[-a,-b,c]]],[[0,0,-1],[[-a,-b,-c],[-a,b,-c],[a,b,-c],[a,-b,-c]]]];
for(const[n,points]of faces)for(const i of[0,1,2,0,2,3])verts.push(points[i][0]+x,points[i][1]+y,points[i][2]+z,...n,...C);}
// Architecture, raised floor and cable trays.
box(0,-.15,0,24,.3,29,'#55676d');box(0,5.5,0,24,.2,29,'#35474e');
box(-12,2.7,0,.3,5.5,29,'#71878d');box(12,2.7,0,.3,5.5,29,'#71878d');box(0,2.7,-14,24,5.5,.3,'#71878d');box(0,2.7,14,24,5.5,.3,'#35474e');
for(let x=-12;x<=12;x+=1.2)box(x,.007,0,.017,.012,28,'#354950');for(let z=-14;z<=14;z+=1.2)box(0,.009,z,24,.012,.017,'#354950');
box(0,.019,-3,1.6,.014,23,'#344b51');for(let z=-13;z<11;z+=1.5)box(0,.03,z,.055,.012,.45,'#a6bd76');
for(const x of[-9,-3,3,9]){box(x,5.2,0,.1,.3,27,'#1b3038');for(const z of[-10,-3,4,11])box(x,5.04,z,2.1,.045,.36,'#d8eee3');}
for(const x of[-10,10])box(x,4.7,-1,.22,.22,25,'#253d46');
// Structural columns and entrance.
for(const x of[-11.4,11.4])for(const z of[-10,-2,6,13]){box(x,2.6,z,.3,5.2,.4,'#384f58');box(x,1.1,z+.22,.34,.12,.025,'#d0b968');}
box(-8,2.5,-13.7,4.8,3.3,.15,'#193941');box(-8,2.45,-13.59,4.3,2.8,.04,'#15323a');
for(let x=-9.8;x<-6;x+=.35)box(x,2.45,-13.54,.028,2.6,.02,'#254952');
// Glass separation: thin mullions with open view to the compute bay.
for(const x of[-10.5,-7,-3.5,0,3.5,7,10.5]){box(x,2.35,1.5,.045,4.6,.055,'#779499');box(x,4.55,1.5,3.5,.035,.055,'#779499');}
box(0,4.6,1.5,24,.3,.25,'#273f48');box(0,4.43,1.65,8,.04,.04,'#b4d781');
const racks=[{x:2.7,z:-4.5},{x:4.7,z:-4.5},{x:6.7,z:-4.5},{x:8.7,z:-4.5},{x:-7,z:-4.5},{x:-4.8,z:-4.5}];
racks.forEach((r,k)=>{box(r.x,1.9,r.z,1.48,3.8,1.8,'#101e25');box(r.x,3.85,r.z,1.55,.12,1.9,'#283d46');box(r.x,.1,r.z,1.6,.2,2,'#192f37');
box(r.x-.69,1.9,r.z+1,.08,3.7,.1,'#3e5660');box(r.x+.69,1.9,r.z+1,.08,3.7,.1,'#3e5660');
for(let i=0;i<12;i++){let y=.4+i*.27;box(r.x,y,r.z+.94,1.28,.215,.13,i%3===0?'#324b55':'#253b44');box(r.x-.34,y,r.z+1.018,.55,.08,.02,'#14272f');for(let q=0;q<6;q++)box(r.x-.53+q*.18,y-.056,r.z+1.024,.08,.015,.016,'#52707a');}
box(r.x+.57,1.9,r.z+1.065,.035,2.6,.025,'#758f98');box(r.x,4.1,r.z,.2,.4,.2,'#304b55');});
// Rear power and cooling infrastructure.
for(let x=-1;x<10;x+=2.8){box(x,2,-12.7,2.3,4,.8,'#4d666f');for(let i=0;i<14;i++)box(x,1+i*.19,-12.25,1.9,.06,.05,'#1c3842');}
for(let z=-11;z<-3;z+=.35){box(11.4,1.8,z,.28,.045,.18,'#344e58');}
// Two engineering desks, screens and keyboards.
for(const x of[-4.2,4.2]){box(x,.88,7.2,3.4,.14,1.6,'#50646b');for(const dx of[-1.4,1.4])box(x+dx,.42,7.2,.10,.82,1.3,'#1d343e');box(x,1.07,6.85,.3,.33,.25,'#253c45');box(x,1.72,6.8,2.5,1.35,.13,'#101f27');box(x,1.73,6.88,2.34,1.17,.02,'#17383c');box(x,.99,7.48,1.4,.045,.45,'#213640');for(let k=0;k<5;k++)box(x,.999,7.32+k*.07,1.25,.01,.018,'#657d84');box(x+1.05,.99,7.5,.2,.07,.28,'#20343c');
box(x,.48,8.55,.8,.1,.8,'#283c45');box(x,.83,8.93,.8,.85,.15,'#2f4750');box(x,.24,8.55,.1,.4,.1,'#637a82');box(x,.06,8.55,.9,.05,.85,'#253e47');}
// Screen content and office details.
for(let i=0;i<8;i++){box(-4.68,2.12-i*.105,6.905,.95-(i%3)*.13,.018,.01,i===0?'#b9e993':i%3?'#72b1b4':'#416e79');box(-3.48,1.86-i*.075,6.905,.48,.016,.01,'#3f747b');}
// A monitor stays blank until its observability service is deployed.
box(-9.5,1.65,-.9,1.8,2.6,.7,'#304a51');for(let i=0;i<12;i++){box(-9.5,.8+i*.15,-.51,1.6,.10,.04,'#132a32');for(let j=0;j<8;j++)box(-10.15+j*.18,.8+i*.15,-.475,.055,.04,.02,'#67877f');}
box(-9.5,3,-.9,2,.15,.85,'#58736f');
box(-9.5,.7,8.4,1.5,1.4,.55,'#394f57');box(-9.5,1.6,8.4,1.9,.6,.65,'#658082');box(-9.5,1.9,8.4,1.7,.04,.6,'#829c8c');
// Perspective geometry upload.
const staticBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,staticBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);const count=verts.length/9;
const dynamicBuffer=gl.createBuffer();
function multiply(a,b){const r=new Float32Array(16);for(let c=0;c<4;c++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)r[c*4+row]+=a[k*4+row]*b[c*4+k];return r;}
function cameraMatrix(){const aspect=canvas.width/canvas.height,f=1/Math.tan(Math.PI/5.3),near=.08,far=80,projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);const right=[cy,0,sy],up=[-sy*sp,cp,cy*sp],back=[-sy*cp,-sp,cy*cp],pos=[state.x,(state.seated?1.72:1.7)+state.bob,state.z];
const view=new Float32Array([right[0],up[0],back[0],0,right[1],up[1],back[1],0,right[2],up[2],back[2],0,-right.reduce((v,x,i)=>v+x*pos[i],0),-up.reduce((v,x,i)=>v+x*pos[i],0),-back.reduce((v,x,i)=>v+x*pos[i],0),1]);return multiply(projection,view);}
function bind(buffer){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);attrs.forEach((a,i)=>{gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,36,i*12);});}
const markers=targets;
const labelHost=document.getElementById('worldLabels');markers.forEach((m,i)=>{const el=document.createElement('button');el.className='world-label';el.innerHTML='<span class="marker-num">'+(i===0?'⌘':String(i).padStart(2,'0'))+'</span><span><strong>'+m.name+'</strong><small>'+m.sub+'</small></span><kbd>E</kbd>';el.onclick=()=>Scene.interact(m);labelHost.append(el);m.el=el;});
function updateMarkers(vp){let focused=null,best=100;for(const m of markers){const distance=Math.hypot(m.p[0]-state.x,m.p[2]-state.z),p=[m.p[0],m.type==='rack'?(distance<3.8?2.35:4.1):m.p[1]+.8,m.p[2],1],clip=[0,0,0,0];for(let i=0;i<4;i++)for(let k=0;k<4;k++)clip[i]+=vp[k*4+i]*p[k];const x=clip[0]/clip[3],y=clip[1]/clip[3];const visible=clip[3]>.2&&Math.abs(x)<1.1&&Math.abs(y)<1.1&&!state.seated;m.el.style.display=visible&&(distance<5.5||m===state.waypoint||m.id==='station')?'flex':'none';m.el.classList.toggle('near',distance<3.8);m.el.style.left=((x+1)/2*innerWidth)+'px';m.el.style.top=((1-y)/2*innerHeight)+'px';if(!state.seated&&distance<3.8){const dx=m.p[0]-state.x,dz=m.p[2]-state.z,dot=(dx*Math.sin(state.yaw)-dz*Math.cos(state.yaw))/Math.max(.1,distance);if(dot>.5&&distance-dot<best){focused=m;best=distance-dot;}}}state.focused=focused;const prompt=document.getElementById('interactPrompt');if(prompt){prompt.hidden=!focused||state.seated||!!document.getElementById('devicePanel')?.classList.contains('open');if(focused)prompt.innerHTML='<kbd>E</kbd><span>'+focused.name+'<small>'+focused.sub+'</small></span><b>交互</b>';}}
let last=0,frames=0;
function frame(t){const dt=Math.min(.05,(t-last)/1000||0);last=t;const ratio=Math.min(devicePixelRatio||1,1.5);if(canvas.width!==Math.floor(innerWidth*ratio)||canvas.height!==Math.floor(innerHeight*ratio)){canvas.width=Math.floor(innerWidth*ratio);canvas.height=Math.floor(innerHeight*ratio);gl.viewport(0,0,canvas.width,canvas.height);}
if(state.transition){const tr=state.transition,k=Math.min(1,dt*7);for(const key of ['x','z','yaw','pitch'])state[key]+=(tr[key]-state[key])*k;if(Math.abs(state.x-tr.x)+Math.abs(state.z-tr.z)<.01)state.transition=null;}
if(state.active&&!state.seated&&!document.getElementById('desktop').classList.contains('open')&&!document.getElementById('manual').classList.contains('open')&&!document.getElementById('devicePanel')?.classList.contains('open')){
let forward=(state.keys.w||state.keys.ArrowUp?1:0)-(state.keys.s||state.keys.ArrowDown?1:0),side=(state.keys.d?1:0)-(state.keys.a?1:0);if(state.keys.ArrowLeft)state.yaw-=dt;if(state.keys.ArrowRight)state.yaw+=dt;let velocity=dt*(state.keys.Shift?5.3:3);
let nx=state.x+(Math.sin(state.yaw)*forward+Math.cos(state.yaw)*side)*velocity,nz=state.z+(-Math.cos(state.yaw)*forward+Math.sin(state.yaw)*side)*velocity;
const blocked=(x,z)=>racks.some(r=>Math.abs(x-r.x)<1.05&&Math.abs(z-r.z)<1.3)||[-4.2,4.2].some(v=>Math.abs(x-v)<1.95&&Math.abs(z-7.2)<1.1);
if(!blocked(nx,state.z))state.x=Math.max(-10.7,Math.min(10.7,nx));if(!blocked(state.x,nz))state.z=Math.max(-11,Math.min(12,nz));state.bob=(forward||side)?Math.sin(t*.009)*.025:state.bob*.85;}
gl.enable(gl.DEPTH_TEST);gl.clearColor(.025,.055,.066,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const vp=cameraMatrix();gl.uniformMatrix4fv(vpLoc,false,vp);bind(staticBuffer);gl.drawArrays(gl.TRIANGLES,0,count);
verts=[];const s=window.engine?.state;racks.forEach((r,k)=>{const ni=k<4?k+2:k===4?0:1,n=s?.nodes[ni],on=n?.power,healthy=n?.health==='OK';const led=!on?'#2b3d43':!healthy?'#ff624f':n?.runtime?'#b6f579':'#f6c767';for(let i=0;i<12;i++)box(r.x+.46,.4+i*.27,r.z+1.025,.045,.045,.025,led);box(r.x,3.63,r.z+1.028,.55,.03,.02,on?'#9ac9b3':'#31474f');
 const start=verts.length;box(r.x,1.95,r.z+1.07,1.34,3.45,.055,'#1b3038');for(let l=0;l<15;l++)box(r.x,1+l*.13,r.z+1.105,1.2,.022,.015,'#526f70');const angle=n?.door?-1.65:0,hinge=r.x-.67;for(let i=start;i<verts.length;i+=9){const x=verts[i]-hinge,z=verts[i+2]-(r.z+1.07),nx=verts[i+3],nz=verts[i+5];verts[i]=hinge+x*Math.cos(angle)+z*Math.sin(angle);verts[i+2]=r.z+1.07-x*Math.sin(angle)+z*Math.cos(angle);verts[i+3]=nx*Math.cos(angle)+nz*Math.sin(angle);verts[i+5]=-nx*Math.sin(angle)+nz*Math.cos(angle);}
 if(n?.cables)for(const [kind,offset,col]of [['power',-.25,'#d4af65'],['mgmt',0,'#72c3d0'],['ib',.25,'#b3d87b']]){if(!n.cables[kind])continue;box(r.x+offset,4.33,r.z,.035,.8,.035,col);box(r.x+offset,4.73,(r.z+1.5)/2,.035,.035,Math.abs(r.z-1.5),col);}
 if(n?.installer)box(r.x,3.18,r.z+1.05,n.installer.progress*.009,.04,.02,'#c2ee8c');});
if(s?.services.monitor){const telemetry=s.telemetry||[];for(let i=0;i<20;i++){const v=telemetry.at(-20+i)?.gpu||0;box(3.18+i*.105,1.35+v*.004,6.92,.045,.04+v*.008,.015,'#99c88b');}}
if(state.waypoint&&!state.seated){const wp=state.waypoint.p;box(wp[0],.035,wp[2]+1.5,.7,.02,.07,'#badf84');box(wp[0],.035,wp[2]+1.5,.07,.02,.7,'#badf84');}
if(state.seated){const dx=(state.deskX||-4.2)+4.2;box(-4.8+dx,1.02,8.12,.18,.16,.55,'#66776e');box(-3.63+dx,1.02,8.12,.18,.16,.55,'#66776e');box(-4.8+dx,1.08,7.88,.19,.09,.24,'#ae9c86');box(-3.63+dx,1.08,7.88,.19,.09,.24,'#ae9c86');}
const run=!!s?.jobs.some(j=>j.status==='RUNNING');for(let i=0;i<12;i++)box(-3.5,1.3+i*.067,6.92,.05+(run?(.12+Math.sin(t/500+i)*.1):.04),.025,.012,run?'#b6f579':'#426e77');
gl.bindBuffer(gl.ARRAY_BUFFER,dynamicBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.DYNAMIC_DRAW);bind(dynamicBuffer);gl.drawArrays(gl.TRIANGLES,0,verts.length/9);
if(frames++%3===0)updateMarkers(vp);requestAnimationFrame(frame);}
canvas.addEventListener('pointerdown',e=>{if(!state.active||state.seated)return;state.drag=true;state.dragDistance=0;state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture?.(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!state.drag)return;state.dragDistance+=Math.abs(e.clientX-state.lastX)+Math.abs(e.clientY-state.lastY);state.yaw+=(e.clientX-state.lastX)*.003;state.pitch=Math.max(-.8,Math.min(.8,state.pitch-(e.clientY-state.lastY)*.003));state.lastX=e.clientX;state.lastY=e.clientY;});
canvas.addEventListener('pointerup',()=>{state.drag=false;if(state.dragDistance<5&&state.focused)Scene.interact(state.focused);});canvas.addEventListener('pointercancel',()=>state.drag=false);
addEventListener('keydown',e=>{if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;state.keys[e.key]=true;if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Tab',' '].includes(e.key))e.preventDefault();if((e.key==='e'||e.key==='E')&&state.active&&!state.seated&&!document.getElementById('devicePanel')?.classList.contains('open')){if(state.focused)Scene.interact(state.focused);else window.showToast?.('靠近并面向设备后按 E 交互');}});
addEventListener('keyup',e=>state.keys[e.key]=false);addEventListener('blur',()=>{state.keys={};state.drag=false;});requestAnimationFrame(frame);
})();
