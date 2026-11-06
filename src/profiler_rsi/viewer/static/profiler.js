"use strict";
// profiler_rsi comparison product: library (table) → (compare | profile).
// Hash-routed SPA over the JSON API in view.py; design language = base.css
// (erndata cook-view family). Light default + base.css ◐ dark toggle.

const $ = s => document.querySelector(s);
const esc = s => String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const view = () => $('#view');
let META = {}, SEL = new Set();
let SORT = { key:'order', dir:1 };

async function api(path, opts){
  const r = await fetch(path, opts);
  if(!r.ok){ let e; try{e=(await r.json()).error}catch(_){} throw new Error(e||('HTTP '+r.status)); }
  return r.json();
}
function toast(msg){ const t=$('#toast'); t.textContent=msg; t.style.display='block';
  clearTimeout(toast._t); toast._t=setTimeout(()=>t.style.display='none',2200); }
function go(h){ location.hash=h; }
function num(v,f){ if(v==null||isNaN(v))return'—';
  if(f==='{:.1%}')return (v*100).toFixed(1)+'%';
  if(f==='{:,.0f}')return Math.round(v).toLocaleString();
  if(f==='{:.1f}%')return v.toFixed(1)+'%';
  if(f==='{:.2f}')return v.toFixed(2);
  if(f==='{:.1f}')return v.toFixed(1); return v; }
function mval(s,key){ const a=s.analysis||{}; if(s.perf && key in s.perf)return s.perf[key];
  return ({idle_pct:a.idle_pct,launches:a.launches,gpu_s:a.gpu_s,span_s:a.span_s,
           avg_us:a.avg_us,concurrency:a.concurrency})[key]; }
// dir 0 (descriptive) → show magnitude in neutral grey, no ▲▼ and no good/bad colour.
function delta(cur,base,dir){
  if(cur==null||base==null||isNaN(cur)||isNaN(base)||base===0)return{t:'',c:'zero'};
  const pct=(cur/base-1)*100; if(Math.abs(pct)<0.5)return{t:'±0',c:'zero'};
  if(!dir)return{t:(pct>0?'+':'')+pct.toFixed(0)+'%',c:'zero'};
  const better=(dir>0)===(pct>0);
  return{t:(pct>0?'▲+':'▼')+pct.toFixed(0)+'%',c:better?'up':'down'}; }
// category colour: single source from api_meta palette (compare._CATEGORY_HUE); grey fallback.
function catHue(c){ return (META.palette&&META.palette.hue[c])||'#9c9a94'; }
// tracelight opens a trace via ?path=<path relative to its --root> (same root as ours).
// Everything — embedded iframe and new-tab link alike — goes through our same-origin
// /tl/ proxy (view.py strips tracelight's X-Frame-Options / CSP), so the user only ever
// sees this origin; tracelight's own port stays internal to the proxy.
function tlUrl(trace){ return (trace&&META.tracelight_url)?`/tl/?path=${encodeURIComponent(trace)}`:''; }

// wall-clock span budget: the four slices sum to ~100% of span, so idle is visible
// (the GPU-active category bar normalises idle away). left→right = productive→wasted.
const BUDGET_SEG=[['compute','计算','var(--accent)'],['overlap_comm','通信重叠','var(--s3)'],
  ['exposed_comm','暴露通信','var(--warn)'],['idle','idle 气泡','var(--s8)']];
function budgetSegs(an){ const b=(an&&an.budget)||{};
  return BUDGET_SEG.map(([k,lab,c])=>{const p=b[k]||0;
    return p>0.1?`<div class=seg data-cat="${esc(lab)}" data-pct="${p.toFixed(1)}" data-s="span ${an.span_s||'?'}s" style="width:${p}%;background:${c}"></div>`:'';}).join(''); }
// drop segments that are ~0 for every experiment (e.g. 通信重叠 on single-node); withVal
// appends the single-experiment % (profile only — meaningless across a multi-exp compare).
function budgetLegend(b,withVal){ const segs=b?BUDGET_SEG.filter(([k])=>(b[k]||0)>0.05):BUDGET_SEG;
  return `<div class=legend>`+segs.map(([k,lab,c])=>
  `<span class="item" style="cursor:default"><i class=sw style="background:${c}"></i>${lab}${withVal?` ${(b[k]||0).toFixed(1)}%`:''}</span>`).join('')+`</div>`; }
// generic idle drivers — why idle is high/low: launch count, kernel granularity, stream occupancy.
function idleDrivers(an){ if(!an)return'';
  const d=[`kernel 启动 <b>${(an.launches||0).toLocaleString()}</b>`,
    `avg <b>${Math.round(an.avg_us||0)}µs</b>/kernel`,
    `流占用 <b>${(an.concurrency||0).toFixed(2)}×</b>`,
    `主流占墙钟 <b>${(an.main_stream_pct||0).toFixed(0)}%</b>`,
    `流数 <b>${an.n_streams||0}</b>`];
  return `<div class=telemetry style="border:0;padding:8px 0 2px">${d.map(x=>`<span>${x}</span>`).join('')}</div>`; }

/* ── telemetry strip + header ──────────────────────────────── */
function telemetry(extra){
  const parts=[`<span><b>${META.count}</b> experiments</span>`,
    `<span>baseline <b>${esc(META.baseline||'—')}</b></span>`,
    `<span>root <b>${esc((META.trace_root||'').split('/').slice(-1)[0])}</b></span>`,
    `<span>manifest <b>${esc(META.manifest_name||'')}</b></span>`];
  if(extra)parts.push(extra);
  $('#telemetry').innerHTML=parts.join('');
  $('#rootmeta').textContent=(META.title||'')+' · '+(META.subtitle||'');
}

/* ── router ────────────────────────────────────────────────── */
async function route(){
  META=await api('/api/meta');
  const h=location.hash||'#/';
  try{
    if(h.startsWith('#/compare')) await renderCompare(new URLSearchParams(h.split('?')[1]||''));
    else if(h.startsWith('#/exp/')) await renderProfile(decodeURIComponent(h.slice(6)));
    else await renderLibrary();
  }catch(e){ view().innerHTML=`<div class=card><div class=empty>加载失败: ${esc(e.message)}</div></div>`; }
}

/* ── place: LIBRARY (clean sortable table) ─────────────────── */
const LIB_COLS=[
  {key:'label', lab:'实验', sort:s=>s.label||s.id, cls:''},
  {key:'model', lab:'模型 / 注意力', sort:s=>(s.config||{}).model||'', cls:'txt'},
  {key:'experts', lab:'专家·topk·层', sort:s=>(s.config||{}).experts||0, cls:'num'},
  {key:'tok_s', lab:'tok/s', sort:s=>(s.perf||{}).tok_s||0, cls:'num'},
  {key:'mfu', lab:'MFU', sort:s=>(s.perf||{}).mfu||0, cls:'num'},
  {key:'idle', lab:'GPU idle', sort:s=>(s.analysis||{}).idle_pct||0, cls:'num'},
];
async function renderLibrary(){
  const d=await api('/api/library');
  telemetry(`<span>with trace <b>${d.strategies.filter(s=>s.has_trace).length}/${d.strategies.length}</b></span>`);
  SEL=new Set([...SEL].filter(id=>d.strategies.some(s=>s.id===id)));
  const col=LIB_COLS.find(c=>c.key===SORT.key);
  const sorted=[...d.strategies].sort((a,b)=>{
    if(!col) return (a.order||0)-(b.order||0);
    const x=col.sort(a), y=col.sort(b);
    const c=(typeof x==='string')?x.localeCompare(y):(x-y);
    return c*SORT.dir; });
  const arr=k=>SORT.key===k?`<span class=arr>${SORT.dir>0?'▲':'▼'}</span>`:'';
  const head=`<tr><th class="pickcell nosort"></th>`+
    LIB_COLS.map(c=>`<th class="${c.cls==='num'?'num':''}" data-sort="${c.key}" title="点击按此列排序">${esc(c.lab)}${arr(c.key)}</th>`).join('')+
    `<th class="num nosort">trace</th><th class="nosort"></th></tr>`;
  const rows=sorted.map(s=>{
    const a=s.analysis||{}, cfg=s.config||{}, pf=s.perf||{}, flag=s.flags&&s.flags.length;
    const dotc=flag?'var(--warn)':(s.has_trace?'var(--s3)':'var(--ink-3)');
    const moe=`${cfg.experts??'?'} · top${cfg.topk??'?'} · ${cfg.moe_layers??'?'}`;
    return `<tr data-id="${esc(s.id)}" data-go="#/exp/${encodeURIComponent(s.id)}" title="点击整行 → ${esc(s.label)} 详情页">
      <td class=pickcell><input type=checkbox class=pick data-id="${esc(s.id)}" title="勾选以加入对比" ${SEL.has(s.id)?'checked':''}></td>
      <td class=name><div><span class=nm>${esc(s.label)}</span><span class=goarrow>›</span>${flag?` <span class="pill pill-warn flag">${esc(s.flags.join(' · '))}</span>`:''}</div>
        <div class=sub>${esc(s.sub||'')}</div></td>
      <td class=txt><span class=sdot style="background:${dotc}"></span>${esc(cfg.model||'—')} · ${esc(cfg.attn||'—')}</td>
      <td class=num>${esc(moe)}</td>
      <td class=num>${num(pf.tok_s,'{:,.0f}')}</td>
      <td class=num>${num(pf.mfu,'{:.1%}')}</td>
      <td class=num>${num(a.idle_pct,'{:.1f}%')}</td>
      <td class=num>${s.has_trace?'<span class=sdot style="background:var(--s3)"></span>':'<span class=notrace>无</span>'}</td>
      <td class=rm><span data-rm="${esc(s.id)}" title="移除">✕</span></td></tr>`;}).join('');
  view().innerHTML=`<div class="card lib">
     <div class=head><h2>实验库</h2><span class=grow></span>
       <span class="meta dim" id=selmeta></span>
       <button class=ghost id=addbtn title="添加对比实验">＋ 添加对比实验</button>
       <button class=primary id=cmpbtn disabled title="先勾选 ≥2 个实验，再点此并排对比">对比选中</button></div>
     ${d.strategies.length?
       `<div class=scroll style="max-height:none"><table class=lib><thead>${head}</thead><tbody>${rows}</tbody></table></div>`:
       `<div class=empty>还没有实验，点 ＋ 添加对比实验 录入第一条</div>`}
     </div>${addForm(d.strategies.find(s=>s.id===d.baseline)||d.strategies[0]||null)}`;
  const upd=()=>{ $('#cmpbtn').disabled=SEL.size<2;
    $('#cmpbtn').textContent=SEL.size?`对比选中 (${SEL.size})`:'对比选中';
    $('#selmeta').textContent=SEL.size?`${SEL.size} 选中`:''; };
  upd();
  view().querySelectorAll('.pick').forEach(c=>c.onclick=e=>{ e.stopPropagation();
    c.checked?SEL.add(c.dataset.id):SEL.delete(c.dataset.id); upd(); });
  view().querySelectorAll('th[data-sort]').forEach(th=>th.onclick=()=>{
    const k=th.dataset.sort;
    if(SORT.key===k) SORT.dir*=-1; else { SORT.key=k; SORT.dir=(k==='label'||k==='model')?1:-1; }
    renderLibrary(); });
  view().querySelectorAll('tr[data-go]').forEach(tr=>tr.onclick=e=>{
    if(e.target.closest('.pick')||e.target.closest('[data-rm]'))return; go(tr.dataset.go); });
  view().querySelectorAll('[data-rm]').forEach(n=>n.onclick=async e=>{ e.stopPropagation();
    if(!confirm('移除实验 '+n.dataset.rm+' ?'))return;
    await api('/api/exp/'+encodeURIComponent(n.dataset.rm),{method:'DELETE'}); toast('已移除'); route(); });
  $('#cmpbtn').onclick=()=>{ if(SEL.size>=2){ const ids=[...SEL];
    go(`#/compare?sel=${ids.map(encodeURIComponent).join(',')}&base=${encodeURIComponent(ids[0])}`); } };
  $('#addbtn').onclick=()=>{ const f=$('#addfold'); f.open=!f.open;
    if(f.open)f.scrollIntoView({behavior:'smooth',block:'nearest'}); };
  wireAddForm();
}
function addForm(tmpl){
  // config/perf prefilled from the baseline row — a new experiment is usually a variant,
  // so you tweak the few fields that differ rather than typing all of them. id/label/trace
  // stay blank (unique per run); idle/launches/kernels come from the trace, never typed.
  const cfg=(tmpl&&tmpl.config)||{}, prf=(tmpl&&tmpl.perf)||{};
  const cf=[['model','ernie_dev'],['attn','窗口flash'],['experts','512'],
    ['moe_ffn','192'],['topk','8'],['moe_layers','6/12'],['shared','0'],['mbs','8']];
  const pf=[['tok_s',''],['tflops',''],['mfu','0.19']];
  const val=(o,k)=>o[k]!=null?` value="${esc(o[k])}"`:'';
  const tl=tmpl?`只需填 id + trace，其余已克隆基准「${esc(tmpl.label||tmpl.id)}」`:'录入一条实验（只需 id + trace）';
  return `<details class="card fold" id=addfold><summary class=head><h2>添加对比实验</h2>
     <span class=grow></span><span class="hint" id=f-sum>${tl}</span></summary>
   <div class=form style="margin-top:12px">
     <label>id *</label><input type=text id=a-id placeholder="唯一标识 e.g. myrun">
     <label>trace</label><input type=text id=a-trace placeholder="相对 trace root, e.g. myrun/rank0_x.nsys-rep（可留空=只有 config/perf）">
     <label></label><div class=btns><button class=primary id=a-save>添加</button>
       <span class=hint>只需 id + trace；其余已克隆基准，需微调再点下方展开</span><span class=hint id=a-msg></span></div>
     <details class=subfold style="margin-top:4px"><summary class=hint style="cursor:pointer">▸ 可选 · label / sub / config / perf（已按基准预填）</summary>
       <div style="margin-top:10px">
         <label>label</label><input type=text id=a-label placeholder="显示名（留空用 id）">
         <label>sub</label><input type=text id=a-sub placeholder="副标题（可留空）">
         <label>config</label><div class=cfgrid>${cf.map(([k,ph])=>`<div class=cfgrp><span class=kv1>${k}</span><input type=text id="a-c-${k}" placeholder="${ph}"${val(cfg,k)}></div>`).join('')}</div>
         <label>perf</label><div class=cfgrid>${pf.map(([k,ph])=>`<div class=cfgrp><span class=kv1>${k}</span><input type=text id="a-p-${k}" placeholder="${ph}"${val(prf,k)}></div>`).join('')}</div>
       </div></details>
   </div></details>`;
}
function wireAddForm(){ const sv=$('#a-save'); if(!sv)return;
  sv.onclick=async()=>{ const g=id=>$(id).value.trim();
    const intf=v=>v===''?undefined:parseInt(v,10), fl=v=>v===''?undefined:parseFloat(v);
    const entry={ id:g('#a-id'), label:g('#a-label')||g('#a-id'), sub:g('#a-sub'),
      trace:g('#a-trace')||undefined,
      config:{model:g('#a-c-model'),attn:g('#a-c-attn'),experts:intf(g('#a-c-experts')),
        moe_ffn:intf(g('#a-c-moe_ffn')),topk:intf(g('#a-c-topk')),moe_layers:g('#a-c-moe_layers'),
        shared:intf(g('#a-c-shared')),mbs:intf(g('#a-c-mbs'))},
      perf:{tok_s:fl(g('#a-p-tok_s')),tflops:fl(g('#a-p-tflops')),mfu:fl(g('#a-p-mfu'))} };
    if(!entry.id){ $('#a-msg').textContent='id 必填'; return; }
    try{ await api('/api/exp',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(entry)});
      toast('已添加 '+entry.label); route(); }
    catch(e){ $('#a-msg').textContent=e.message; } };
}

/* ── place: COMPARE ────────────────────────────────────────── */
async function renderCompare(q){
  const sel=(q.get('sel')||'').split(',').filter(Boolean);
  let base=q.get('base')||sel[0];
  const d=await api('/api/compare?ids='+sel.map(encodeURIComponent).join(','));
  const S=d.strategies, byId=Object.fromEntries(S.map(s=>[s.id,s]));
  if(!byId[base])base=S[0]&&S[0].id;
  telemetry(`<span>compare <b>${S.length}</b></span><span>cats <b>${d.categories.length}</b></span>`);
  const b=byId[base];
  // metric matrix
  let mx=`<table class=matrix><thead><tr><th class=row>指标</th>`+S.map(s=>
    `<th class="${s.id===base?'base':''}" data-base="${esc(s.id)}" title="${s.id===base?'当前基准':'点击把这列设为基准'}"><span class=lab>${esc(s.label)}</span><span class=sb>${esc(s.sub||'')}</span>${s.id===base?'<span class=basetag>◀ 基准</span>':'<span class=setbase>设为基准</span>'}</th>`).join('')+`</tr></thead><tbody>`;
  for(const m of d.metrics){ const dirc=m.dir>0?' ↑':m.dir<0?' ↓':'';
    mx+=`<tr><td class=row title="${esc(m.help||'')}">${esc(m.label)}<span class=rowdir>${dirc}</span>${m.src==='declared'?' <span class="srctag decl" title="manifest 手填值">声明</span>':''}</td>`;
    for(const s of S){ const v=mval(s,m.key), bv=mval(b,m.key);
      const dl=s.id===base?{t:'',c:'zero'}:delta(v,bv,m.dir);
      mx+=`<td class="num ${s.id===base?'isbase':''}"><span class=v>${num(v,d.metric_fmt[m.key])}</span>${dl.t?`<span class="d ${dl.c}">${dl.t}</span>`:''}</td>`; }
    mx+=`</tr>`; }
  mx+=`<tr><td class=row>timeline</td>`+S.map(s=>{const u=tlUrl(s.trace);
    return `<td class=num>${u?`<a target=_blank href="${u}">tracelight ↗</a>`:'<span class="meta dim">—</span>'}</td>`;}).join('')+`</tr>`;
  mx+=`<tr><td class=row>profile</td>`+S.map(s=>`<td class=num><a data-go="#/exp/${encodeURIComponent(s.id)}">打开 →</a></td>`).join('')+`</tr></tbody></table>`;
  // category composition (shared order; baseline tick guides)
  const bbc=(b.analysis||{}).by_category||{}; let acc=0; const guides=d.categories.map(c=>{acc+=(bbc[c]||0);return acc;}).slice(0,-1);
  const catrows=S.map(s=>{const bc=(s.analysis||{}).by_category||{};
    const segs=d.categories.map(c=>{const p=bc[c]||0;return p>0?`<div class=seg data-cat="${esc(c)}" style="width:${p}%;background:${d.category_hue[c]}" data-pct="${p.toFixed(1)}" data-s="${esc(s.label)}"></div>`:'';}).join('');
    return `<div class=catrow><div class=nm data-go="#/exp/${encodeURIComponent(s.id)}">${esc(s.label)}<span class=mb> mbs${(s.config||{}).mbs||'?'}</span></div>
      <div class=catbar>${segs}<div class=guides>${guides.map(g=>`<i style="left:${g}%"></i>`).join('')}</div></div></div>`;}).join('');
  const legend=`<div class=legend>`+d.categories.map(c=>`<span class="item" data-cat="${esc(c)}"><i class=sw style="background:${d.category_hue[c]}"></i>${esc(c)}</span>`).join('')+`</div>`;
  // wall-clock span budget per experiment (idle visible) — the core "why slower" story
  const budMax={}; for(const s of S){const bb=(s.analysis||{}).budget||{}; for(const[k] of BUDGET_SEG)budMax[k]=Math.max(budMax[k]||0,bb[k]||0);}
  const budrows=S.map(s=>{const a=s.analysis||{};
    return `<div class=catrow><div class=nm data-go="#/exp/${encodeURIComponent(s.id)}">${esc(s.label)}<span class=mb> ${(a.idle_pct||0).toFixed(0)}% idle · ${(a.concurrency||0).toFixed(2)}×</span></div>
      <div class=catbar>${a.budget?budgetSegs(a):'<span class="meta dim" style="padding:0 8px">无 trace</span>'}</div></div>`;}).join('');
  // operator diff table (union × experiments, Δ vs base)
  const opmap={};
  for(const s of S)for(const k of ((s.analysis||{}).top_kernels||[])){ (opmap[k.name]=opmap[k.name]||{name:k.name,cat:k.cat,v:{}}).v[s.id]=k.time_pct; }
  const ops=Object.values(opmap).sort((a,b2)=>((b2.v[base]||0)-(a.v[base]||0))).slice(0,20);
  let opt=`<table class=matrix><thead><tr><th class=row>kernel</th>`+S.map(s=>`<th class="num ${s.id===base?'base':''}"><span class=lab>${esc(s.label)}</span></th>`).join('')+`</tr></thead><tbody>`+
    ops.map(o=>`<tr><td class=row title="${esc(o.name)}"><span class=kdot style="background:${d.category_hue[o.cat]||'var(--s-other)'}"></span>${esc(o.name.slice(0,40))}</td>`+
      S.map(s=>{const v=o.v[s.id], bv=o.v[base]; const dl=(s.id===base||v==null)?{t:'',c:'zero'}:delta(v,bv,-1);
        return `<td class="num ${s.id===base?'isbase':''}">${v==null?'<span class="meta dim">—</span>':`<span class=v>${v.toFixed(1)}%</span>${dl.t?`<span class="d ${dl.c}">${dl.t}</span>`:''}`}</td>`;}).join('')+`</tr>`).join('')+`</tbody></table>`;
  // config diff
  let cf=`<table class=matrix><thead><tr><th class=row>参数</th>`+S.map(s=>`<th class="num ${s.id===base?'base':''}"><span class=lab>${esc(s.label)}</span></th>`).join('')+`</tr></thead><tbody>`;
  for(const r of d.config_rows){ cf+=`<tr><td class=row>${esc(r.label)}</td>`; let prev=null;
    for(const s of S){const val=(s.config||{})[r.key]; const bval=(b.config||{})[r.key];
      const chg=prev!==null&&String(val)!==String(prev); const vsb=s.id!==base&&String(val)!==String(bval);
      cf+=`<td class="num ${chg?'chg':''} ${vsb&&!chg?'vsbase':''} ${val==bval&&s.id!==base?'same':''}"><span class=v>${val==null?'—':esc(val)}</span></td>`; prev=val;}
    cf+=`</tr>`; }
  cf+=`</tbody></table>`;
  view().innerHTML=`<div class=nav><a class=ghost data-go="#/">← 实验库</a>
      <span class=t>对比 ${S.length} 个实验</span><span class=sub>基准 = ${esc(base)} · 点列头切换基准</span></div>
    <div class=card><div class=head><h2>核心指标 · 值 + Δ vs 基准</h2></div><div class=scroll style="max-height:none">${mx}</div></div>
    <div class=card><div class=head><h2>墙钟预算对比 · idle / 计算 / 通信（100% = 各自 span）</h2><span class=grow></span><span class="meta dim">每条归一到自己的 span，比的是占比不是绝对墙钟</span></div>
      ${budgetLegend(budMax)}<div class=catwrap>${budrows}</div></div>
    <div class=card><div class=head><h2>KERNEL 时间构成 · 100% GPU-active · 固定类别顺序</h2><span class=grow></span><span class="meta dim">点图例只看一类</span></div>
      ${legend}<div class=catwrap>${catrows}</div></div>
    <div class=card><div class=head><h2>算子 DIFF · union top kernels × 实验 · time% (Δ vs 基准)</h2><span class=grow></span><span class="meta dim">「—」= 不在该实验 top-12，非一定为 0</span></div><div class=scroll style="max-height:none">${opt}</div></div>
    <div class=card><div class=head><h2>配置参数 DIFF · 相邻列变化高亮 · 对基准不同加左边框</h2></div><div class=scroll style="max-height:none">${cf}</div></div>`;
  view().querySelectorAll('[data-go]').forEach(n=>n.onclick=()=>go(n.dataset.go));
  view().querySelectorAll('th[data-base]').forEach(th=>th.onclick=()=>go(`#/compare?sel=${sel.map(encodeURIComponent).join(',')}&base=${encodeURIComponent(th.dataset.base)}`));
  view().querySelectorAll('.legend .item').forEach(it=>it.onclick=()=>{
    const on=it.classList.toggle('off'); it.style.opacity=on?'.4':''; const cat=it.dataset.cat;
    view().querySelectorAll(`.seg[data-cat="${CSS.escape(cat)}"]`).forEach(se=>se.classList.toggle('dim',on));});
}

/* ── place: PROFILE (secondary / detail page) ──────────────── */
async function renderProfile(id){
  const s=await api('/api/exp/'+encodeURIComponent(id)); if(!s){view().innerHTML='<div class=card><div class=empty>无此实验</div></div>';return;}
  telemetry(`<span>experiment <b>${esc(s.label)}</b></span>`);
  const a=s.analysis||{}, cfg=s.config||{}, pf=s.perf||{};
  const tl=tlUrl(s.trace);
  const meter=(pct,bad)=>`<div class=meter><i style="width:${Math.max(0,Math.min(100,pct||0))}%;${bad?'background:var(--warn)':''}"></i></div>`;
  const tile=(k,v,sub,m,flag,tag)=>`<div class="stat ${flag?'flag':''}"><div class=k>${k}${tag?`<span class="srctag ${tag==='声明'?'decl':'meas'}" title="${tag==='声明'?'manifest 手填值':'从 trace 实测'}">${tag}</span>`:''}</div><div class=v>${v}</div>${m||''}${sub?`<div class=sub>${sub}</div>`:''}</div>`;
  const idleHi=a.idle_pct>=35;
  const tiles=[
    tile('tok/s/GPU',num(pf.tok_s,'{:,.0f}'),null,null,false,'声明'),
    tile('TFLOP/s',num(pf.tflops,'{:.1f}'),null,null,false,'声明'),
    tile('MFU',num(pf.mfu,'{:.1%}'),null,meter((pf.mfu||0)*100),false,'声明'),
    tile('GPU idle',num(a.idle_pct,'{:.1f}%'),idleHi?'偏高':'',meter(a.idle_pct,idleHi),idleHi,'实测'),
    tile('暴露通信',num(a.exposed_of_span_pct,'{:.1f}%'),'占 span',meter(a.exposed_of_span_pct),false,'实测'),
    tile('kernel 启动',num(a.launches,'{:,.0f}'),null,null,false,'实测'),
    tile('GPU-active',num(a.gpu_s,'{:.1f}'),'s',null,false,'实测'),
    tile('distinct kernel',num(a.distinct,'{:,.0f}'),null,null,false,'实测'),
  ].join('');
  const cats=Object.keys(a.by_category||{}).length? Object.entries(a.by_category).sort((x,y)=>y[1]-x[1]) : [];
  const hue=catHue;
  const stack=cats.length?`<div class=catbar>`+cats.map(([c,p])=>`<div class=seg data-cat="${esc(c)}" data-pct="${p.toFixed(1)}" data-s="${esc(s.label)}" style="width:${p}%;background:${hue(c)}"></div>`).join('')+`</div>`:'';
  const legend=cats.length?`<div class=legend style="margin-top:9px">`+cats.map(([c])=>`<span class=item style="cursor:default"><i class=sw style="background:${hue(c)}"></i>${esc(c)}</span>`).join('')+`</div>`:'';
  const cattab=cats.map(([c,p])=>`<tr><td><span class=kdot style="background:${hue(c)}"></span>${esc(c)}</td><td class=num>${p.toFixed(1)}%</td><td style="width:42%"><div class=catmini><i style="width:${p}%;background:${hue(c)}"></i></div></td></tr>`).join('');
  const ops=(a.top_kernels||[]).map(k=>`<tr><td class=knm title="${esc(k.name)}">${esc(k.name)}</td><td class=kcat><span class=kdot style="background:${hue(k.cat)}"></span>${esc(k.cat)}</td><td class=num>${k.time_pct.toFixed(1)}%</td><td class=num>${Math.round(k.avg_us)}µs</td><td class=num>${k.count.toLocaleString()}</td></tr>`).join('');
  const cfgrid=Object.entries(cfg).map(([k,v])=>`<div class=cfgrp><span class=kv1>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
  const budget=a.budget?`
    <div class=card><div class=head><h2>墙钟预算 · 每步时间去哪了</h2><span class=grow></span>
        <span class="meta dim">100% = span ${a.span_s||'?'}s</span></div>
      <div class=catbar style="height:28px">${budgetSegs(a)}</div>${budgetLegend(a.budget,true)}
      ${idleDrivers(a)}</div>`:'';
  const opanalysis=cats.length?`
    <div class=card><div class=head><h2>算子分析 · GPU 干活时在算什么（占 GPU-active，idle 已在上方墙钟预算）</h2></div>
      ${stack}${legend}
      <div class=scroll style="max-height:none;margin-top:12px"><table><thead><tr><th>类别</th><th class=num>time%</th><th></th></tr></thead><tbody>${cattab}</tbody></table></div></div>
    <div class=card><div class=head><h2>算子分析 · 算子 top ${(a.top_kernels||[]).length}</h2></div><div class=scroll>
      <table><thead><tr><th>kernel</th><th>类别</th><th class=num>time%</th><th class=num>avg</th><th class=num>count</th></tr></thead><tbody>${ops}</tbody></table></div></div>`
    :`<div class=card><div class=empty>这条实验还没有 trace 实测的算子分析</div></div>`;
  view().innerHTML=`<div class=nav><a class=ghost data-go="#/">← 实验库</a>
      <span class=t>${esc(s.label)}</span><span class=sub>${esc(s.sub||'')}</span><span class=grow></span>
      ${tl?'':'<span class="pill pill-warn">无 trace</span>'}</div>
    <div class=stats>${tiles}</div>
    ${budget}
    <div class=card><div class=head><h2>时间线 · tracelight（内嵌）</h2><span class=grow></span>
        ${tl?`<a class=ghost target=_blank href="${tl}" style="padding:4px 10px;border-radius:5px">在新窗口打开 ↗</a>`:''}</div>
      ${tl?`<div class=tlbar><span class=path>${esc(s.trace)}</span></div>
        <iframe class=tlframe src="${tl}" title="tracelight timeline" loading=lazy></iframe>`
        :`<div class=empty>这条实验没有 trace，无法内嵌时间线</div>`}</div>
    ${opanalysis}
    <div class=card><div class=head><h2>配置参数</h2></div><div class=cfgrid>${cfgrid}</div></div>`;
  view().querySelectorAll('[data-go]').forEach(n=>n.onclick=()=>go(n.dataset.go));
}

/* ── tooltip + chrome ──────────────────────────────────────── */
document.addEventListener('mousemove',e=>{const t=e.target,tip=$('#tip');
  if(t.classList&&t.classList.contains('seg')&&t.dataset.pct){
    tip.innerHTML=`<b>${esc(t.dataset.s||'')}</b><br>${esc(t.dataset.cat)}<br>${t.dataset.pct}% of GPU-active`;
    tip.style.display='block'; tip.style.left=Math.min(e.clientX+14,innerWidth-330)+'px'; tip.style.top=(e.clientY+16)+'px';}
  else tip.style.display='none';});
addEventListener('hashchange',route);
addEventListener('DOMContentLoaded',()=>{
  $('#refresh').onclick=route;
  $('#help').onclick=()=>toast('库→勾选≥2→对比选中 · 点行看单实验 profile · 列头排序/切基准 · 图例只看一类');
  $('#theme').onclick=()=>{const r=document.documentElement;
    r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';};
  route();
});
