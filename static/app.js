/* POLARIS v3 Dashboard — JS Engine with Negotiation */
const C={amber:'#d97706',teal:'#0d9488',rose:'#e11d48',sky:'#0284c7',lime:'#65a30d',violet:'#7c3aed',orange:'#ea580c',indigo:'#4f46e5',emerald:'#059669',grid:'#e4e4e7',tick:'#71717a'};
let ws,ch={},D={s:[],sat:[],gdp:[],pol:[],rew:[],tru:[],tom:[],coal:0,veto:0,bet:0,evt:0,tomTotal:0,tomCorrect:0,tomRewardSum:0,hist:[]};
const MX=250;

function tab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.p===id));document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+id));}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const m={'1':'live','2':'neg','3':'war','4':'causal','5':'metrics','6':'risk','7':'hist'};if(m[e.key])tab(m[e.key]);if(e.key===' '){e.preventDefault();pause();}});

function conn(){
  const proto=location.protocol==='https:'?'wss':'ws';ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{document.getElementById('wsd').className='dot up';document.getElementById('wsl').textContent='ONLINE';};
  ws.onclose=()=>{document.getElementById('wsd').className='dot dn';document.getElementById('wsl').textContent='RECONNECTING';setTimeout(conn,2000);};
  ws.onmessage=e=>{const m=JSON.parse(e.data);({init:onInit,step:onStep,episode_start:onStart,episode_end:onEnd,pause:onPause}[m.type]||(()=>{}))(m);};
}
function tx(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o));}

function go(){tx({cmd:'start',seed:+document.getElementById('seed').value||42,chaos:parseFloat(document.getElementById('v-chaos').textContent),task_id:document.getElementById('sel-task').value});}
function pause(){tx({cmd:'pause'});}
function reset(){clearD();tx({cmd:'reset',seed:+document.getElementById('seed').value||42});}
function spd(v){tx({cmd:'speed',value:v});document.querySelectorAll('.controls .btn').forEach(b=>{if(b.textContent.includes('x'))b.classList.toggle('active',b.textContent.trim()===v+'x');});}
function regime(v){tx({cmd:'regime',regime:v});}
document.getElementById('chaos-r').addEventListener('input',e=>{const v=(e.target.value/100).toFixed(2);document.getElementById('v-chaos').textContent=v;tx({cmd:'chaos',value:+v});});
function scrub(i){i=+i;if(D.hist[i])onStep(D.hist[i],1);}
function exp(){const b=new Blob([JSON.stringify(D.hist,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='polaris_v3_trace.json';a.click();}

function lineOpt(color,yMin,yMax){return{type:'line',data:{labels:[],datasets:[{data:[],borderColor:color,backgroundColor:color+'15',fill:true,tension:.4,pointRadius:0,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{min:yMin,max:yMax,grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}}},plugins:{legend:{display:false}}}};}

function initCharts(){
  ch.sat=new Chart(document.getElementById('c-sat'),lineOpt(C.teal,0,100));
  ch.gdp=new Chart(document.getElementById('c-gdp'),lineOpt(C.sky,0,200));
  ch.pol=new Chart(document.getElementById('c-pol'),lineOpt(C.rose,0,500));
  ch.rt=new Chart(document.getElementById('c-rt'),{type:'line',data:{labels:[],datasets:[
    {label:'Reward',data:[],borderColor:C.amber,tension:.4,pointRadius:0,borderWidth:1.5,fill:false},
    {label:'Trust',data:[],borderColor:C.teal,tension:.4,pointRadius:0,borderWidth:1.5,fill:false,borderDash:[3,3]}
  ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{min:0,max:1,grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}}},plugins:{legend:{position:'top',labels:{color:'#3f3f46',font:{family:'IBM Plex Mono',size:9},boxWidth:12,padding:8}}}}});
  
  ch.inf=new Chart(document.getElementById('c-inf'),{type:'radar',data:{labels:['M0','M1','M2','M3','M4'],datasets:[{data:[.5,.5,.5,.5,.5],borderColor:C.amber,backgroundColor:C.amber+'20',pointBackgroundColor:C.amber,pointRadius:3,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:1,grid:{color:C.grid},angleLines:{color:C.grid},pointLabels:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:10}},ticks:{display:false}}},plugins:{legend:{display:false}},animation:{duration:200}}});
  
  ch.radar=new Chart(document.getElementById('c-radar'),{type:'radar',data:{labels:['Economy','Environment','Social','Health','Cooperation'],datasets:[{label:'Current',data:[.5,.5,.5,.5,.5],borderColor:C.indigo,backgroundColor:C.indigo+'18',borderWidth:1.5},{label:'Baseline',data:[.3,.3,.3,.3,.3],borderColor:'#3f3f46',backgroundColor:'#3f3f4622',borderDash:[3,3],borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:1,grid:{color:C.grid},angleLines:{color:C.grid},pointLabels:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:9}},ticks:{display:false}}},plugins:{legend:{position:'top',labels:{color:'#71717a',font:{family:'IBM Plex Mono',size:9},boxWidth:10,padding:6}}},animation:{duration:200}}});
  
  ch.rbd=new Chart(document.getElementById('c-rbd'),{type:'bar',data:{labels:['Base','Pareto','Coop','ToM','Total'],datasets:[{data:[0,0,0,0,0],backgroundColor:[C.amber,C.lime,C.teal,C.indigo,C.sky],borderRadius:2,barThickness:16}]},options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',scales:{x:{grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}},y:{grid:{display:false},ticks:{color:'#a1a1aa',font:{family:'IBM Plex Mono',size:10}},border:{display:false}}},plugins:{legend:{display:false}},animation:{duration:150}}});

  // v3: ToM accuracy chart
  ch.tom=new Chart(document.getElementById('c-tom'),{type:'line',data:{labels:[],datasets:[{label:'ToM Reward',data:[],borderColor:C.indigo,backgroundColor:C.indigo+'15',fill:true,tension:.4,pointRadius:0,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{min:-.2,max:.3,grid:{color:C.grid,lineWidth:.5},ticks:{color:C.tick,font:{family:'IBM Plex Mono',size:9}},border:{color:C.grid}}},plugins:{legend:{display:false}}}});
}

function clearD(){D={s:[],sat:[],gdp:[],pol:[],rew:[],tru:[],tom:[],coal:0,veto:0,bet:0,evt:0,tomTotal:0,tomCorrect:0,tomRewardSum:0,hist:[]};Object.values(ch).forEach(c=>{if(c.data){c.data.labels=[];c.data.datasets.forEach(d=>d.data=[]);c.update();}});document.getElementById('feed').innerHTML='';document.getElementById('agents').innerHTML='';document.getElementById('neg-feed').innerHTML='';document.getElementById('minister-proposals').innerHTML='<div class="neg-empty">Run a negotiation task to see minister proposals</div>';document.getElementById('agent-decision').innerHTML='<div class="neg-empty">Awaiting agent response...</div>';document.getElementById('vote-result').innerHTML='<div class="neg-empty">No vote yet</div>';document.getElementById('briefing-feed').innerHTML='<div class="neg-empty">No briefings yet</div>';}

function onInit(m){if(m.history)m.history.forEach(s=>onStep(s,1));if(m.episodes)m.episodes.forEach(addRow);}
function onStart(m){clearD();document.getElementById('b-reg').textContent=m.task_id||'NEGOTIATION';}
function onPause(m){document.getElementById('btn-pause').textContent=m.paused?'RESUME':'PAUSE';}

function onStep(s,q){
  D.hist.push(s);
  const push=(a,v)=>{a.push(v);if(a.length>MX)a.shift();};
  const lbl=''+s.step;
  push(D.s,lbl);push(D.sat,s.state.public_satisfaction);push(D.gdp,s.state.gdp_index);push(D.pol,s.state.pollution_index);push(D.rew,s.reward);push(D.tru,s.council.institutional_trust);
  
  const set=(c,d)=>{c.data.labels=D.s.slice();c.data.datasets[0].data=d.slice();c.update();};
  set(ch.sat,D.sat);set(ch.gdp,D.gdp);set(ch.pol,D.pol);
  ch.rt.data.labels=D.s.slice();ch.rt.data.datasets[0].data=D.rew.slice();ch.rt.data.datasets[1].data=D.tru.slice();ch.rt.update();
  
  // top bar
  document.getElementById('v-step').textContent=s.step;
  document.getElementById('v-rew').textContent=s.reward.toFixed(3);
  document.getElementById('v-trust').textContent=s.council.institutional_trust.toFixed(3);
  const st=Math.max(0,Math.min(100,Math.round((s.state.public_satisfaction/100+s.state.gdp_index/200+(1-s.state.pollution_index/500))/3*100)));
  document.getElementById('v-stab').textContent=st+'%';
  const sp=document.getElementById('v-stab').parentElement;
  sp.className='pill '+(st>60?'ok':st>30?'warn':'bad');
  
  // counters
  if(s.council.coalition_formed)D.coal++;
  D.veto+=(s.council.vetoes||[]).length;
  if(s.council.betrayal)D.bet++;
  D.evt+=(s.active_events||[]).length;
  document.getElementById('b-coal').textContent=D.coal;
  document.getElementById('b-veto').textContent=D.veto;
  document.getElementById('b-bet').textContent=D.bet;
  document.getElementById('b-evt').textContent=D.evt;
  
  const iv=s.council.influence_vector||[];
  if(iv.length){document.getElementById('b-inf')||0;ch.inf.data.datasets[0].data=iv;ch.inf.data.labels=iv.map((_,i)=>'M'+i);ch.inf.update();}
  
  // ministers
  const ms=s.council.ministers||[];
  if(ms.length){document.getElementById('agents').innerHTML=ms.map((m,i)=>`<div class="agent a${i}"><div class="name">${m.name||'Agent '+i}</div><div class="meta">${m.role||'general'}</div><div class="bar"><div class="bar-fill" style="width:${(m.influence*100).toFixed(0)}%"></div></div><div class="stats">INF ${(m.influence||0).toFixed(3)} · ${m.proposal||'—'}</div></div>`).join('');}
  
  // ═══════════════════════════════════════════════════════
  // v3: NEGOTIATION TAB RENDERING
  // ═══════════════════════════════════════════════════════
  
  const neg = s.negotiation || {};
  const negOutcome = s.negotiation_outcome || {};
  const proposals = neg.minister_proposals || [];
  
  if(proposals.length > 0) {
    const roleClass = (role) => {
      if(role.includes('Finance')) return 'm-finance';
      if(role.includes('Environment')) return 'm-environment';
      if(role.includes('Health')) return 'm-health';
      if(role.includes('Industry')) return 'm-industry';
      if(role.includes('Social')) return 'm-social';
      return '';
    };
    
    document.getElementById('minister-proposals').innerHTML = proposals.map(p => `
      <div class="minister-card ${roleClass(p.role)}">
        <div class="minister-header">
          <div class="minister-name">${p.emoji || ''} ${p.minister}</div>
          <div class="minister-role">${p.role}</div>
        </div>
        <div class="minister-proposal">
          Proposes: <span class="action-badge">${(p.proposed_action||'').replace(/_/g,' ')}</span>
        </div>
        <div class="minister-argument">"${p.argument}"</div>
        <div class="minister-meta">
          ${p.veto_threat ? '<span class="minister-tag tag-veto">VETO THREAT</span>' : ''}
          <span class="minister-tag tag-trust">Trust: ${Math.round((p.trust_level||0)*100)}%</span>
          ${p.coalition_offer ? '<span class="minister-tag tag-coalition">Coalition</span>' : ''}
        </div>
        ${p.hidden_agenda_hint ? `<div class="minister-intel">${p.hidden_agenda_hint}</div>` : ''}
      </div>
    `).join('');
  }
  
  // Agent decision
  if(s.agent_action) {
    const a = s.agent_action;
    document.getElementById('agent-decision').innerHTML = `
      <div class="decision-action">${(a.action||'').replace(/_/g,' ').toUpperCase()}</div>
      ${a.reasoning ? `<div class="decision-reasoning">${a.reasoning}</div>` : ''}
      ${(a.coalition_target||[]).length ? `<div style="font-size:11px;color:var(--t3);margin-bottom:4px">COALITION:</div><div class="decision-coalition">${a.coalition_target.map(c=>`<span class="coalition-member">${c}</span>`).join('')}</div>` : ''}
      ${(a.veto_prediction||[]).length ? `<div class="decision-veto-pred">Veto prediction: ${a.veto_prediction.join(', ')}</div>` : ''}
    `;
  }
  
  // Vote outcome
  if(negOutcome.support_count !== undefined) {
    const total = (negOutcome.support_count||0) + (negOutcome.oppose_count||0);
    const supPct = total > 0 ? (negOutcome.support_count/total*100) : 50;
    const oppPct = 100 - supPct;
    const statusClass = negOutcome.vetoed ? 'vote-vetoed' : (negOutcome.approved ? 'vote-approved' : 'vote-vetoed');
    const statusText = negOutcome.vetoed ? `VETOED by ${negOutcome.veto_by}` : (negOutcome.approved ? 'APPROVED' : 'REJECTED');
    
    document.getElementById('vote-result').innerHTML = `
      <div class="vote-bar">
        <div class="vote-support" style="width:${supPct}%">${negOutcome.support_count} FOR</div>
        <div class="vote-oppose" style="width:${oppPct}%">${negOutcome.oppose_count} AGAINST</div>
      </div>
      <div class="vote-status ${statusClass}">${statusText}</div>
      <div class="vote-detail">
        <div>Supporters: ${(negOutcome.supporters||[]).join(', ') || 'None'}</div>
        <div>Opposers: ${(negOutcome.opposers||[]).join(', ') || 'None'}</div>
        <div>Coalition formed: ${negOutcome.coalition_formed ? 'Yes' : 'No'}</div>
        <div>Cooperation: ${((negOutcome.cooperation_score||0)*100).toFixed(0)}%</div>
      </div>
    `;
    
    // ToM stats
    if(negOutcome.veto_prediction_correct !== undefined) {
      D.tomTotal++;
      if(negOutcome.veto_prediction_correct) D.tomCorrect++;
      D.tomRewardSum += (negOutcome.tom_reward || 0);
      push(D.tom, negOutcome.tom_reward || 0);
      
      document.getElementById('tom-accuracy').textContent = D.tomTotal > 0 ? Math.round(D.tomCorrect/D.tomTotal*100)+'%' : '--';
      document.getElementById('tom-coalitions').textContent = D.coal;
      document.getElementById('tom-vetoes').textContent = D.veto;
      document.getElementById('tom-reward').textContent = D.tomRewardSum.toFixed(2);
      document.getElementById('v-tom').textContent = D.tomTotal > 0 ? Math.round(D.tomCorrect/D.tomTotal*100)+'%' : '--';
      document.getElementById('b-tom').textContent = D.tomRewardSum.toFixed(2);
      
      ch.tom.data.labels = D.s.slice(-D.tom.length);
      ch.tom.data.datasets[0].data = D.tom.slice();
      ch.tom.update();
    }
    
    // Negotiation feed
    if(!q) {
      const f = document.getElementById('neg-feed');
      const action = (negOutcome.final_action||s.action||'').replace(/_/g,' ');
      let cls = 'neg-msg';
      if(negOutcome.vetoed) cls += ' vetoed';
      else if(negOutcome.coalition_formed) cls += ' coalition';
      else if(negOutcome.approved) cls += ' approved';
      f.innerHTML = `<div class="${cls}"><span class="neg-step">Step ${s.step}</span> <span class="neg-action">${action}</span> · CoopScore: ${((negOutcome.cooperation_score||0)*100).toFixed(0)}% ${negOutcome.vetoed?'· VETOED by '+negOutcome.veto_by:''}${negOutcome.coalition_formed?' · COALITION':''}${negOutcome.tom_reward>0?' · ToM +'+negOutcome.tom_reward.toFixed(2):''}</div>` + f.innerHTML;
      if(f.children.length > 80) f.removeChild(f.lastChild);
    }
  }
  
  // Briefings
  const newBriefing = s.new_briefing || '';
  const activeBriefings = s.active_briefings || [];
  if(newBriefing || activeBriefings.length) {
    const bf = document.getElementById('briefing-feed');
    if(newBriefing && !q) {
      const cat = newBriefing.startsWith('\u{1F534}') ? 'threat' : newBriefing.startsWith('\u{1F7E2}') ? 'opportunity' : 'warning';
      bf.innerHTML = `<div class="briefing-item briefing-${cat}">${newBriefing}</div>` + bf.innerHTML;
      if(bf.children.length > 20) bf.removeChild(bf.lastChild);
    }
  }
  
  // ═══════════════════════════════════════════════════════
  // END v3 NEGOTIATION
  // ═══════════════════════════════════════════════════════
  
  // war room feed
  if(!q){
    const f=document.getElementById('feed');
    const a=s.council_action||s.action;
    let cls='msg',h=`<span class="who">Step ${s.step}</span><span class="ts">${a}</span><br>`;
    if(s.council.coalition_formed){cls+=' coal';h+=`<span class="act">Coalition \u25B8 str ${s.council.coalition_strength.toFixed(2)}</span>`;}
    if((s.council.vetoes||[]).length){cls+=' veto';h+=`<span class="act">${s.council.vetoes.length} veto(s)</span>`;}
    if(s.council.betrayal){cls+=' veto';h+=`<span class="act">\u26A1 BETRAYAL</span>`;}
    f.innerHTML=`<div class="${cls}">${h}</div>`+f.innerHTML;
    if(f.children.length>60)f.removeChild(f.lastChild);
  }
  
  // causal
  const cc=s.explanation.causal_chain||[];
  document.getElementById('chain').innerHTML=cc.map((c,i)=>`<div class="node">${typeof c==='string'?c:JSON.stringify(c)}</div>${i<cc.length-1?'<div class="arrow">\u2193</div>':''}`).join('');
  document.getElementById('narr').textContent=s.explanation.nl_narrative||'\u2014';
  const cfs=s.explanation.counterfactuals||[];
  document.getElementById('cfs').innerHTML=cfs.slice(0,5).map(c=>`<div class="cf"><strong>${c.action||'Alt'}</strong> <span class="delta ${(c.reward_delta||0)>=0?'pos':'neg'}">${(c.reward_delta||0)>=0?'+':''}${(c.reward_delta||0).toFixed(4)}</span> <span style="color:var(--t4)">${c.explanation||''}</span></div>`).join('');
  const al=s.explanation.risk_alerts||[];
  document.getElementById('alerts').innerHTML=al.length?al.map(a=>`<div class="alert-box">${typeof a==='string'?a:a.message||JSON.stringify(a)}</div>`).join(''):'<span style="color:var(--t4)">Clear</span>';
  
  // KPIs
  document.getElementById('k-surv').textContent=st+'%';
  document.getElementById('k-coop').textContent=(D.coal/Math.max(s.step,1)).toFixed(2);
  document.getElementById('k-osc').textContent=D.hist.length>=3?D.hist.slice(-20).filter((_,i,a)=>i>=2&&a[i]&&a[i-2]&&a[i].action===a[i-2].action&&a[i].action!==(a[i-1]||{}).action).length:'0';
  document.getElementById('k-align').textContent=(s.explanation.alignment_score||50).toFixed(0);
  document.getElementById('k-pareto').textContent=s.state.public_satisfaction>20&&s.state.gdp_index>20?'OK':'RISK';
  
  ch.radar.data.datasets[0].data=[Math.min(1,s.state.gdp_index/150),Math.min(1,1-s.state.pollution_index/500),Math.min(1,s.state.public_satisfaction/100),Math.min(1,(s.state.healthcare_index||50)/100),Math.min(1,D.coal/Math.max(s.step,1))];
  ch.radar.update();
  const rb=s.reward_breakdown||{};
  ch.rbd.data.datasets[0].data=[rb.base||0,rb.pareto||0,rb.cooperation||0,rb.tom_reward||0,rb.total||0];ch.rbd.update();
  
  // heatmap
  const rk=(v,lo,hi)=>{const n=(v-lo)/(hi-lo);return n<.3?'lo':n<.6?'md':n<.85?'hi':'cr';};
  document.getElementById('hmap').innerHTML=[['SAT',rk(100-s.state.public_satisfaction,0,100),s.state.public_satisfaction.toFixed(0)],['GDP',rk(200-s.state.gdp_index,0,200),s.state.gdp_index.toFixed(0)],['POLL',rk(s.state.pollution_index,0,500),s.state.pollution_index.toFixed(0)],['HEALTH',rk(100-(s.state.healthcare_index||50),0,100),(s.state.healthcare_index||50).toFixed(0)]].map(([l,c,v])=>`<div class="hcell ${c}"><span class="hl">${l}</span>${v}</div>`).join('');
  
  const W=[];
  if(s.state.public_satisfaction<30)W.push('Satisfaction critical ('+s.state.public_satisfaction.toFixed(0)+')');
  if(s.state.gdp_index<40)W.push('GDP declining ('+s.state.gdp_index.toFixed(0)+')');
  if(s.state.pollution_index>300)W.push('Pollution dangerous ('+s.state.pollution_index.toFixed(0)+')');
  if(s.council.institutional_trust<.3)W.push('Trust collapsing');
  document.getElementById('warns').innerHTML=W.length?W.map(w=>`<div class="warn-item">\u25B8 ${w}</div>`).join(''):'<span style="color:var(--t4)">Clear</span>';
  document.getElementById('evts').innerHTML=(s.active_events||[]).length?(s.active_events||[]).map(e=>`<div style="padding:3px 0;border-bottom:1px solid var(--b1);font-family:IBM Plex Mono;font-size:11px">${typeof e==='string'?e:e.name||JSON.stringify(e)}</div>`).join(''):'<span style="color:var(--t4)">None</span>';
  
  document.getElementById('sc-r').max=D.hist.length-1;document.getElementById('sc-r').value=D.hist.length-1;document.getElementById('sc-hi').textContent=D.hist.length;
}

function onEnd(m){
  addRow(m);
  const f=document.getElementById('feed');
  f.innerHTML=`<div class="msg ${m.collapsed?'veto':'coal'}"><span class="who">END</span><span class="ts">${m.steps} steps</span><br><span class="act">Score: ${m.score} \u00B7 ${m.collapsed?'COLLAPSED':'SURVIVED'}</span></div>`+f.innerHTML;
}
function addRow(e){
  const t=document.getElementById('eptb');
  const r=document.createElement('tr');
  r.innerHTML=`<td>${e.timestamp||'\u2014'}</td><td>${e.seed}</td><td>${e.task_id||'—'}</td><td>${(e.chaos||0).toFixed(1)}</td><td>${e.steps}</td><td>${(e.score||0).toFixed(4)}</td><td style="color:${e.collapsed?'var(--rose)':'var(--lime)'}"> ${e.collapsed?'COLLAPSED':'SURVIVED'}</td>`;
  t.prepend(r);
}

initCharts();conn();
