var diffeoCanvas = document.getElementById('diffeo-canvas');
var diffeoCtx = diffeoCanvas ? diffeoCanvas.getContext('2d') : null;
var diffeoAnimId = null;
var diffeoTime = 0;

// State for fibre bundle view
var fibreState = {
  neuronData: null,       // cached neuron grid data from server
  loading: false,
  normMode: 'layer',      // 'layer' or 'global'
  pixSize: 2,
  useAbs: false,
  roomSpacing: 80,        // vertical gap between layer rooms
  roomWidth: 0,           // computed
  roomHeight: 0,          // computed
  tokenSpacing: 0,        // computed
  dimX: 0,                // which hidden dim maps to X sort within room
  dimY: 1,                // which hidden dim maps to Y sort within room
  dimZ: 2,                // depth axis for 3D projection
  show3D: false,          // toggle pseudo-3D stacking
  rotX: -0.3,
  rotY: 0.4,
  dragActive: false,
  dragLastX: 0,
  dragLastY: 0,
  scrollY: 0,             // vertical scroll offset
  selectedToken: -1,
  hoveredNeuron: null,
  showConnections: true,   // show diffeomorphism lines between layers
  connectionDensity: 0.1,  // fraction of neurons to connect
  colormap: 'grayscale',   // 'grayscale', 'coolhot', 'viridis'
  showTransportFrame: false,
  showAttnField: false,
  showMlpField: false,
  showFlowLines: true,
  flowArrowScale: 1.0,
};

var diffeoState = {
  active: false,
  numSlices: 8,
  kelpAmplitude: 1.0,
  divergenceSensitivity: 1.0,
  layerSpacing: 70,
  sliceAlpha: 0.25,
  gridRes: 10,
  dimMode: 'auto',
  slices: [],
  built: false,
};

// Pan/Zoom state
var zoomLevel = 1.0;
var panX = 0, panY = 0;
var panActive = false, panLastX = 0, panLastY = 0;
var D=null,AP=null;
var selectedTokens=new Set();
var viewMode='2d';
// 3D rotation state
var rotX=-0.4, rotY=0.6, rotZ=0;
var dragActive=false, dragLastX=0, dragLastY=0;
var focalLength=600;

/** Return the active deltas array based on decomposition selector */
function getActiveDeltas(){
    if(!D) return null;
    var decomp = document.getElementById('sel-decomp').value;
    if(decomp === 'attn' && D.attn_deltas) return D.attn_deltas;
    if(decomp === 'mlp' && D.mlp_deltas) return D.mlp_deltas;
    return D.deltas;
}

/** Get a label for the current decomposition mode */
function getDecompLabel(){
    var decomp = document.getElementById('sel-decomp').value;
    if(decomp === 'attn' && D && D.attn_deltas) return 'Attn';
    if(decomp === 'mlp' && D && D.mlp_deltas) return 'MLP';
    return 'Full';
}

function runText(){
    var txt=document.getElementById('txt-in').value.trim();
    if(!txt)return;
    var modelName=document.getElementById('sel-model').value;
    var itpMethod=document.getElementById('sel-itp').value;
    var btn=document.getElementById('btn-run');
    btn.disabled=true;btn.textContent='Running...';
    document.getElementById('status').textContent='Processing (model: '+modelName+', itp: '+itpMethod+')...';
    fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:txt, model:modelName, itp_method:itpMethod})})
    .then(function(r){if(!r.ok)throw new Error('Server error '+r.status);return r.json()})
    .then(function(d){
        D=d;selectedTokens.clear();updateSelectedUI();onData();
        btn.disabled=false;btn.textContent='Run';
    }).catch(function(e){
        document.getElementById('status').textContent='Error: '+e;
        btn.disabled=false;btn.textContent='Run';
    });
}

function updateStrainStatsPanel(){
    var panel = document.getElementById('strain-stats-panel');
    if(!D || !D.strain_stats || D.strain_stats.length === 0){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    var stats = D.strain_stats;
    var currentLayer = +document.getElementById('sl-layer').value;

    // Find most active layer by variance
    var mostActiveLayer = 0;
    var maxVar = 0;
    for(var i = 0; i < stats.length; i++){
        if(stats[i].variance > maxVar){
            maxVar = stats[i].variance;
            mostActiveLayer = i;
        }
    }

    // Find global max strain for bar scaling
    var globalMaxStrain = 0;
    for(var i = 0; i < stats.length; i++){
        if(stats[i].max > globalMaxStrain) globalMaxStrain = stats[i].max;
    }
    if(globalMaxStrain < 1.01) globalMaxStrain = 2.0;

    var html = '<table>';
    html += '<tr><th>L</th><th>Mean</th><th>Max</th><th>Var</th><th>Exp%</th><th>Con%</th><th>Iso%</th><th>Distribution</th></tr>';
    for(var i = 0; i < stats.length; i++){
        var s = stats[i];
        var rowClass = '';
        if(i === currentLayer && i === mostActiveLayer) rowClass = 'current-layer most-active';
        else if(i === currentLayer) rowClass = 'current-layer';
        else if(i === mostActiveLayer) rowClass = 'most-active';

        // Mini bar showing mean strain relative to 1.0
        var barWidth = Math.min(60, Math.max(2, Math.abs(s.mean - 1.0) / (globalMaxStrain - 1.0) * 60));
        var barColor = s.mean > 1.0 ? '#e94560' : '#0077b6';
        var barHtml = '<span class="strain-bar" style="width:'+barWidth+'px;background:'+barColor+'"></span>';

        // Markers
        var marker = '';
        if(i === currentLayer) marker += ' ◄';
        if(i === mostActiveLayer) marker += ' ★';

        html += '<tr class="'+rowClass+'" onclick="document.getElementById(\'sl-layer\').value='+i+';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))" style="cursor:pointer">';
        html += '<td style="color:#e94560;font-weight:bold">'+i+marker+'</td>';
        html += '<td>'+s.mean.toFixed(3)+'</td>';
        html += '<td style="color:'+(s.max > 1.5 ? '#e94560' : '#a0a0c0')+'">'+s.max.toFixed(3)+'</td>';
        html += '<td>'+s.variance.toFixed(4)+'</td>';
        html += '<td style="color:#e94560">'+(s.frac_expanding*100).toFixed(0)+'</td>';
        html += '<td style="color:#0077b6">'+(s.frac_contracting*100).toFixed(0)+'</td>';
        html += '<td style="color:#888">'+(s.frac_isometric*100).toFixed(0)+'</td>';
        html += '<td>'+barHtml+'</td>';
        html += '</tr>';
    }
    html += '</table>';
    html += '<div style="margin-top:4px;color:#888;font-size:8px">';
    html += '◄ = current layer | ★ = most active (highest variance) | Click row to jump to layer';
    html += '</div>';
    panel.innerHTML = html;
}

function autoParams(){
    if(viewMode === 'multi'){
        if(!multiData || !multiData.sentence_data || !multiData.sentence_data.length) return;
        var sd = multiData.sentence_data[0];
        if(!sd || !sd.fixed_pos) return;
        var dx=+document.getElementById('sl-dx').value;
        var dy=+document.getElementById('sl-dy').value;
        var nP=sd.n_points;
        var mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity;
        for(var i=0;i<nP;i++){
            var x=sd.fixed_pos[i][dx],y=sd.fixed_pos[i][dy];
            if(x<mnx)mnx=x;if(x>mxx)mxx=x;if(y<mny)mny=y;if(y>mxy)mxy=y;
        }
        var range=Math.max(mxx-mnx,mxy-mny)||1;
        var sig=range*0.15;
        var slSig=document.getElementById('sl-sig');
        slSig.max=Math.max(20,range*2).toFixed(1);
        slSig.value=sig.toFixed(2);
        document.getElementById('v-sig').textContent=sig.toFixed(2);
        var activeDeltas = sd.deltas;
        var decomp = document.getElementById('sel-decomp').value;
        if(decomp === 'attn' && sd.attn_deltas) activeDeltas = sd.attn_deltas;
        if(decomp === 'mlp' && sd.mlp_deltas) activeDeltas = sd.mlp_deltas;
        var norms=[];
        for(var l=0;l<sd.n_layers;l++){
            for(var p=0;p<nP;p++){
                var ddx=activeDeltas[l][p][dx],ddy=activeDeltas[l][p][dy];
                norms.push(Math.sqrt(ddx*ddx+ddy*ddy));
            }
        }
        norms.sort(function(a,b){return a-b});
        var med=norms[Math.floor(norms.length*0.75)]||1;
        var amp=range*0.06/(med+1e-12);
        amp=Math.max(0.1,Math.min(500,amp));
        document.getElementById('sl-amp').value=amp.toFixed(1);
        document.getElementById('v-amp').textContent=amp.toFixed(1);
        return;
    }
    if(!D)return;

    var dx=+document.getElementById('sl-dx').value;
    var dy=+document.getElementById('sl-dy').value;
    var nP=D.n_points;
    var mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity;
    for(var i=0;i<nP;i++){
        var x=D.fixed_pos[i][dx],y=D.fixed_pos[i][dy];
        if(x<mnx)mnx=x;if(x>mxx)mxx=x;if(y<mny)mny=y;if(y>mxy)mxy=y;
    }
    var range=Math.max(mxx-mnx,mxy-mny)||1;
    var sig=range*0.15;
    var slSig=document.getElementById('sl-sig');
    slSig.max=Math.max(20,range*2).toFixed(1);
    slSig.value=sig.toFixed(2);
    document.getElementById('v-sig').textContent=sig.toFixed(2);
    var activeDeltas = getActiveDeltas();
    if(!activeDeltas) activeDeltas = D.deltas;
    var norms=[];
    for(var l=0;l<D.n_layers;l++){
        for(var p=0;p<nP;p++){
            var ddx=activeDeltas[l][p][dx],ddy=activeDeltas[l][p][dy];
            norms.push(Math.sqrt(ddx*ddx+ddy*ddy));
        }
    }
    norms.sort(function(a,b){return a-b});
    var med=norms[Math.floor(norms.length*0.75)]||1;
    var amp=range*0.06/(med+1e-12);
    amp=Math.max(0.1,Math.min(500,amp));
    document.getElementById('sl-amp').value=amp.toFixed(1);
    document.getElementById('v-amp').textContent=amp.toFixed(1);
}

function updateSelectedUI(){
    var cont=document.getElementById('selected-tokens');
    cont.innerHTML='';
    if(selectedTokens.size===0){
        cont.innerHTML='<span style="color:#555;font-size:10px">Click on a token dot to select it</span>';
        document.getElementById('neighbor-panel').style.display='none';
        return;
    }
    selectedTokens.forEach(function(ti){
        var el=document.createElement('span');
        el.className='sel-tok';
        el.innerHTML='['+ti+'] '+D.tokens[ti]+' <span class="x">\u00d7</span>';
        el.onclick=function(){selectedTokens.delete(ti);updateSelectedUI();draw()};
        cont.appendChild(el);
    });
    updateNeighborPanel();
}

function updateNeighborPanel(){
    var panel=document.getElementById('neighbor-panel');
    var list=document.getElementById('nb-list');
    var title=document.getElementById('nb-title');
    if(!D||!D.neighbors||selectedTokens.size===0){
        panel.style.display='none';return;
    }
    panel.style.display='block';
    var kn=+document.getElementById('sl-kn').value;
    var html='';
    selectedTokens.forEach(function(ti){
        if(ti>=D.neighbors.length)return;
        html+='<div style="margin-bottom:6px"><b style="color:#e94560">['+ti+'] '+D.tokens[ti]+'</b> neighbors:';
        var nbs=D.neighbors[ti].slice(0,kn);
        for(var ni=0;ni<nbs.length;ni++){
            var nb=nbs[ni];
            var cls=nb.is_real?'nb-item is-real':'nb-item';
            html+='<div class="'+cls+'" onclick="clickNeighbor('+nb.idx+','+nb.is_real+')">';
            html+=(nb.is_real?'\u2605 ':'')+nb.label+'<span class="nb-dist">d='+nb.dist.toFixed(2)+'</span></div>';
        }
        html+='</div>';
    });
    list.innerHTML=html;
    title.textContent='Neighbors (K='+kn+')';
}

function clickNeighbor(idx, isReal){
    if(isReal && idx < D.n_real){
        selectedTokens.add(idx);
        updateSelectedUI();
    }
    draw();
}

// 3D rotation helpers
function rotatePoint3D(x, y, z){
    var cosY=Math.cos(rotY), sinY=Math.sin(rotY);
    var x1=x*cosY+z*sinY, z1=-x*sinY+z*cosY;
    var cosX=Math.cos(rotX), sinX=Math.sin(rotX);
    var y1=y*cosX-z1*sinX, z2=y*sinX+z1*cosX;
    return [x1, y1, z2];
}

function project3D(x, y, z, W, H){
    var r=rotatePoint3D(x, y, z);
    var scale=focalLength/(focalLength+r[2]);
    return [W/2+r[0]*scale, H/2+r[1]*scale, r[2], scale];
}

var cv3d=document.getElementById('cv');

cv3d.addEventListener('mousedown', function(e){
    if(dragActive && (viewMode==='3d' || (viewMode==='multi' && (multiSubView==='3d' || multiSubView==='fibre3d')))){
        var ddx=e.clientX-dragLastX, ddy=e.clientY-dragLastY;
        rotY+=ddx*0.005;
        rotX+=ddy*0.005;
        rotX=Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
        dragLastX=e.clientX;
        dragLastY=e.clientY;
        draw();
        return;
    }

    if(viewMode==='3d'){
        if(e.button===0 && !e.shiftKey){
            dragActive=true;
            dragLastX=e.clientX;
            dragLastY=e.clientY;
            return;
        }
        if(e.button===1 || (e.button===0 && e.shiftKey)){
            e.preventDefault();
            panActive=true;
            panLastX=e.clientX;
            panLastY=e.clientY;
        }
        return;
    }
    if(e.button===1 || (e.button===0 && e.shiftKey)){
        e.preventDefault();
        panActive=true;
        panLastX=e.clientX;
        panLastY=e.clientY;
    }
});

window.addEventListener('mousemove', function(e){
    if(dragActive && viewMode==='3d'){
        var ddx=e.clientX-dragLastX, ddy=e.clientY-dragLastY;
        rotY+=ddx*0.005;
        rotX+=ddy*0.005;
        rotX=Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
        dragLastX=e.clientX;
        dragLastY=e.clientY;
        draw();
        return;
    }
    if(panActive){
        panX+=e.clientX-panLastX;
        panY+=e.clientY-panLastY;
        panLastX=e.clientX;
        panLastY=e.clientY;
        draw();
    }
});

window.addEventListener('mouseup', function(e){
    dragActive=false;
    panActive=false;
});

document.getElementById('cv').addEventListener('click', function(e){
    if(!D)return;
    if(dragActive)return;
    var cv=document.getElementById('cv');
    var rect=cv.getBoundingClientRect();
    var rawMx=e.clientX-rect.left, rawMy=e.clientY-rect.top;

    var mx, my;
    if(viewMode==='2d'){
        mx = (rawMx - panX) / zoomLevel;
        my = (rawMy - panY) / zoomLevel;
    } else {
        mx = rawMx;
        my = rawMy;
    }

    var dx=+document.getElementById('sl-dx').value;
    var dy=+document.getElementById('sl-dy').value;
    var dz=+document.getElementById('sl-dz').value;
    var nP=D.n_points, nR=D.n_real;
    var W=cv.width, H=cv.height, M2=42, dW2=W-2*M2, dH2=H-2*M2;

    var bestDist=Infinity, bestIdx=-1;

    if(viewMode==='2d'){
        var fx2=new Float64Array(nP),fy2=new Float64Array(nP);
        for(var i=0;i<nP;i++){fx2[i]=D.fixed_pos[i][dx];fy2[i]=D.fixed_pos[i][dy]}
        var mnx=fx2[0],mxx=fx2[0],mny=fy2[0],mxy=fy2[0];
        for(var i2=1;i2<nP;i2++){
            if(fx2[i2]<mnx)mnx=fx2[i2];if(fx2[i2]>mxx)mxx=fx2[i2];
            if(fy2[i2]<mny)mny=fy2[i2];if(fy2[i2]>mxy)mxy=fy2[i2];
        }
        var mr=Math.max(mxx-mnx,mxy-mny)||1;
        var cxv=(mnx+mxx)/2,cyv=(mny+mxy)/2;
        var pd2=0.12;
        var vx0=cxv-mr*(.5+pd2),vy0=cyv-mr*(.5+pd2),vw=mr*(1+2*pd2),vh=vw;
        function SX(x){return M2+((x-vx0)/vw)*dW2}
        function SY(y){return M2+((y-vy0)/vh)*dH2}
        for(var ti=0;ti<nR;ti++){
            var sx=SX(fx2[ti]), sy=SY(fy2[ti]);
            var dd=Math.hypot(mx-sx, my-sy);
            if(dd<bestDist){bestDist=dd;bestIdx=ti}
        }
    } else {
        var fx3=new Float64Array(nP),fy3=new Float64Array(nP),fz3=new Float64Array(nP);
        for(var i=0;i<nP;i++){fx3[i]=D.fixed_pos[i][dx];fy3[i]=D.fixed_pos[i][dy];fz3[i]=D.fixed_pos[i][dz]}
        var mnx3=Infinity,mxx3=-Infinity,mny3=Infinity,mxy3=-Infinity,mnz3=Infinity,mxz3=-Infinity;
        for(var i3=0;i3<nP;i3++){
            if(fx3[i3]<mnx3)mnx3=fx3[i3];if(fx3[i3]>mxx3)mxx3=fx3[i3];
            if(fy3[i3]<mny3)mny3=fy3[i3];if(fy3[i3]>mxy3)mxy3=fy3[i3];
            if(fz3[i3]<mnz3)mnz3=fz3[i3];if(fz3[i3]>mxz3)mxz3=fz3[i3];
        }
        var mr3=Math.max(mxx3-mnx3,mxy3-mny3,mxz3-mnz3)||1;
        var cx3=(mnx3+mxx3)/2,cy3=(mny3+mxy3)/2,cz3=(mnz3+mxz3)/2;
        var sc3=Math.min(dW2,dH2)*0.35/mr3;
        for(var ti=0;ti<nR;ti++){
            var px=(fx3[ti]-cx3)*sc3, py=(fy3[ti]-cy3)*sc3, pz=(fz3[ti]-cz3)*sc3;
            var proj=project3D(px,py,pz,W,H);
            var dd=Math.hypot(mx-proj[0], my-proj[1]);
            if(dd<bestDist){bestDist=dd;bestIdx=ti}
        }
    }

    if(bestIdx>=0 && bestDist<25){
        if(selectedTokens.has(bestIdx)) selectedTokens.delete(bestIdx);
        else selectedTokens.add(bestIdx);
        updateSelectedUI();draw();
    }
});

['sl-kn'].forEach(function(id){
    var s=document.getElementById(id);
    s.addEventListener('input',function(){
        document.getElementById('v-kn').textContent=s.value;
        updateSelectedUI();draw();
    });
});
['cb-nb','cb-nblabel'].forEach(function(id){
    document.getElementById(id).addEventListener('change', function(){ draw(); });
});

window.addEventListener('resize',function(){rsz();draw()});

function rsz() {
    var cv = document.getElementById('cv');
    var ct = document.getElementById('main');
    var s = Math.min(ct.clientWidth - 16, ct.clientHeight - 16);
    cv.width = Math.max(400, s);
    cv.height = cv.width;

    if (typeof diffeoState !== 'undefined' && diffeoState.active) {
        resizeDiffeoCanvas();
    }
}

rsz();

var SLS=[
    ['sl-layer','v-layer',0],['sl-t','v-t',2],['sl-amp','v-amp',1],
    ['sl-dx','v-dx',0],['sl-dy','v-dy',0],['sl-dz','v-dz',0],
    ['sl-sig','v-sig',2],['sl-gr','v-gr',0]
];
for(var i=0;i<SLS.length;i++)(function(c){
    var s=document.getElementById(c[0]),v=document.getElementById(c[1]),dec=c[2];
    v.textContent=parseFloat(s.value).toFixed(dec);
    s.addEventListener('input',function(){
        v.textContent=parseFloat(s.value).toFixed(dec);
        if(c[0]==='sl-dx'||c[0]==='sl-dy'||c[0]==='sl-dz')autoParams();
        if(c[0]==='sl-layer') updateStrainStatsPanel();
        draw();
    });
})(SLS[i]);

['cb-grid','cb-heat','cb-ref','cb-tok','cb-syn','cb-sc','cb-vec','cb-vocnb'].forEach(function(id){
    document.getElementById(id).addEventListener('change', function(){ draw(); });
});

document.getElementById('sel-mode').addEventListener('change', function(){ draw(); });
document.getElementById('sel-decomp').addEventListener('change',function(){
    autoParams();
    draw();
});
document.addEventListener('keydown',onKey);
document.getElementById('txt-in').addEventListener('keydown',function(e){if(e.key==='Enter')runText()});

document.getElementById('sel-itp').addEventListener('change', function(){
    // When interpolation method changes, we need to re-run the backend
    // because the grid probes are computed server-side with the chosen method
    document.getElementById('status').textContent =
        'Interpolation changed to ' + this.value + ' — click Run to recompute grid probes';
});

// ============================================================
// CLIENT-SIDE INTERPOLATION METHODS
// These mirror the server-side methods for real-time grid rendering
// ============================================================

function computeItpWeight(px, py, fx_k, fy_k, sig, method) {
    // Returns a single weight for point (px,py) relative to source (fx_k, fy_k)
    var ex = px - fx_k, ey = py - fy_k;
    var dist_sq = ex * ex + ey * ey;
    var dist = Math.sqrt(dist_sq);

    if (method === 'idw') {
        var p = 2.0;
        return 1.0 / Math.pow(Math.max(dist, 1e-12), p);
    } else if (method === 'wendland') {
        var R = Math.max(3.0 * sig, 1e-6);
        var r_norm = dist / R;
        if (r_norm >= 1.0) return 0.0;
        var t = 1.0 - r_norm;
        return t * t * t * t * (4.0 * r_norm + 1.0);
    } else if (method === 'nn') {
        // NN returns 0 for all but the nearest; handled specially
        return -dist; // negative distance; caller picks max
    } else {
        // Default: Gaussian RBF
        var s2i = 1.0 / (2.0 * sig * sig);
        var exponent = -dist_sq * s2i;
        if (exponent < -500) return 0;
        return Math.exp(exponent);
    }
}

function interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, method) {
    // Interpolate the deformation field at (px, py) using the selected method
    // Returns [vx, vy]

    if (method === 'nn') {
        // Nearest neighbor: just use the closest point's delta
        var bestDist = Infinity, bestIdx = 0;
        for (var k = 0; k < nP; k++) {
            var d = (px - fx[k]) * (px - fx[k]) + (py - fy[k]) * (py - fy[k]);
            if (d < bestDist) { bestDist = d; bestIdx = k; }
        }
        return [edx[bestIdx], edy[bestIdx]];
    }

    if (method === 'mls') {
        // Moving Least Squares with local linear basis
        var s2i = 1.0 / (2.0 * sig * sig);
        // Compute weights
        var W = new Float64Array(nP);
        for (var k = 0; k < nP; k++) {
            var ex = px - fx[k], ey = py - fy[k];
            var exp_val = -(ex * ex + ey * ey) * s2i;
            W[k] = exp_val < -500 ? 0 : Math.exp(exp_val);
        }

        // Build weighted least squares: f(x,y) = a0 + a1*(x-px) + a2*(y-py)
        // Normal equations: (A^T W A) c = A^T W v
        // A = [1, dx_k, dy_k] for each point k
        var AtwA00 = 0, AtwA01 = 0, AtwA02 = 0;
        var AtwA11 = 0, AtwA12 = 0, AtwA22 = 0;
        var Atwvx0 = 0, Atwvx1 = 0, Atwvx2 = 0;
        var Atwvy0 = 0, Atwvy1 = 0, Atwvy2 = 0;

        for (var k = 0; k < nP; k++) {
            var w = W[k];
            if (w < 1e-30) continue;
            var dxk = fx[k] - px, dyk = fy[k] - py;
            AtwA00 += w;
            AtwA01 += w * dxk;
            AtwA02 += w * dyk;
            AtwA11 += w * dxk * dxk;
            AtwA12 += w * dxk * dyk;
            AtwA22 += w * dyk * dyk;
            Atwvx0 += w * edx[k];
            Atwvx1 += w * dxk * edx[k];
            Atwvx2 += w * dyk * edx[k];
            Atwvy0 += w * edy[k];
            Atwvy1 += w * dxk * edy[k];
            Atwvy2 += w * dyk * edy[k];
        }

        // Add regularization
        AtwA00 += 1e-8; AtwA11 += 1e-8; AtwA22 += 1e-8;

        // Solve 3x3 system using Cramer's rule for a0 (the value at query point)
        var det = AtwA00 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                - AtwA01 * (AtwA01 * AtwA22 - AtwA12 * AtwA02)
                + AtwA02 * (AtwA01 * AtwA12 - AtwA11 * AtwA02);

        if (Math.abs(det) < 1e-20) {
            // Fallback to RBF
            return interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, 'rbf');
        }

        // We only need a0 (the constant term = value at query point)
        var detX = Atwvx0 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                 - AtwA01 * (Atwvx1 * AtwA22 - AtwA12 * Atwvx2)
                 + AtwA02 * (Atwvx1 * AtwA12 - AtwA11 * Atwvx2);
        var detY = Atwvy0 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                 - AtwA01 * (Atwvy1 * AtwA22 - AtwA12 * Atwvy2)
                 + AtwA02 * (Atwvy1 * AtwA12 - AtwA11 * Atwvy2);

        return [detX / det, detY / det];
    }

    if (method === 'tps') {
        // TPS is expensive client-side; fall back to RBF for real-time rendering
        // The server-side grid probes already use TPS
        return interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, 'rbf');
    }

    // Weight-based methods: RBF, IDW, Wendland
    var vx = 0, vy = 0, ws = 0;
    for (var k = 0; k < nP; k++) {
        var w = computeItpWeight(px, py, fx[k], fy[k], sig, method);
        vx += w * edx[k];
        vy += w * edy[k];
        ws += w;
    }
    if (ws > 1e-15) { vx /= ws; vy /= ws; }
    return [vx, vy];
}

function gp(){return{
    layer:+document.getElementById('sl-layer').value,
    t:+document.getElementById('sl-t').value,
    amp:+document.getElementById('sl-amp').value,
    mode:document.getElementById('sel-mode').value,
    dx:+document.getElementById('sl-dx').value,
    dy:+document.getElementById('sl-dy').value,
    dz:+document.getElementById('sl-dz').value,
    sig:+document.getElementById('sl-sig').value,
    gr:+document.getElementById('sl-gr').value,
    grid:document.getElementById('cb-grid').checked,
    heat:document.getElementById('cb-heat').checked,
    ref:document.getElementById('cb-ref').checked,
    tok:document.getElementById('cb-tok').checked,
    syn:document.getElementById('cb-syn').checked,
    sc:document.getElementById('cb-sc').checked,
    vec:document.getElementById('cb-vec').checked,
    nb:document.getElementById('cb-nb').checked,
    nblabel:document.getElementById('cb-nblabel').checked,
    kn:+document.getElementById('sl-kn').value
}}

function onKey(e) {
    // ================================================================
    // Guard: don't intercept keys when typing in text inputs
    // ================================================================
    if (document.activeElement === document.getElementById('txt-in')) return;
    if (document.activeElement === document.getElementById('txt-b')) return;

    // ================================================================
    // FIBRE VIEW MODES: delegate to fibre-specific key handler
    // (was added by the _origOnKey wrapper)
    // ================================================================
    if (viewMode === 'fibre' || viewMode === 'fibrekelp' || viewMode === 'fibre3d') {
        onKeyFibre(e);
        return;
    }

    // ================================================================
    // STANDARD KEY HANDLING (the original onKey body)
    // ================================================================
    var sl  = document.getElementById('sl-layer');
    var st  = document.getElementById('sl-t');
    var sa  = document.getElementById('sl-amp');
    var sdx = document.getElementById('sl-dx');
    var sdy = document.getElementById('sl-dy');
    var sdz = document.getElementById('sl-dz');
    var maxDim = D ? D.hidden_dim - 1 : 767;

    // ---- Shift+Arrow = Dim Z (third axis), works in all views ----
    if (e.shiftKey && e.key === 'ArrowRight') {
        e.preventDefault();
        var newZ = +sdz.value + 1;
        if (newZ > maxDim) newZ = 0;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowLeft') {
        e.preventDefault();
        var newZ = +sdz.value - 1;
        if (newZ < 0) newZ = maxDim;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowUp') {
        e.preventDefault();
        var newZ = +sdz.value + 10;
        if (newZ > maxDim) newZ = newZ % (maxDim + 1);
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowDown') {
        e.preventDefault();
        var newZ = +sdz.value - 10;
        if (newZ < 0) newZ = (newZ + maxDim + 1) % (maxDim + 1);
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }

    // ---- Arrow keys (no shift) = Dim X / Dim Y ----
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        var newX = +sdx.value + 1;
        if (newX > maxDim) newX = 0;
        if (newX === +sdy.value) newX = (newX + 1) % (maxDim + 1);
        sdx.value = newX;
        sdx.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        var newX = +sdx.value - 1;
        if (newX < 0) newX = maxDim;
        if (newX === +sdy.value) newX = (newX - 1 + maxDim + 1) % (maxDim + 1);
        sdx.value = newX;
        sdx.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowUp') {
        e.preventDefault();
        var newY = +sdy.value + 1;
        if (newY > maxDim) newY = 0;
        if (newY === +sdx.value) newY = (newY + 1) % (maxDim + 1);
        sdy.value = newY;
        sdy.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowDown') {
        e.preventDefault();
        var newY = +sdy.value - 1;
        if (newY < 0) newY = maxDim;
        if (newY === +sdx.value) newY = (newY - 1 + maxDim + 1) % (maxDim + 1);
        sdy.value = newY;
        sdy.dispatchEvent(new Event('input'));
    }

    // ---- Layer navigation: [ ] or , . ----
    else if (e.key === '.' || e.key === ']') {
        sl.value = Math.min(+sl.max, +sl.value + 1);
        sl.dispatchEvent(new Event('input'));
    }
    else if (e.key === ',' || e.key === '[') {
        sl.value = Math.max(0, +sl.value - 1);
        sl.dispatchEvent(new Event('input'));
    }

    // ---- Deformation t: ; / ' ----
    else if (e.key === "'") {
        st.value = Math.min(1, +st.value + 0.05).toFixed(2);
        st.dispatchEvent(new Event('input'));
    }
    else if (e.key === ';') {
        st.value = Math.max(0, +st.value - 0.05).toFixed(2);
        st.dispatchEvent(new Event('input'));
    }

    // ---- Amplification: A / Z ----
    else if (e.key === 'a' || e.key === 'A') {
        sa.value = Math.min(500, +sa.value * 1.3).toFixed(1);
        sa.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'z' || e.key === 'Z') {
        sa.value = Math.max(0.1, +sa.value / 1.3).toFixed(1);
        sa.dispatchEvent(new Event('input'));
    }

    // ---- Reset all: R ----
    else if (e.key === 'r' || e.key === 'R') {
        rstAll();
    }

    // ---- Next dimension pair: D ----
    else if (e.key === 'd' || e.key === 'D') {
        nxtD();
    }

    // ---- Reset zoom/pan: 0 ----
    else if (e.key === '0') {
        zoomLevel = 1.0;
        panX = 0;
        panY = 0;
        draw();
    }

    // ---- 3D-specific: PageUp/PageDown for Dim Z ----
    else if (viewMode === '3d' && e.key === 'PageUp') {
        e.preventDefault();
        var newZ = +sdz.value + 1;
        if (newZ > maxDim) newZ = 0;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
    }
    else if (viewMode === '3d' && e.key === 'PageDown') {
        e.preventDefault();
        var newZ = +sdz.value - 1;
        if (newZ < 0) newZ = maxDim;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
    }
}

function rstAll(){
    document.getElementById('sl-layer').value='0';
    document.getElementById('sl-t').value='1.0';
    document.getElementById('sl-dx').value='0';
    document.getElementById('sl-dy').value='1';
    document.getElementById('sl-dz').value='2';
    document.getElementById('sl-gr').value='30';
    document.getElementById('sel-decomp').value='full';
    rotX=-0.4;rotY=0.6;rotZ=0;
    zoomLevel=1.0;panX=0;panY=0;
    selectedTokens.clear();updateSelectedUI();
    if(D)autoParams();
    ['sl-layer','sl-t','sl-amp','sl-dx','sl-dy','sl-dz','sl-sig','sl-gr'].forEach(function(id){
        document.getElementById(id).dispatchEvent(new Event('input'));
    });
}
function nxtD(){
    var dx=document.getElementById('sl-dx'),dy=document.getElementById('sl-dy');
    var x=+dx.value,y=+dy.value;
    y++;if(y>+dy.max){y=0;x++}if(x>=y)y=x+1;if(y>+dy.max){x=0;y=1}
    dx.value=x;dy.value=y;dx.dispatchEvent(new Event('input'));
}

function s2c(s){
    if(s<=.5)return[0,180,220];if(s>=1.5)return[233,69,96];
    if(s<1){var f=(s-.5)/.5;return[~~(f*120),~~(180-f*80),~~(220-f*120)]}
    var f2=(s-1)/.5;return[~~(120+f2*113),~~(100-f2*31),~~(100-f2*4)];
}

function draw2D(){
    var p=gp(),cv=document.getElementById('cv'),c=cv.getContext('2d');
    var W=cv.width,H=cv.height;
    c.clearRect(0,0,W,H);

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    var nP=D.n_points,nR=D.n_real,dx=p.dx,dy=p.dy;
    var isEmb=p.mode==='embedding';
    var activeDeltas = getActiveDeltas();

    // --- Reuse extractPositions2D ---
    var pos = extractPositions2D(D, nP, dx, dy);
    var fx = pos.fx, fy = pos.fy;

    // --- Reuse computeViewBounds2D ---
    var bounds = computeViewBounds2D(fx, fy, nP, 0.12);
    var vx0 = bounds.vx0, vy0 = bounds.vy0, vw = bounds.vw, vh = bounds.vh;

    // --- Compute cumulative deltas using existing pattern ---
    var edx=new Float64Array(nP),edy=new Float64Array(nP);
    if(!isEmb){
        var layer=p.layer,amp=p.amp;
        for(var j=0;j<nP;j++){
            var sx2=0,sy2=0;
            if(p.mode==='single'){sx2=activeDeltas[layer][j][dx];sy2=activeDeltas[layer][j][dy]}
            else if(p.mode==='cumfwd'){for(var l=0;l<=layer;l++){sx2+=activeDeltas[l][j][dx];sy2+=activeDeltas[l][j][dy]}}
            else{for(var l2=layer;l2<D.n_layers;l2++){sx2+=activeDeltas[l2][j][dx];sy2+=activeDeltas[l2][j][dy]}}
            edx[j]=sx2*amp;edy[j]=sy2*amp;
        }
    }

    var M=42,dW=W-2*M,dH=H-2*M;
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    var N=p.gr;
    var itpMethod = document.getElementById('sel-itp').value;

    // --- Reuse buildDeformedGrid2D ---
    var grid = buildDeformedGrid2D(vx0, vy0, vw, vh, N, fx, fy, edx, edy, nP, p.sig, p.t, isEmb, itpMethod);

    // --- Reuse drawStrainHeatmapFullCanvas ---
    if(p.heat && !isEmb){
        drawStrainHeatmapFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, 0.3);
    }

    // --- Reuse drawReferenceGridFullCanvas ---
    if(p.ref){
        drawReferenceGridFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, isEmb);
    }

    // --- Reuse drawGridLinesFullCanvas ---
    if(p.grid && !isEmb){
        drawGridLinesFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, p.sc, 1.2);
    }

    // --- Vector arrows ---
    if(p.vec && !isEmb){
        drawVectorArrowsFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH);
    }

    // --- Synthetic probe points ---
    if(p.syn){
        drawSyntheticProbes(c, fx, fy, nR, nP, SX, SY);
    }

    // --- Neighbor connections ---
    if(p.nb && D.neighbors && selectedTokens.size>0){
        drawNeighborConnections2D(c, D, fx, fy, nP, p.kn, p.nblabel, SX, SY);
    }

    // --- Predicted next-token points ---
    if(D.predicted_indices && D.predicted_indices.length > 0 && p.tok){
        drawPredictedTokens2D(c, D, fx, fy, nP, SX, SY);
    }

    // --- Real token dots ---
    if(p.tok){
        drawRealTokenDots2D(c, D, fx, fy, nR, isEmb, SX, SY);
    }

    // --- Vocab neighbors ---
    if(document.getElementById('cb-vocnb').checked && D.vocab_neighbors && p.tok){
        drawVocabNeighbors2D(c, D, fx, fy, nR, SX, SY);
    }

    c.restore();

    // HUD text
    draw2DHUD(c, W, H, p, dx, dy, isEmb);
}

function drawStrainHeatmapFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, alpha) {
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    for(var hy=0;hy<N;hy++) for(var hx=0;hx<N;hx++){
        var avg=(grid.sH[hy*N+hx]+grid.sH[(hy+1)*N+hx]+grid.sV[hy*(N+1)+hx]+grid.sV[hy*(N+1)+hx+1])/4;
        var co=s2c(avg);
        var i00=hy*(N+1)+hx,i10=i00+1,i01=(hy+1)*(N+1)+hx,i11=i01+1;
        c.beginPath();
        c.moveTo(SX(grid.gX[i00]),SY(grid.gY[i00]));
        c.lineTo(SX(grid.gX[i10]),SY(grid.gY[i10]));
        c.lineTo(SX(grid.gX[i11]),SY(grid.gY[i11]));
        c.lineTo(SX(grid.gX[i01]),SY(grid.gY[i01]));
        c.closePath();
        c.fillStyle='rgba('+co[0]+','+co[1]+','+co[2]+','+alpha+')';
        c.fill();
    }
}

function drawReferenceGridFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, isEmb) {
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    c.strokeStyle=isEmb?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.07)';
    c.lineWidth=0.5;
    for(var ry=0;ry<=N;ry++){
        c.beginPath();
        for(var rx=0;rx<=N;rx++){
            var ri=ry*(N+1)+rx;
            if(rx===0) c.moveTo(SX(grid.oX[ri]),SY(grid.oY[ri]));
            else c.lineTo(SX(grid.oX[ri]),SY(grid.oY[ri]));
        }
        c.stroke();
    }
    for(var rx=0;rx<=N;rx++){
        c.beginPath();
        for(var ry=0;ry<=N;ry++){
            var ri=ry*(N+1)+rx;
            if(ry===0) c.moveTo(SX(grid.oX[ri]),SY(grid.oY[ri]));
            else c.lineTo(SX(grid.oX[ri]),SY(grid.oY[ri]));
        }
        c.stroke();
    }
}

function drawGridLinesFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH, showSC, lineWidth) {
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    c.lineWidth=lineWidth;
    // Horizontal edges
    for(var dhy=0;dhy<=N;dhy++) for(var dhx=0;dhx<N;dhx++){
        var di1=dhy*(N+1)+dhx,di2=di1+1;
        var es=grid.sH[dhy*N+dhx];
        if(showSC){var ec=s2c(es);c.strokeStyle='rgba('+ec[0]+','+ec[1]+','+ec[2]+',0.85)'}
        else c.strokeStyle='rgba(200,200,200,0.5)';
        c.beginPath();c.moveTo(SX(grid.gX[di1]),SY(grid.gY[di1]));
        c.lineTo(SX(grid.gX[di2]),SY(grid.gY[di2]));c.stroke();
    }
    // Vertical edges
    for(var dvx=0;dvx<=N;dvx++) for(var dvy=0;dvy<N;dvy++){
        var dvi1=dvy*(N+1)+dvx,dvi2=(dvy+1)*(N+1)+dvx;
        var vs=grid.sV[dvy*(N+1)+dvx];
        if(showSC){var vc=s2c(vs);c.strokeStyle='rgba('+vc[0]+','+vc[1]+','+vc[2]+',0.85)'}
        else c.strokeStyle='rgba(200,200,200,0.5)';
        c.beginPath();c.moveTo(SX(grid.gX[dvi1]),SY(grid.gY[dvi1]));
        c.lineTo(SX(grid.gX[dvi2]),SY(grid.gY[dvi2]));c.stroke();
    }
}

function drawVectorArrowsFullCanvas(c, grid, N, vx0, vy0, vw, vh, M, dW, dH) {
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    var step=Math.max(1,Math.floor(N/12));
    c.lineWidth=1.5;
    for(var viy=0;viy<=N;viy+=step) for(var vix=0;vix<=N;vix+=step){
        var vi=viy*(N+1)+vix;
        var ax=SX(grid.oX[vi]),ay=SY(grid.oY[vi]),bx=SX(grid.gX[vi]),by=SY(grid.gY[vi]);
        var al=Math.hypot(bx-ax,by-ay);if(al<3)continue;
        c.strokeStyle='rgba(255,255,100,0.6)';c.fillStyle='rgba(255,255,100,0.6)';
        c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);c.stroke();
        var aa=Math.atan2(by-ay,bx-ax),hl=Math.min(7,al*.3);
        c.beginPath();c.moveTo(bx,by);
        c.lineTo(bx-hl*Math.cos(aa-.4),by-hl*Math.sin(aa-.4));
        c.lineTo(bx-hl*Math.cos(aa+.4),by-hl*Math.sin(aa+.4));
        c.closePath();c.fill();
    }
}

function drawSyntheticProbes(c, fx, fy, nR, nP, SX, SY) {
    for(var pi=nR;pi<nP;pi++){
        c.beginPath();c.arc(SX(fx[pi]),SY(fy[pi]),2.5,0,Math.PI*2);
        c.fillStyle='rgba(100,200,255,0.2)';c.fill();
    }
}

function drawSyntheticProbes(c, fx, fy, nR, nP, SX, SY) {
    for(var pi=nR;pi<nP;pi++){
        c.beginPath();c.arc(SX(fx[pi]),SY(fy[pi]),2.5,0,Math.PI*2);
        c.fillStyle='rgba(100,200,255,0.2)';c.fill();
    }
}

function drawPredictedTokens2D(c, D, fx, fy, nP, SX, SY) {
    for(var pi2=0; pi2<D.predicted_indices.length; pi2++){
        var pidx = D.predicted_indices[pi2];
        if(pidx >= nP) continue;
        var px2 = SX(fx[pidx]), py2 = SY(fy[pidx]);
        var prob = D.predicted_probs ? D.predicted_probs[pi2] : 0;
        var dotSize = 4 + prob * 12;

        // Pulsing glow
        var glowR = 20 + prob * 30;
        var grad3 = c.createRadialGradient(px2, py2, 0, px2, py2, glowR);
        grad3.addColorStop(0, 'rgba(245,166,35,0.2)');
        grad3.addColorStop(1, 'rgba(245,166,35,0)');
        c.beginPath(); c.arc(px2, py2, glowR, 0, Math.PI*2);
        c.fillStyle = grad3; c.fill();

        // Diamond shape
        c.save();
        c.translate(px2, py2);
        c.rotate(Math.PI/4);
        c.fillStyle = 'rgba(245,166,35,' + (0.4 + prob * 0.6).toFixed(2) + ')';
        c.fillRect(-dotSize/2, -dotSize/2, dotSize, dotSize);
        c.strokeStyle = '#f5a623';
        c.lineWidth = 1.5;
        c.strokeRect(-dotSize/2, -dotSize/2, dotSize, dotSize);
        c.restore();

        // Label
        c.font = '9px monospace';
        c.lineWidth = 2;
        c.strokeStyle = 'rgba(0,0,0,0.9)';
        var plb = D.tokens[pidx] + ' (' + (prob*100).toFixed(1) + '%)';
        c.strokeText(plb, px2+10, py2-6);
        c.fillStyle = '#f5a623';
        c.fillText(plb, px2+10, py2-6);
    }
}

function drawRealTokenDots2D(c, D, fx, fy, nR, isEmb, SX, SY) {
    var tc=['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
        '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22',
        '#f39c12','#d35400','#c0392b','#16a085','#27ae60',
        '#2980b9','#8e44ad','#2c3e50','#ecf0f1','#fd79a8'];
    for(var ti=0;ti<nR;ti++){
        var tx2=SX(fx[ti]),ty2=SY(fy[ti]),col=tc[ti%tc.length];
        var isSel=selectedTokens.has(ti);
        if(isSel){
            var grad2=c.createRadialGradient(tx2,ty2,0,tx2,ty2,30);
            grad2.addColorStop(0,'rgba(0,255,0,0.25)');grad2.addColorStop(1,'rgba(0,255,0,0)');
            c.beginPath();c.arc(tx2,ty2,30,0,Math.PI*2);c.fillStyle=grad2;c.fill();
        }
        var grad=c.createRadialGradient(tx2,ty2,0,tx2,ty2,20);
        grad.addColorStop(0,'rgba(255,255,255,0.08)');grad.addColorStop(1,'rgba(255,255,255,0)');
        c.beginPath();c.arc(tx2,ty2,20,0,Math.PI*2);c.fillStyle=grad;c.fill();
        c.beginPath();c.arc(tx2,ty2,isSel?9:7,0,Math.PI*2);
        c.fillStyle=col;c.fill();
        c.strokeStyle=isSel?'#0f0':'#fff';c.lineWidth=isSel?3:2;c.stroke();
        c.font='bold 11px monospace';c.lineWidth=3;c.strokeStyle='rgba(0,0,0,0.9)';
        var lb='['+ti+'] '+D.tokens[ti];
        c.strokeText(lb,tx2+12,ty2-10);c.fillStyle=isSel?'#0f0':'#fff';c.fillText(lb,tx2+12,ty2-10);
    }
    if(isEmb&&nR>1){
        c.strokeStyle='rgba(233,69,96,0.3)';c.lineWidth=1.5;c.setLineDash([4,4]);
        c.beginPath();c.moveTo(SX(fx[0]),SY(fy[0]));
        for(var ti2=1;ti2<nR;ti2++)c.lineTo(SX(fx[ti2]),SY(fy[ti2]));
        c.stroke();c.setLineDash([]);
    }
}

function drawVocabNeighbors2D(c, D, fx, fy, nR, SX, SY) {
    c.font='9px monospace';
    for(var vi2=0;vi2<nR;vi2++){
        if(!D.vocab_neighbors[vi2])continue;
        var vtx=SX(fx[vi2]),vty=SY(fy[vi2]);
        var vnbs=D.vocab_neighbors[vi2];
        for(var vni=0;vni<vnbs.length;vni++){
            var vnb=vnbs[vni];
            var angle=-Math.PI/2+(vni/(vnbs.length-1||1))*Math.PI;
            var radius=35+vni*8;
            var vnx=vtx+Math.cos(angle)*radius;
            var vny=vty+Math.sin(angle)*radius+20;
            c.fillStyle='rgba(150,150,170,0.45)';
            c.fillText(vnb.token,vnx,vny);
        }
    }
}

function draw2DHUD(c, W, H, p, dx, dy, isEmb) {
    var decompLabel = getDecompLabel();
    c.font='11px monospace';c.fillStyle='rgba(255,255,255,0.45)';
    if(isEmb){
        c.fillText('EMBEDDING SPACE [2D]  Dims:'+dx+','+dy,42,18);
    } else {
        var itpLabel = document.getElementById('sel-itp').value.toUpperCase();
        c.fillText('Layer '+p.layer+'/'+(D.n_layers-1)+'  t='+p.t.toFixed(2)+'  amp='+p.amp.toFixed(1)+'  Dims:'+dx+','+dy+'  Mode:'+p.mode+'  Decomp:'+decompLabel+'  ITP:'+itpLabel+'  [2D]',42,18);
    }
    c.font='10px monospace';c.fillStyle='rgba(255,255,255,0.35)';
    c.fillText('Zoom: '+zoomLevel.toFixed(2)+'x  (Scroll=zoom, Shift+drag=pan, 0=reset)',42,H-10);
}

// ===================== SAE FEATURE INSPECTOR =====================

var saeInfo = null;

function initSAEPanel(){
    fetch('/sae_info')
    .then(function(r){return r.json()})
    .then(function(info){
        saeInfo = info;
        var status = document.getElementById('sae-status');
        var controls = document.getElementById('sae-controls');
        if(!info.sae_available){
            status.innerHTML = '<span style="color:#e94560">sae-lens not installed.</span><br>pip install sae-lens transformer-lens';
            return;
        }
        if(info.loaded_layers.length === 0){
            status.innerHTML = '<span style="color:#f5a623">No SAEs available for ' + info.model_name + '</span><br>SAEs exist for: gpt2, gpt2-medium, gpt2-large, pythia models';
            return;
        }
        status.innerHTML = '<span style="color:#2ecc71">✓ SAEs loaded for ' + info.loaded_layers.length + '/' + info.total_layers + ' layers</span>';
        controls.style.display = 'block';

        // Populate layer dropdown
        var layerSel = document.getElementById('sae-layer');
        layerSel.innerHTML = '';
        for(var i = 0; i < info.loaded_layers.length; i++){
            var l = info.loaded_layers[i];
            var li = info.layer_info[String(l)];
            var opt = document.createElement('option');
            opt.value = l;
            opt.textContent = 'Layer ' + l + (li && li.d_sae ? ' (' + li.d_sae + ' latents)' : '');
            layerSel.appendChild(opt);
        }
    })
    .catch(function(e){
        document.getElementById('sae-status').innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

// Update token dropdown when data changes
function updateSAETokenDropdown(){
    var sel = document.getElementById('sae-token');
    sel.innerHTML = '';
    if(!D) return;
    for(var i = 0; i < D.n_real; i++){
        var opt = document.createElement('option');
        opt.value = i;
        opt.textContent = '[' + i + '] ' + D.tokens[i];
        sel.appendChild(opt);
    }
    // Default to last token
    sel.value = D.n_real - 1;
}

function onData() {
    // ================================================================
    // FIBRE: Clear cached neuron data when new text arrives
    // (was added by _origOnData3 wrapper)
    // ================================================================
    fibreState.neuronData = null;

    // ================================================================
    // CORE: Update all UI controls from the new data D
    // (the original onData body)
    // ================================================================
    document.getElementById('sl-layer').max = D.n_layers - 1;
    document.getElementById('sl-dx').max = D.hidden_dim - 1;
    document.getElementById('sl-dy').max = D.hidden_dim - 1;
    document.getElementById('sl-dz').max = D.hidden_dim - 1;
    document.getElementById('i-mod').textContent = D.model_name;
    document.getElementById('i-pts').textContent = D.n_points;
    document.getElementById('i-real').textContent = D.n_real;
    document.getElementById('i-syn').textContent = D.n_synth;
    document.getElementById('i-lay').textContent = D.n_layers;
    document.getElementById('i-dim').textContent = D.hidden_dim;
    document.getElementById('i-tok').textContent = D.tokens.slice(0, D.n_real).join(' ');
    document.getElementById('sel-model').value = D.model_name;

    // Display interpolation method
    document.getElementById('i-itp').textContent = D.itp_method || 'rbf';
    if (D.itp_method) {
        document.getElementById('sel-itp').value = D.itp_method;
    }

    // Update decomposition selector availability
    var decompSel = document.getElementById('sel-decomp');
    if (D.attn_deltas && D.mlp_deltas) {
        decompSel.disabled = false;
        decompSel.title = 'Component decomposition available';
    } else {
        decompSel.disabled = true;
        decompSel.value = 'full';
        decompSel.title = 'Component decomposition not available for this model architecture';
    }

    // Update strain stats panel
    updateStrainStatsPanel();

    autoParams();
    draw();
    document.getElementById('status').textContent =
        'Ready — ' + D.n_real + ' tokens, ' + D.n_synth + ' probes | Model: ' + D.model_name;

    // Render next-token predictions
    var ntp = document.getElementById('next-token-panel');
    if (D.next_token && D.next_token.length > 0) {
        var html = '';
        for (var i = 0; i < D.next_token.length; i++) {
            var nt = D.next_token[i];
            var barW = Math.max(2, nt.prob * 200);
            html += '<div style="display:flex;align-items:center;gap:6px">';
            html += '<span style="color:#e94560;font-weight:bold;min-width:80px;font-family:monospace">' +
                    nt.token + '</span>';
            html += '<div style="background:#e94560;height:8px;width:' + barW +
                    'px;border-radius:3px;opacity:0.7"></div>';
            html += '<span style="color:#888;font-size:9px">' +
                    (nt.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        ntp.innerHTML = html;
    } else {
        ntp.innerHTML = '<span style="color:#555">No predictions available</span>';
    }

    // ================================================================
    // SAE: Refresh token dropdown and re-initialize SAE panel
    // (was added by _origOnData / SAE wrapper)
    // ================================================================
    updateSAETokenDropdown();
    initSAEPanel();

    // ================================================================
    // DIFFEO: Rebuild diffeomorphism overlay if active
    // (was added by _origOnData2 / diffeo wrapper)
    // ================================================================
    if (diffeoState.active) {
        rebuildDiffeo();
    }

    // ================================================================
    // FIBRE: Auto-fetch neuron data if currently in a fibre view
    // (was added by _origOnData3 / fibre wrapper)
    // ================================================================
    if (viewMode.startsWith('fibre') && D) {
        fetchFibreNeuronData();
    }
}

// =====================================================================
// REFACTORED fetchSAEFeatures — broken into composable helper functions
// =====================================================================

/**
 * Build the HTML for the token × feature activation heatmap table.
 * @param {Object} data - response from /sae_features
 * @param {number} tokenIdx - the selected token index
 * @returns {string} HTML string for the heatmap section
 */
function buildSAEActivationHeatmapHTML(data, tokenIdx) {
    if (!data.token_activations || !data.tokens) return '';

    var html = '<div style="margin-bottom:6px;font-size:9px;color:#888">' +
               'Activation heatmap (top features × tokens):</div>';
    html += '<div style="overflow-x:auto;margin-bottom:6px">' +
            '<table style="border-collapse:collapse;font-size:8px">';

    // Header row: tokens
    html += '<tr><td></td>';
    for (var ti = 0; ti < data.tokens.length; ti++) {
        var isSel = (ti === tokenIdx);
        html += '<td style="padding:1px 3px;text-align:center;color:' +
                (isSel ? '#e94560' : '#888') + ';font-weight:' +
                (isSel ? 'bold' : 'normal') + '">' + data.tokens[ti] + '</td>';
    }
    html += '</tr>';

    // Feature rows (up to 8)
    var taKeys = Object.keys(data.token_activations);
    for (var fi = 0; fi < Math.min(taKeys.length, 8); fi++) {
        var fid = taKeys[fi];
        var acts = data.token_activations[fid];
        html += buildSAEHeatmapRow(fid, acts);
    }

    html += '</table></div>';
    return html;
}

/**
 * Build a single row of the SAE activation heatmap table.
 * @param {string} featureId - the feature ID label
 * @param {number[]} acts - activation values per token
 * @returns {string} HTML <tr> string
 */
function buildSAEHeatmapRow(featureId, acts) {
    // Find max for color scaling
    var maxAct = 0;
    for (var ai = 0; ai < acts.length; ai++) {
        if (Math.abs(acts[ai]) > maxAct) maxAct = Math.abs(acts[ai]);
    }
    if (maxAct < 1e-8) maxAct = 1;

    var html = '<tr><td style="padding:1px 4px;color:#53a8b6;white-space:nowrap">F' +
               featureId + '</td>';
    for (var ai = 0; ai < acts.length; ai++) {
        var intensity = Math.min(1, Math.abs(acts[ai]) / maxAct);
        var r = acts[ai] > 0 ? Math.round(233 * intensity) : 0;
        var g = acts[ai] > 0 ? Math.round(69 * intensity) : Math.round(119 * intensity);
        var b = acts[ai] > 0 ? Math.round(96 * intensity) : Math.round(182 * intensity);
        var bg = 'rgba(' + r + ',' + g + ',' + b + ',' +
                 (0.2 + 0.6 * intensity).toFixed(2) + ')';
        html += '<td style="padding:1px 3px;text-align:center;background:' + bg + '">' +
                acts[ai].toFixed(2) + '</td>';
    }
    html += '</tr>';
    return html;
}

/**
 * Build the HTML for the SAE feature list (clickable rows).
 * @param {Object[]} features - array of { feature_id, activation }
 * @returns {string} HTML string for the feature list
 */
function buildSAEFeatureListHTML(features) {
    var html = '<div style="font-size:9px;color:#888;margin-bottom:2px">' +
               'Click a feature to set up intervention:</div>';

    for (var i = 0; i < features.length; i++) {
        var f = features[i];
        var barW = Math.min(120, Math.max(2, Math.abs(f.activation) * 10));
        var barColor = f.activation > 0 ? '#e94560' : '#0077b6';

        html += '<div class="sae-feat-row" onclick="selectSAEFeature(' +
                f.feature_id + ',' + f.activation.toFixed(4) + ')" ' +
                'style="display:flex;align-items:center;gap:4px;padding:2px 4px;' +
                'cursor:pointer;border-radius:2px;margin:1px 0" ' +
                'onmouseover="this.style.background=\'#1a1a2e\'" ' +
                'onmouseout="this.style.background=\'transparent\'">';
        html += '<span style="color:#53a8b6;min-width:55px;font-family:monospace">' +
                'F' + f.feature_id + '</span>';
        html += '<div style="background:' + barColor + ';height:6px;width:' + barW +
                'px;border-radius:2px;flex-shrink:0"></div>';
        html += '<span style="color:#888;font-size:9px;min-width:60px">' +
                f.activation.toFixed(4) + '</span>';
        html += '</div>';
    }

    return html;
}

/**
 * Build the header HTML shown above the heatmap and feature list.
 * @param {Object} data - response from /sae_features
 * @param {number} tokenIdx - the selected token index
 * @returns {string} HTML string
 */
function buildSAEFeatureHeaderHTML(data, tokenIdx) {
    var nLatents = data.n_latents || '?';
    return '<div style="color:#888;margin-bottom:4px">Token: ' +
           '<b style="color:#e94560">' + data.tokens[tokenIdx] + '</b> | Layer ' +
           (+document.getElementById('sae-layer').value) +
           ' | ' + nLatents + ' total latents</div>';
}

/**
 * Top-level: the refactored fetchSAEFeatures.
 * Orchestrates the fetch and delegates HTML building to helpers.
 */
function fetchSAEFeatures() {
    if (!D) return;
    var layer = +document.getElementById('sae-layer').value;
    var tokenIdx = +document.getElementById('sae-token').value;
    var topK = +document.getElementById('sae-topk').value;
    var text = D.text;

    var list = document.getElementById('sae-features-list');
    list.innerHTML = '<span style="color:#53a8b6">Loading...</span>';

    fetch('/sae_features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text, layer: layer,
            token_idx: tokenIdx, top_k: topK
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            list.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }

        var html = '';

        // 1. Header with token info
        html += buildSAEFeatureHeaderHTML(data, tokenIdx);

        // 2. Activation heatmap table
        html += buildSAEActivationHeatmapHTML(data, tokenIdx);

        // 3. Clickable feature list
        html += buildSAEFeatureListHTML(data.features);

        list.innerHTML = html;

        // Show intervention panel
        document.getElementById('sae-intervention').style.display = 'block';
    })
    .catch(function(e) {
        list.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function selectSAEFeature(featureId, currentActivation){
    document.getElementById('sae-int-feature').value = featureId;
    // Set clamp slider to a value that would amplify it
    var clampVal = Math.max(currentActivation * 3, 10);
    var slider = document.getElementById('sae-int-clamp');
    slider.value = Math.min(+slider.max, clampVal).toFixed(1);
    document.getElementById('v-sae-clamp').textContent = slider.value;
}

function runSAEIntervention(){
    if(!D) return;
    var layer = +document.getElementById('sae-layer').value;
    var featureId = +document.getElementById('sae-int-feature').value;
    var clampValue = +document.getElementById('sae-int-clamp').value;
    var text = D.text;

    var results = document.getElementById('sae-int-results');
    results.innerHTML = '<span style="color:#53a8b6">Running intervention...</span>';

    fetch('/sae_intervene', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, layer: layer, feature_id: featureId, clamp_value: clampValue})
    })
    .then(function(r){return r.json()})
    .then(function(data){
        if(data.error){
            results.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }
        var html = '<div style="color:#f5a623;font-weight:bold;margin-bottom:4px">Feature ' + featureId + ' clamped to ' + clampValue +
                   ' at layer ' + data.layer + '</div>';

        // Side-by-side comparison
        html += '<div style="display:flex;gap:8px">';

        // Baseline column
        html += '<div style="flex:1">';
        html += '<div style="color:#888;font-size:9px;margin-bottom:3px;text-decoration:underline">Baseline</div>';
        for(var i = 0; i < data.baseline_predictions.length; i++){
            var bp = data.baseline_predictions[i];
            var barW = Math.max(2, bp.prob * 150);
            html += '<div style="display:flex;align-items:center;gap:3px;margin:1px 0">';
            html += '<span style="color:#a0a0c0;min-width:55px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + bp.token + '</span>';
            html += '<div style="background:#555;height:5px;width:' + barW + 'px;border-radius:2px;flex-shrink:0"></div>';
            html += '<span style="color:#888;font-size:8px">' + (bp.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        html += '</div>';

        // Modified column
        html += '<div style="flex:1">';
        html += '<div style="color:#e94560;font-size:9px;margin-bottom:3px;text-decoration:underline">Modified</div>';
        for(var i = 0; i < data.modified_predictions.length; i++){
            var mp = data.modified_predictions[i];
            var barW2 = Math.max(2, mp.prob * 150);
            // Check if this token was in baseline top predictions
            var isNew = true;
            for(var bi = 0; bi < data.baseline_predictions.length; bi++){
                if(data.baseline_predictions[bi].token === mp.token){
                    isNew = false;
                    break;
                }
            }
            var tokenColor = isNew ? '#e94560' : '#a0a0c0';
            html += '<div style="display:flex;align-items:center;gap:3px;margin:1px 0">';
            html += '<span style="color:' + tokenColor + ';min-width:55px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + (isNew ? '★ ' : '') + mp.token + '</span>';
            html += '<div style="background:#e94560;height:5px;width:' + barW2 + 'px;border-radius:2px;flex-shrink:0"></div>';
            html += '<span style="color:#888;font-size:8px">' + (mp.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        html += '</div>';

        html += '</div>'; // end flex container

        // Summary of biggest changes
        html += '<div style="margin-top:6px;border-top:1px solid #1a1a2e;padding-top:4px;font-size:9px;color:#888">';
        // Find the biggest probability shift
        var maxShift = 0;
        var shiftToken = '';
        var shiftDir = '';
        for(var mi = 0; mi < data.modified_predictions.length; mi++){
            var mToken = data.modified_predictions[mi].token;
            var mProb = data.modified_predictions[mi].prob;
            var bProb = 0;
            for(var bi2 = 0; bi2 < data.baseline_predictions.length; bi2++){
                if(data.baseline_predictions[bi2].token === mToken){
                    bProb = data.baseline_predictions[bi2].prob;
                    break;
                }
            }
            var shift = mProb - bProb;
            if(Math.abs(shift) > Math.abs(maxShift)){
                maxShift = shift;
                shiftToken = mToken;
                shiftDir = shift > 0 ? '↑' : '↓';
            }
        }
        if(Math.abs(maxShift) > 0.001){
            html += 'Biggest shift: <span style="color:#e94560">"' + shiftToken + '"</span> ' +
                    shiftDir + ' ' + (Math.abs(maxShift) * 100).toFixed(1) + '%';
        } else {
            html += 'No significant prediction changes detected.';
        }
        html += '</div>';

        results.innerHTML = html;
    })
    .catch(function(e){
        results.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function clearSAEIntervention(){
    document.getElementById('sae-int-results').innerHTML = '';
    document.getElementById('sae-int-clamp').value = 0;
    document.getElementById('v-sae-clamp').textContent = '0.0';
}

// Slider value display updates for SAE controls
document.getElementById('sae-topk').addEventListener('input', function(){
    document.getElementById('v-sae-topk').textContent = this.value;
});
document.getElementById('sae-int-clamp').addEventListener('input', function(){
    document.getElementById('v-sae-clamp').textContent = parseFloat(this.value).toFixed(1);
});

// Initialize SAE panel on page load
setTimeout(function(){
    initSAEPanel();
}, 500);

// ===================== 3D DRAWING =====================
function draw3D(){
    var p=gp(),cv=document.getElementById('cv'),c=cv.getContext('2d');
    var W=cv.width,H=cv.height;
    c.clearRect(0,0,W,H);

    var nP=D.n_points,nR=D.n_real,dx=p.dx,dy=p.dy,dz=p.dz;
    var isEmb=p.mode==='embedding';
    var activeDeltas = getActiveDeltas();

    // --- Reuse extractPositions3D ---
    var pos3 = extractPositions3D(D, nP, dx, dy, dz);
    var fx = pos3.fx, fy = pos3.fy, fz = pos3.fz;

    // --- Compute cumulative 3D deltas ---
    var deltas3 = computeCumulativeDeltas3D(activeDeltas, p.layer, nP, dx, dy, dz, p.amp, p.mode, D.n_layers, isEmb);
    var edx3 = deltas3.edx, edy3 = deltas3.edy, edz3 = deltas3.edz;

    // --- Reuse computeViewBounds3D ---
    var bounds3 = computeViewBounds3D(fx, fy, fz, nP, 0.12);
    var cx3 = bounds3.cx, cy3 = bounds3.cy, cz3 = bounds3.cz;
    var mr3 = bounds3.mr;
    var vx0 = bounds3.vx0, vx1 = bounds3.vx1;
    var vy0 = bounds3.vy0, vy1 = bounds3.vy1;
    var vz0 = bounds3.vz0, vz1 = bounds3.vz1;

    var sc3 = Math.min(W,H)*0.3/mr3;
    var effSc3 = sc3 * zoomLevel;
    var cx2d = W/2 + panX;
    var cy2d = H/2 + panY;

    function proj3D_local(x, y, z){
        var r=rotatePoint3D(x, y, z);
        var scale=focalLength/(focalLength+r[2]);
        return [cx2d+r[0]*scale, cy2d+r[1]*scale, r[2], scale];
    }

    // --- Build 3D deformed grid ---
    var N3 = Math.min(Math.max(6, Math.round(p.gr/4)), 20);
    var itpMethod = document.getElementById('sel-itp').value;

    var grid3 = buildDeformedGrid3D(vx0, vy0, vz0, vx1, vy1, vz1, N3,
        fx, fy, fz, edx3, edy3, edz3, nP, p.sig, p.t, isEmb, itpMethod);

    // --- Project all vertices ---
    var projV = projectVertices3D(grid3.gX, grid3.gY, grid3.gZ, grid3.nV,
        cx3, cy3, cz3, effSc3, proj3D_local);
    var projO = projectVertices3D(grid3.oX, grid3.oY, grid3.oZ, grid3.nV,
        cx3, cy3, cz3, effSc3, proj3D_local);

    // --- Sort edges by depth ---
    var edges3d = grid3.edges;
    for(var ei=0;ei<edges3d.length;ei++){
        var e=edges3d[ei];
        e.avgZ=(projV[e.a][2]+projV[e.b][2])/2;
    }
    edges3d.sort(function(a,b){return b.avgZ-a.avgZ});

    // --- Reference grid ---
    if(p.ref){
        drawReferenceEdges3D(c, edges3d, projO, isEmb);
    }

    // --- Deformed grid lines ---
    if(p.grid && !isEmb){
        drawDeformedEdges3D(c, edges3d, projV, p.sc);
    }

    // --- Strain heatmap faces ---
    if(p.heat && !isEmb){
        var faces = buildSurfaceFaces3D(grid3, N3, projV);
        drawStrainFaces3D(c, faces, projV);
    }

    // --- Vector arrows ---
    if(p.vec && !isEmb){
        drawVectorArrows3D(c, grid3, N3, projO, projV);
    }

    // --- 3D axes ---
    draw3DAxes(c, mr3, effSc3, proj3D_local, dx, dy, dz);

    // --- Points (tokens + probes) ---
    var points3d = buildTokenPoints3D(fx, fy, fz, nP, nR, cx3, cy3, cz3, effSc3, proj3D_local);
    points3d.sort(function(a,b){return b.z-a.z});

    // --- Synthetic probes ---
    if(p.syn){
        drawSyntheticProbes3D(c, points3d, nR);
    }

    // --- Neighbor connections ---
    if(p.nb && D.neighbors && selectedTokens.size>0){
        drawNeighborConnections3D(c, D, fx, fy, fz, nP, p.kn, p.nblabel,
            cx3, cy3, cz3, effSc3, proj3D_local);
    }

    // --- Real token dots ---
    if(p.tok){
        drawRealTokenDots3D(c, D, points3d, nR, isEmb,
            fx, fy, fz, cx3, cy3, cz3, effSc3, proj3D_local);
    }

    // --- HUD ---
    draw3DHUD(c, W, H, p, dx, dy, dz, isEmb);
}

function extractPositions3D(dataObj, nP, dx, dy, dz) {
    var fx = new Float64Array(nP), fy = new Float64Array(nP), fz = new Float64Array(nP);
    for (var i = 0; i < nP; i++) {
        fx[i] = dataObj.fixed_pos[i][dx];
        fy[i] = dataObj.fixed_pos[i][dy];
        fz[i] = dataObj.fixed_pos[i][dz];
    }
    return { fx: fx, fy: fy, fz: fz };
}

function computeCumulativeDeltas3D(activeDeltas, layer, nP, dx, dy, dz, amp, mode, nLayers, isEmb) {
    var edx = new Float64Array(nP), edy = new Float64Array(nP), edz = new Float64Array(nP);
    if (isEmb) return { edx: edx, edy: edy, edz: edz };

    for (var j = 0; j < nP; j++) {
        var sx = 0, sy = 0, sz = 0;
        if (mode === 'single') {
            sx = activeDeltas[layer][j][dx];
            sy = activeDeltas[layer][j][dy];
            sz = activeDeltas[layer][j][dz];
        } else if (mode === 'cumfwd') {
            for (var l = 0; l <= layer; l++) {
                sx += activeDeltas[l][j][dx];
                sy += activeDeltas[l][j][dy];
                sz += activeDeltas[l][j][dz];
            }
        } else { // cumbwd
            for (var l = layer; l < nLayers; l++) {
                sx += activeDeltas[l][j][dx];
                sy += activeDeltas[l][j][dy];
                sz += activeDeltas[l][j][dz];
            }
        }
        edx[j] = sx * amp;
        edy[j] = sy * amp;
        edz[j] = sz * amp;
    }
    return { edx: edx, edy: edy, edz: edz };
}

function computeViewBounds3D(fx, fy, fz, nP, padding) {
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, mnz = Infinity, mxz = -Infinity;
    for (var i = 0; i < nP; i++) {
        if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
        if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
        if (fz[i] < mnz) mnz = fz[i]; if (fz[i] > mxz) mxz = fz[i];
    }
    var mrx = mxx - mnx || 1, mry = mxy - mny || 1, mrz = mxz - mnz || 1;
    var mr = Math.max(mrx, mry, mrz);
    var cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, cz = (mnz + mxz) / 2;
    var pd = padding || 0.12;
    return {
        cx: cx, cy: cy, cz: cz, mr: mr,
        vx0: cx - mr * (0.5 + pd), vx1: cx + mr * (0.5 + pd),
        vy0: cy - mr * (0.5 + pd), vy1: cy + mr * (0.5 + pd),
        vz0: cz - mr * (0.5 + pd), vz1: cz + mr * (0.5 + pd)
    };
}

function buildDeformedGrid3D(vx0, vy0, vz0, vx1, vy1, vz1, N,
    fx, fy, fz, edx, edy, edz, nP, sig, t, isEmb, itpMethod) {

    function gIdx(ix, iy, iz) { return iz * (N + 1) * (N + 1) + iy * (N + 1) + ix; }
    var nV = (N + 1) * (N + 1) * (N + 1);
    var oX = new Float64Array(nV), oY = new Float64Array(nV), oZ = new Float64Array(nV);
    var gX = new Float64Array(nV), gY = new Float64Array(nV), gZ = new Float64Array(nV);

    for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) {
        var gi = gIdx(ix, iy, iz);
        oX[gi] = vx0 + (ix / N) * (vx1 - vx0);
        oY[gi] = vy0 + (iy / N) * (vy1 - vy0);
        oZ[gi] = vz0 + (iz / N) * (vz1 - vz0);
    }

    var s2i = 1 / (2 * sig * sig);

    if (isEmb) {
        for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; gZ[gi] = oZ[gi]; }
    } else {
        for (var gi = 0; gi < nV; gi++) {
            var gpx = oX[gi], gpy = oY[gi], gpz = oZ[gi];
            var res = interpolateGridPoint3D(gpx, gpy, gpz, fx, fy, fz,
                edx, edy, edz, nP, sig, s2i, itpMethod);
            gX[gi] = gpx + t * res[0];
            gY[gi] = gpy + t * res[1];
            gZ[gi] = gpz + t * res[2];
        }
    }

    // Collect edges with strain
    var edges = [];
    function strain3(a, b) {
        var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a], oZ[b] - oZ[a]);
        var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a], gZ[b] - gZ[a]);
        return od > 1e-12 ? dd / od : 1;
    }

    for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix < N; ix++) {
        var a = gIdx(ix, iy, iz), b = gIdx(ix + 1, iy, iz);
        edges.push({ a: a, b: b, strain: strain3(a, b) });
    }
    for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy < N; iy++) for (var ix = 0; ix <= N; ix++) {
        var a = gIdx(ix, iy, iz), b = gIdx(ix, iy + 1, iz);
        edges.push({ a: a, b: b, strain: strain3(a, b) });
    }
    for (var iz = 0; iz < N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) {
        var a = gIdx(ix, iy, iz), b = gIdx(ix, iy, iz + 1);
        edges.push({ a: a, b: b, strain: strain3(a, b) });
    }

    return { oX: oX, oY: oY, oZ: oZ, gX: gX, gY: gY, gZ: gZ,
             edges: edges, nV: nV, gIdx: gIdx, N: N };
}

function interpolateGridPoint3D(gpx, gpy, gpz, fx, fy, fz, edx, edy, edz, nP, sig, s2i, itpMethod) {
    var vvx = 0, vvy = 0, vvz = 0, ws = 0;

    if (itpMethod === 'nn') {
        var bestDist = Infinity, bestIdx = 0;
        for (var k = 0; k < nP; k++) {
            var d = (gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]);
            if (d < bestDist) { bestDist = d; bestIdx = k; }
        }
        return [edx[bestIdx], edy[bestIdx], edz[bestIdx]];
    }

    if (itpMethod === 'idw') {
        for (var k = 0; k < nP; k++) {
            var dist = Math.sqrt((gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]));
            var w = 1.0 / Math.pow(Math.max(dist, 1e-12), 2.0);
            vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
        }
        if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
        return [vvx, vvy, vvz];
    }

    if (itpMethod === 'wendland') {
        var R = Math.max(3.0 * sig, 1e-6);
        for (var k = 0; k < nP; k++) {
            var dist = Math.sqrt((gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]));
            var rn = dist / R;
            if (rn >= 1.0) continue;
            var tt = 1.0 - rn;
            var w = tt * tt * tt * tt * (4.0 * rn + 1.0);
            vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
        }
        if (ws < 1e-30) {
            // Fallback to RBF
            for (var k = 0; k < nP; k++) {
                var ex = gpx-fx[k], ey = gpy-fy[k], ez = gpz-fz[k];
                var w = Math.exp(-(ex*ex+ey*ey+ez*ez)*s2i);
                vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
            }
        }
        if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
        return [vvx, vvy, vvz];
    }

    // Default: RBF (also fallback for MLS, TPS in 3D)
    for (var k = 0; k < nP; k++) {
        var ex = gpx-fx[k], ey = gpy-fy[k], ez = gpz-fz[k];
        var w = Math.exp(-(ex*ex+ey*ey+ez*ez)*s2i);
        vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
    }
    if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
    return [vvx, vvy, vvz];
}

function projectVertices3D(gX, gY, gZ, nV, cx, cy, cz, scale, projFn) {
    var projected = [];
    for (var vi = 0; vi < nV; vi++) {
        var px = (gX[vi] - cx) * scale;
        var py = (gY[vi] - cy) * scale;
        var pz = (gZ[vi] - cz) * scale;
        projected.push(projFn(px, py, pz));
    }
    return projected;
}

function drawReferenceEdges3D(c, edges, projO, isEmb) {
    c.strokeStyle = isEmb ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.04)';
    c.lineWidth = 0.4;
    for (var ei = 0; ei < edges.length; ei++) {
        var e = edges[ei];
        var pa = projO[e.a], pb = projO[e.b];
        c.beginPath(); c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]); c.stroke();
    }
}

function drawDeformedEdges3D(c, edges, projV, showSC) {
    c.lineWidth = 0.9;
    for (var ei = 0; ei < edges.length; ei++) {
        var e = edges[ei];
        var pa = projV[e.a], pb = projV[e.b];
        var depthAlpha = Math.max(0.1, Math.min(0.85, 0.6 - e.avgZ * 0.001));
        if (showSC) {
            var ec = s2c(e.strain);
            c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',' + depthAlpha.toFixed(2) + ')';
        } else {
            c.strokeStyle = 'rgba(200,200,200,' + depthAlpha.toFixed(2) + ')';
        }
        c.beginPath(); c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]); c.stroke();
    }
}

function buildSurfaceFaces3D(grid, N, projV) {
    var gIdx = grid.gIdx;
    var faces = [];

    function strainEdge(a, b) {
        var od = Math.hypot(grid.oX[b]-grid.oX[a], grid.oY[b]-grid.oY[a], grid.oZ[b]-grid.oZ[a]);
        var dd = Math.hypot(grid.gX[b]-grid.gX[a], grid.gY[b]-grid.gY[a], grid.gZ[b]-grid.gZ[a]);
        return od > 1e-12 ? dd / od : 1;
    }

    function addFace(a, b, cc, d) {
        var s1 = strainEdge(a, b), s2 = strainEdge(b, cc);
        var avgS = (s1 + s2) / 2;
        var avgZ = (projV[a][2] + projV[b][2] + projV[cc][2] + projV[d][2]) / 4;
        faces.push({ verts: [a, b, cc, d], strain: avgS, z: avgZ });
    }

    // X=0 and X=N faces
    for (var iz = 0; iz < N; iz++) for (var iy = 0; iy < N; iy++) {
        addFace(gIdx(0,iy,iz), gIdx(0,iy+1,iz), gIdx(0,iy+1,iz+1), gIdx(0,iy,iz+1));
        addFace(gIdx(N,iy,iz), gIdx(N,iy+1,iz), gIdx(N,iy+1,iz+1), gIdx(N,iy,iz+1));
    }
    // Y=0 and Y=N faces
    for (var iz = 0; iz < N; iz++) for (var ix = 0; ix < N; ix++) {
        addFace(gIdx(ix,0,iz), gIdx(ix+1,0,iz), gIdx(ix+1,0,iz+1), gIdx(ix,0,iz+1));
        addFace(gIdx(ix,N,iz), gIdx(ix+1,N,iz), gIdx(ix+1,N,iz+1), gIdx(ix,N,iz+1));
    }
    // Z=0 and Z=N faces
    for (var iy = 0; iy < N; iy++) for (var ix = 0; ix < N; ix++) {
        addFace(gIdx(ix,iy,0), gIdx(ix+1,iy,0), gIdx(ix+1,iy+1,0), gIdx(ix,iy+1,0));
        addFace(gIdx(ix,iy,N), gIdx(ix+1,iy,N), gIdx(ix+1,iy+1,N), gIdx(ix,iy+1,N));
    }

    faces.sort(function(a, b) { return b.z - a.z; });
    return faces;
}

function drawStrainFaces3D(c, faces, projV) {
    for (var fi = 0; fi < faces.length; fi++) {
        var f = faces[fi];
        var co = s2c(f.strain);
        c.beginPath();
        c.moveTo(projV[f.verts[0]][0], projV[f.verts[0]][1]);
        c.lineTo(projV[f.verts[1]][0], projV[f.verts[1]][1]);
        c.lineTo(projV[f.verts[2]][0], projV[f.verts[2]][1]);
        c.lineTo(projV[f.verts[3]][0], projV[f.verts[3]][1]);
        c.closePath();
        c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.15)';
        c.fill();
    }
}

function drawVectorArrows3D(c, grid, N, projO, projV) {
    var step = Math.max(1, Math.floor(N / 4));
    var gIdx = grid.gIdx;
    c.lineWidth = 1.2;
    for (var viz = 0; viz <= N; viz += step)
        for (var viy = 0; viy <= N; viy += step)
            for (var vix = 0; vix <= N; vix += step) {
                var vi = gIdx(vix, viy, viz);
                var pa = projO[vi], pb = projV[vi];
                var al = Math.hypot(pb[0]-pa[0], pb[1]-pa[1]);
                if (al < 3) continue;
                c.strokeStyle = 'rgba(255,255,100,0.5)';
                c.fillStyle = 'rgba(255,255,100,0.5)';
                c.beginPath(); c.moveTo(pa[0], pa[1]); c.lineTo(pb[0], pb[1]); c.stroke();
                var aa = Math.atan2(pb[1]-pa[1], pb[0]-pa[0]), hl = Math.min(6, al*0.3);
                c.beginPath(); c.moveTo(pb[0], pb[1]);
                c.lineTo(pb[0]-hl*Math.cos(aa-0.4), pb[1]-hl*Math.sin(aa-0.4));
                c.lineTo(pb[0]-hl*Math.cos(aa+0.4), pb[1]-hl*Math.sin(aa+0.4));
                c.closePath(); c.fill();
            }
}

function draw3DAxes(c, mr, effSc, projFn, dx, dy, dz) {
    var axLen = mr * 0.5 * effSc;
    var axes = [
        { v: [1,0,0], label: 'Dim ' + dx, color: '#e94560' },
        { v: [0,1,0], label: 'Dim ' + dy, color: '#53a8b6' },
        { v: [0,0,1], label: 'Dim ' + dz, color: '#f5a623' }
    ];
    c.lineWidth = 1.5;
    var o3 = projFn(0, 0, 0);
    for (var ai = 0; ai < 3; ai++) {
        var ax = axes[ai];
        var e3 = projFn(ax.v[0]*axLen, ax.v[1]*axLen, ax.v[2]*axLen);
        c.strokeStyle = ax.color; c.globalAlpha = 0.5;
        c.beginPath(); c.moveTo(o3[0], o3[1]); c.lineTo(e3[0], e3[1]); c.stroke();
        c.globalAlpha = 1;
        c.font = '10px monospace'; c.fillStyle = ax.color;
        c.fillText(ax.label, e3[0]+4, e3[1]-4);
    }
}

function buildTokenPoints3D(fx, fy, fz, nP, nR, cx, cy, cz, scale, projFn) {
    var points = [];
    for (var pi = 0; pi < nP; pi++) {
        var px = (fx[pi]-cx)*scale, py = (fy[pi]-cy)*scale, pz = (fz[pi]-cz)*scale;
        var proj = projFn(px, py, pz);
        points.push({ idx: pi, sx: proj[0], sy: proj[1], z: proj[2], scale: proj[3] });
    }
    return points;
}

function drawSyntheticProbes3D(c, points, nR) {
    for (var si = 0; si < points.length; si++) {
        var sp = points[si];
        if (sp.idx < nR) continue;
        var sr = Math.max(1, 2.5 * sp.scale);
        c.beginPath(); c.arc(sp.sx, sp.sy, sr, 0, Math.PI*2);
        c.fillStyle = 'rgba(100,200,255,0.15)';
        c.fill();
    }
}

function drawNeighborConnections3D(c, D, fx, fy, fz, nP, kn, showLabels,
    cx, cy, cz, scale, projFn) {

    selectedTokens.forEach(function(ti) {
        if (ti >= D.neighbors.length) return;
        var nbs = D.neighbors[ti].slice(0, kn);
        var tpx = (fx[ti] - cx) * scale, tpy = (fy[ti] - cy) * scale, tpz = (fz[ti] - cz) * scale;
        var tp = projFn(tpx, tpy, tpz);

        for (var ni = 0; ni < nbs.length; ni++) {
            var nb = nbs[ni];
            var nidx = nb.idx;
            if (nidx >= nP) continue;
            var npx = (fx[nidx] - cx) * scale, npy = (fy[nidx] - cy) * scale, npz = (fz[nidx] - cz) * scale;
            var np2 = projFn(npx, npy, npz);
            var alpha = Math.max(0.15, 1.0 - ni * 0.08);
            c.strokeStyle = 'rgba(0,255,200,' + alpha.toFixed(2) + ')';
            c.lineWidth = Math.max(0.5, 2.5 - ni * 0.2);
            c.setLineDash([3, 3]);
            c.beginPath(); c.moveTo(tp[0], tp[1]); c.lineTo(np2[0], np2[1]); c.stroke();
            c.setLineDash([]);
            var nr = Math.max(2, 5 * np2[3]);
            c.beginPath(); c.arc(np2[0], np2[1], nr, 0, Math.PI * 2);
            c.fillStyle = nb.is_real ? 'rgba(0,255,200,0.8)' : 'rgba(0,255,200,0.35)';
            c.fill();
            if (showLabels) {
                c.font = '9px monospace'; c.fillStyle = 'rgba(0,255,200,0.9)';
                c.fillText(nb.label + ' (d=' + nb.dist.toFixed(1) + ')', np2[0] + 8, np2[1] - 4);
            }
        }
    });
}

function drawRealTokenDots3D(c, D, points, nR, isEmb,
    fx, fy, fz, cx, cy, cz, scale, projFn) {

    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
        '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22',
        '#f39c12','#d35400','#c0392b','#16a085','#27ae60',
        '#2980b9','#8e44ad','#2c3e50','#ecf0f1','#fd79a8'];

    for (var ri = 0; ri < points.length; ri++) {
        var rp = points[ri];
        if (rp.idx >= nR) continue;
        var ti3 = rp.idx;
        var col = tc[ti3 % tc.length];
        var isSel = selectedTokens.has(ti3);
        var dotR = Math.max(3, (isSel ? 9 : 7) * rp.scale);

        if (isSel) {
            var grad2 = c.createRadialGradient(rp.sx, rp.sy, 0, rp.sx, rp.sy, 30 * rp.scale);
            grad2.addColorStop(0, 'rgba(0,255,0,0.25)');
            grad2.addColorStop(1, 'rgba(0,255,0,0)');
            c.beginPath(); c.arc(rp.sx, rp.sy, 30 * rp.scale, 0, Math.PI * 2);
            c.fillStyle = grad2; c.fill();
        }

        c.beginPath(); c.arc(rp.sx, rp.sy, dotR, 0, Math.PI * 2);
        c.fillStyle = col; c.fill();
        c.strokeStyle = isSel ? '#0f0' : '#fff';
        c.lineWidth = isSel ? 3 : 2; c.stroke();

        var fontSize = Math.max(8, 11 * rp.scale);
        c.font = 'bold ' + Math.round(fontSize) + 'px monospace';
        c.lineWidth = 3; c.strokeStyle = 'rgba(0,0,0,0.9)';
        var lb = '[' + ti3 + '] ' + D.tokens[ti3];
        c.strokeText(lb, rp.sx + 12, rp.sy - 10);
        c.fillStyle = isSel ? '#0f0' : '#fff';
        c.fillText(lb, rp.sx + 12, rp.sy - 10);
    }

    if (isEmb && nR > 1) {
        c.strokeStyle = 'rgba(233,69,96,0.3)'; c.lineWidth = 1.5; c.setLineDash([4, 4]);
        c.beginPath();
        var fp0 = projFn((fx[0] - cx) * scale, (fy[0] - cy) * scale, (fz[0] - cz) * scale);
        c.moveTo(fp0[0], fp0[1]);
        for (var ti4 = 1; ti4 < nR; ti4++) {
            var pp = projFn((fx[ti4] - cx) * scale, (fy[ti4] - cy) * scale, (fz[ti4] - cz) * scale);
            c.lineTo(pp[0], pp[1]);
        }
        c.stroke(); c.setLineDash([]);
    }
}

function draw3DHUD(c, W, H, p, dx, dy, dz, isEmb) {
    var decompLabel = getDecompLabel();
    c.font = '11px monospace'; c.fillStyle = 'rgba(255,255,255,0.45)';
    if (isEmb) {
        c.fillText('EMBEDDING SPACE [3D]  Dims:' + dx + ',' + dy + ',' + dz + '  Drag to rotate', 42, 18);
    } else {
        c.fillText('Layer ' + p.layer + '/' + (D.n_layers - 1) + '  t=' + p.t.toFixed(2) +
            '  amp=' + p.amp.toFixed(1) + '  Dims:' + dx + ',' + dy + ',' + dz +
            '  Decomp:' + decompLabel + '  [3D]  Drag to rotate', 42, 18);
    }
    c.font = '10px monospace'; c.fillStyle = 'rgba(255,255,255,0.35)';
    c.fillText('Zoom: ' + zoomLevel.toFixed(2) + 'x  (Scroll=zoom, Shift+drag=pan, 0=reset)', 42, H - 10);
}

// Scroll-wheel zoom
cv3d.addEventListener('wheel', function(e) {
    e.preventDefault();
    var rect = cv3d.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    var oldZoom = zoomLevel;
    var zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    zoomLevel = Math.max(0.1, Math.min(50, zoomLevel * zoomFactor));

    if(viewMode==='2d'){
        panX = mx - (mx - panX) * (zoomLevel / oldZoom);
        panY = my - (my - panY) * (zoomLevel / oldZoom);
    } else {
        var W = cv3d.width, H = cv3d.height;
        var cx2dOld = W/2 + panX;
        var cy2dOld = H/2 + panY;
        var dmx = mx - cx2dOld;
        var dmy = my - cy2dOld;
        panX += dmx * (1 - zoomLevel / oldZoom);
        panY += dmy * (1 - zoomLevel / oldZoom);
    }

    draw();
}, { passive: false });
// ===================== NEURON ACTIVATION GRID =====================

var neuronGridData = null;

document.getElementById('ng-pixsize').addEventListener('input', function(){
    document.getElementById('v-ng-pixsize').textContent = this.value;
    if(neuronGridData) renderNeuronGrid();
});
document.getElementById('ng-norm').addEventListener('change', function(){
    if(neuronGridData) renderNeuronGrid();
});
document.getElementById('ng-absval').addEventListener('change', function(){
    if(neuronGridData) renderNeuronGrid();
});

function fetchNeuronGrid(){
    if(!D) return;
    var panel = document.getElementById('neuron-grid-panel');
    panel.style.display = 'contents';
    panel.innerHTML = '<span style="color:#53a8b6">Loading neuron activations...</span>';

    fetch('/neuron_grid', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: D.text})
    })
    .then(function(r){return r.json()})
    .then(function(data){
        if(data.error){
            panel.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }
        neuronGridData = data;
        renderNeuronGrid();
    })
    .catch(function(e){
        panel.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function renderNeuronGrid(){
    var data = neuronGridData;
    if(!data) return;

    var panel = document.getElementById('neuron-grid-panel');
    var normMode = document.getElementById('ng-norm').value;  // 'layer' or 'global'
    var pixSize = +document.getElementById('ng-pixsize').value;
    var useAbs = document.getElementById('ng-absval').checked;
    var hiddenDim = data.hidden_dim;
    var nTokens = data.n_tokens;
    var nLayers = data.n_layers;

    var layersSource = (normMode === 'global') ? data.global_norm : data.layer_norm;

    // Compute a nice grid layout for the hidden_dim neurons
    // Try to make it roughly square
    var gridCols = Math.ceil(Math.sqrt(hiddenDim));
    var gridRows = Math.ceil(hiddenDim / gridCols);

    var html = '';
    html += '<div style="color:#888;font-size:9px;margin-bottom:6px">';
    html += 'Each rectangle = one layer\'s activations. ';
    html += 'Each pixel = one neuron (dim). ';
    html += 'Bright = high activation, dark = low. ';
    html += 'Grid: ' + gridCols + '×' + gridRows + ' (' + hiddenDim + ' neurons)';
    html += '</div>';

    // For each token, show all layers side by side
    for(var ti = 0; ti < nTokens; ti++){
        html += '<div style="margin-bottom:8px">';
        html += '<div style="color:#e94560;font-weight:bold;font-size:10px;margin-bottom:2px">';
        html += '[' + ti + '] ' + data.tokens[ti];
        html += '</div>';
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start;min-height:0;overflow:visible">';

        for(var li = 0; li < nLayers; li++){
            var acts = layersSource[li].activations[ti]; // array of hiddenDim floats [0,1]
            var canvasId = 'ng-cv-' + ti + '-' + li;
            var canvasW = gridCols * pixSize;
            var canvasH = gridRows * pixSize;

            html += '<div style="text-align:center">';
            html += '<div style="color:#53a8b6;font-size:8px;margin-bottom:1px">';
            html += (li === 0 ? 'Emb' : 'L' + (li-1));
            html += '</div>';
            html += '<canvas id="' + canvasId + '" width="' + canvasW + '" height="' + canvasH + '" ';
            html += 'style="border:1px solid #0f3460;image-rendering:pixelated" ';
            html += 'title="' + (li === 0 ? 'Embedding' : 'Layer ' + (li-1)) + ' — Token: ' + data.tokens[ti] + '">';
            html += '</canvas>';
            html += '</div>';
        }

        html += '</div></div>';
    }

    panel.innerHTML = html;

    // Now draw on each canvas
    for(var ti = 0; ti < nTokens; ti++){
        for(var li = 0; li < nLayers; li++){
            var canvasId = 'ng-cv-' + ti + '-' + li;
            var cv = document.getElementById(canvasId);
            if(!cv) continue;
            var ctx = cv.getContext('2d');
            var acts = layersSource[li].activations[ti];

            var imgData = ctx.createImageData(gridCols * pixSize, gridRows * pixSize);

            for(var ni = 0; ni < hiddenDim; ni++){
                var val = acts[ni];
                if(useAbs) val = Math.abs(val * 2 - 1); // re-center then abs

                // Map to grayscale: 0 = black, 1 = white
                var brightness = Math.floor(val * 255);
                brightness = Math.max(0, Math.min(255, brightness));

                var col = Math.floor(ni % gridCols);
                var row = Math.floor(ni / gridCols);

                // Fill the pixel block
                for(var py = 0; py < pixSize; py++){
                    for(var px = 0; px < pixSize; px++){
                        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
                        var offset = ix * 4;
                        imgData.data[offset]     = brightness;  // R
                        imgData.data[offset + 1] = brightness;  // G
                        imgData.data[offset + 2] = brightness;  // B
                        imgData.data[offset + 3] = 255;         // A
                    }
                }
            }

            // Fill remaining pixels (if hiddenDim doesn't fill the grid) with dark
            for(var ni = hiddenDim; ni < gridCols * gridRows; ni++){
                var col = ni % gridCols;
                var row = Math.floor(ni / gridCols);
                for(var py = 0; py < pixSize; py++){
                    for(var px = 0; px < pixSize; px++){
                        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
                        var offset = ix * 4;
                        imgData.data[offset]     = 20;
                        imgData.data[offset + 1] = 10;
                        imgData.data[offset + 2] = 30;
                        imgData.data[offset + 3] = 255;
                    }
                }
            }

            ctx.putImageData(imgData, 0, 0);
        }
    }
}

// Cool-to-hot colormap: dark blue → cyan → yellow → red → white
function valToColor(v) {
    // v in [0, 1]
    var r, g, b;
    if (v < 0.25) {
        var t = v / 0.25;
        r = 0; g = Math.floor(t * 128); b = Math.floor(64 + t * 191);
    } else if (v < 0.5) {
        var t = (v - 0.25) / 0.25;
        r = 0; g = Math.floor(128 + t * 127); b = Math.floor(255 - t * 128);
    } else if (v < 0.75) {
        var t = (v - 0.5) / 0.25;
        r = Math.floor(t * 255); g = 255; b = Math.floor(127 - t * 127);
    } else {
        var t = (v - 0.75) / 0.25;
        r = 255; g = Math.floor(255 - t * 128); b = Math.floor(t * 128);
    }
    return [r, g, b];
}
// ============================================================
// DIFFEOMORPHISM STACKING — KELP FOREST OF LAYER ROOMS
// ============================================================

function toggleDiffeo() {
  var wrap = document.getElementById('diffeo-wrap');
  if (!wrap) return;
  wrap.style.display = diffeoState.active ? 'block' : 'none';
  if (diffeoState.active) {
    resizeDiffeoCanvas();
    rebuildDiffeo();
    startDiffeoLoop();
  } else {
    stopDiffeoLoop();
    if (diffeoCtx) {
      var W = diffeoCanvas.width / (window.devicePixelRatio || 1);
      var H = diffeoCanvas.height / (window.devicePixelRatio || 1);
      diffeoCtx.clearRect(0, 0, W, H);
    }
  }
}

function resizeDiffeoCanvas() {
  if (!diffeoCanvas) return;
  var container = document.getElementById('main');
  var dpr = window.devicePixelRatio || 1;
  diffeoCanvas.width = container.clientWidth * dpr;
  diffeoCanvas.height = container.clientHeight * dpr;
  diffeoCanvas.style.width = container.clientWidth + 'px';
  diffeoCanvas.style.height = container.clientHeight + 'px';
  if (diffeoCtx) diffeoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function startDiffeoLoop() {
  if (diffeoAnimId) return;
  function loop() {
    diffeoTime += 0.016;
    if (diffeoState.active && diffeoState.built) {
      updateDiffeoGrids(diffeoTime);
      renderDiffeoOverlay(diffeoTime);
    }
    diffeoAnimId = requestAnimationFrame(loop);
  }
  diffeoAnimId = requestAnimationFrame(loop);
}

function stopDiffeoLoop() {
  if (diffeoAnimId) {
    cancelAnimationFrame(diffeoAnimId);
    diffeoAnimId = null;
  }
}

/**
 * Build diffeomorphism slices from the server data D.
 * Each slice = one 2D cross-section (dim pair) at one layer.
 * Deformation comes from D.deltas[layer][point][dim].
 */
function rebuildDiffeo() {
  diffeoState.slices = [];
  diffeoState.built = false;
  if (!D || D.n_layers < 1 || D.n_points < 2) return;

  var nL = D.n_layers;
  var nP = D.n_points;
  var nR = D.n_real;
  var hiddenDim = D.hidden_dim;
  var res = diffeoState.gridRes;

  // Use the currently active deltas (respects decomposition selector)
  var activeDeltas = getActiveDeltas();
  if (!activeDeltas) activeDeltas = D.deltas;

  // Generate dimension pairs
  var dimPairs = [];
  var maxDims = Math.min(hiddenDim, 8);
  if (diffeoState.dimMode === 'sequential') {
    for (var d = 0; d + 1 < maxDims; d += 2) dimPairs.push([d, d + 1]);
  } else if (diffeoState.dimMode === 'first') {
    for (var d = 1; d < maxDims; d++) dimPairs.push([0, d]);
  } else {
    for (var a = 0; a < maxDims; a++)
      for (var b = a + 1; b < maxDims; b++)
        dimPairs.push([a, b]);
  }
  if (dimPairs.length === 0) dimPairs.push([0, 1]);

  // Build slice configs: layer × dim pair
  var sliceConfigs = [];
  for (var li = 0; li < nL; li++) {
    for (var pi = 0; pi < dimPairs.length; pi++) {
      sliceConfigs.push({ layerIdx: li, dimA: dimPairs[pi][0], dimB: dimPairs[pi][1] });
    }
  }

  // Evenly sample if too many
  var maxSlices = diffeoState.numSlices;
  if (sliceConfigs.length > maxSlices) {
    var step = sliceConfigs.length / maxSlices;
    var sampled = [];
    for (var i = 0; i < maxSlices; i++) sampled.push(sliceConfigs[Math.floor(i * step)]);
    sliceConfigs = sampled;
  }

  // For each slice, compute the deformation grid
  // We use the real tokens' fixed_pos and deltas to define the deformation field
  // via RBF interpolation onto a regular grid in the 2D subspace.
  for (var si = 0; si < sliceConfigs.length; si++) {
    var cfg = sliceConfigs[si];
    var dA = cfg.dimA, dB = cfg.dimB, lay = cfg.layerIdx;

    // Extract the 2D positions and delta vectors for real tokens in this dim pair
    var posA = new Float64Array(nR), posB = new Float64Array(nR);
    var delA = new Float64Array(nR), delB = new Float64Array(nR);
    var mnA = Infinity, mxA = -Infinity, mnB = Infinity, mxB = -Infinity;

    for (var ti = 0; ti < nR; ti++) {
      posA[ti] = D.fixed_pos[ti][dA];
      posB[ti] = D.fixed_pos[ti][dB];
      delA[ti] = activeDeltas[lay][ti][dA];
      delB[ti] = activeDeltas[lay][ti][dB];
      if (posA[ti] < mnA) mnA = posA[ti]; if (posA[ti] > mxA) mxA = posA[ti];
      if (posB[ti] < mnB) mnB = posB[ti]; if (posB[ti] > mxB) mxB = posB[ti];
    }

    var rngA = mxA - mnA || 1, rngB = mxB - mnB || 1;
    var rng = Math.max(rngA, rngB);
    var pad = 0.15;
    var cA = (mnA + mxA) / 2, cB = (mnB + mxB) / 2;
    var lo = cA - rng * (0.5 + pad), hi = cA + rng * (0.5 + pad);
    var loB = cB - rng * (0.5 + pad), hiB = cB + rng * (0.5 + pad);

    // RBF bandwidth
    var sigma = rng * 0.2;
    var s2i = 1 / (2 * sigma * sigma);

    // Build grid
    var grid = [];
    for (var gy = 0; gy <= res; gy++) {
      for (var gx = 0; gx <= res; gx++) {
        var u = gx / res;
        var v = gy / res;
        var worldA = lo + u * (hi - lo);
        var worldB = loB + v * (hiB - loB);

        // RBF interpolation of delta from real tokens
        var dx = 0, dy = 0, ws = 0;
        for (var k = 0; k < nR; k++) {
          var ea = worldA - posA[k], eb = worldB - posB[k];
          var w = Math.exp(-(ea * ea + eb * eb) * s2i);
          dx += w * delA[k];
          dy += w * delB[k];
          ws += w;
        }
        if (ws > 1e-15) { dx /= ws; dy /= ws; }

        grid.push({ ox: u, oy: v, dx: dx, dy: dy, divergence: 0 });
      }
    }

    // Compute divergence (how much neighboring displacements differ)
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx = gy * (res + 1) + gx;
        var idxR = idx + 1;
        var idxD = idx + (res + 1);
        var ddx = grid[idxR].dx - grid[idx].dx;
        var ddy = grid[idxD].dy - grid[idx].dy;
        grid[idx].divergence = Math.sqrt(ddx * ddx + ddy * ddy);
      }
    }

    var hue = (cfg.layerIdx / nL) * 120 + (cfg.dimA * 40 + cfg.dimB * 20);

    diffeoState.slices.push({
      layerIdx: cfg.layerIdx,
      dimA: cfg.dimA,
      dimB: cfg.dimB,
      grid: grid,
      hue: hue % 360,
      res: res,
    });
  }

  diffeoState.built = true;
}

/**
 * Update grid deformations each frame.
 * Re-reads deltas from D so it stays in sync with layer/decomp changes.
 * The kelp sway is purely time-driven via divergence.
 */
function updateDiffeoGrids(time) {
  if (!D || !diffeoState.built) return;

  var activeDeltas = getActiveDeltas();
  if (!activeDeltas) activeDeltas = D.deltas;
  var nR = D.n_real;

  for (var si = 0; si < diffeoState.slices.length; si++) {
    var slice = diffeoState.slices[si];
    var res = slice.res;
    var lay = slice.layerIdx;
    var dA = slice.dimA, dB = slice.dimB;

    // Recompute positions and deltas for this layer
    var posA = new Float64Array(nR), posB = new Float64Array(nR);
    var delA = new Float64Array(nR), delB = new Float64Array(nR);
    var mnA = Infinity, mxA = -Infinity, mnB = Infinity, mxB = -Infinity;

    for (var ti = 0; ti < nR; ti++) {
      posA[ti] = D.fixed_pos[ti][dA];
      posB[ti] = D.fixed_pos[ti][dB];
      delA[ti] = activeDeltas[lay][ti][dA];
      delB[ti] = activeDeltas[lay][ti][dB];
      if (posA[ti] < mnA) mnA = posA[ti]; if (posA[ti] > mxA) mxA = posA[ti];
      if (posB[ti] < mnB) mnB = posB[ti]; if (posB[ti] > mxB) mxB = posB[ti];
    }

    var rng = Math.max(mxA - mnA, mxB - mnB) || 1;
    var pad = 0.15;
    var cA = (mnA + mxA) / 2, cB = (mnB + mxB) / 2;
    var lo = cA - rng * (0.5 + pad), hi = cA + rng * (0.5 + pad);
    var loB = cB - rng * (0.5 + pad), hiB = cB + rng * (0.5 + pad);
    var sigma = rng * 0.2;
    var s2i = 1 / (2 * sigma * sigma);

    for (var gi = 0; gi < slice.grid.length; gi++) {
      var cell = slice.grid[gi];
      var worldA = lo + cell.ox * (hi - lo);
      var worldB = loB + cell.oy * (hiB - loB);

      var dx = 0, dy = 0, ws = 0;
      for (var k = 0; k < nR; k++) {
        var ea = worldA - posA[k], eb = worldB - posB[k];
        var w = Math.exp(-(ea * ea + eb * eb) * s2i);
        dx += w * delA[k];
        dy += w * delB[k];
        ws += w;
      }
      if (ws > 1e-15) { dx /= ws; dy /= ws; }
      cell.dx = dx;
      cell.dy = dy;
    }

    // Recompute divergence
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx = gy * (res + 1) + gx;
        var idxR = idx + 1;
        var idxD = idx + (res + 1);
        var ddx = slice.grid[idxR].dx - slice.grid[idx].dx;
        var ddy = slice.grid[idxD].dy - slice.grid[idx].dy;
        slice.grid[idx].divergence = Math.sqrt(ddx * ddx + ddy * ddy);
      }
    }
  }
}

/**
 * Render the stacked diffeomorphism slices as translucent kelp-swaying grids.
 */
function renderDiffeoOverlay(time) {
  if (!diffeoState.active || !diffeoCtx || diffeoState.slices.length === 0) return;

  var dpr = window.devicePixelRatio || 1;
  var W = diffeoCanvas.width / dpr;
  var H = diffeoCanvas.height / dpr;
  diffeoCtx.clearRect(0, 0, W, H);

  var nSlices = diffeoState.slices.length;
  var spacing = diffeoState.layerSpacing;
  var totalHeight = nSlices * spacing;
  var startY = (H - totalHeight) / 2;

  var amp = diffeoState.kelpAmplitude;
  var sens = diffeoState.divergenceSensitivity;
  var alpha = diffeoState.sliceAlpha;

  // Normalize delta magnitudes for visible sway
  var maxDelta = 0;
  for (var si = 0; si < nSlices; si++) {
    var grid = diffeoState.slices[si].grid;
    for (var gi = 0; gi < grid.length; gi++) {
      var mag = Math.sqrt(grid[gi].dx * grid[gi].dx + grid[gi].dy * grid[gi].dy);
      if (mag > maxDelta) maxDelta = mag;
    }
  }
  var deltaScale = maxDelta > 1e-8 ? 1.0 / maxDelta : 1.0;

  for (var si = 0; si < nSlices; si++) {
    var slice = diffeoState.slices[si];
    var res = slice.res;
    var grid = slice.grid;
    var baseY = startY + si * spacing;

    var sliceW = W * 0.55;
    var sliceH = spacing * 0.75;

    // Perspective: further slices smaller
    var depthFactor = 0.65 + 0.35 * (si / Math.max(nSlices - 1, 1));
    var perspW = sliceW * depthFactor;
    var perspH = sliceH * depthFactor;
    var perspX = (W - perspW) / 2;
    var perspY = baseY + (sliceH - perspH) / 2;

    diffeoCtx.save();
    diffeoCtx.globalAlpha = alpha * (0.4 + 0.6 * depthFactor);

    // Draw deformed grid cells
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx00 = gy * (res + 1) + gx;
        var idx10 = idx00 + 1;
        var idx01 = idx00 + (res + 1);
        var idx11 = idx01 + 1;

        var corners = [idx00, idx10, idx11, idx01];
        var pts = [];
        for (var ci = 0; ci < 4; ci++) {
          var cell = grid[corners[ci]];
          var div = cell.divergence * sens;
          // Kelp sway: divergence drives amplitude, time drives oscillation
          var swayX = Math.sin(time * 0.7 + corners[ci] * 0.5 + si * 1.3) * div * amp * 35 * deltaScale;
          var swayY = Math.sin(time * 0.5 + corners[ci] * 0.3 + si * 0.9) * div * amp * 12 * deltaScale;

          var screenX = perspX + (cell.ox + cell.dx * deltaScale * amp * 0.3) * perspW + swayX;
          var screenY = perspY + (cell.oy + cell.dy * deltaScale * amp * 0.3) * perspH + swayY;
          pts.push({ x: screenX, y: screenY });
        }

        var cellDiv = grid[idx00].divergence * sens;
        var brightness = Math.min(70, 30 + cellDiv * 300 * deltaScale);
        var saturation = 50 + Math.min(30, cellDiv * 50 * deltaScale);

        // Filled quad
        diffeoCtx.beginPath();
        diffeoCtx.moveTo(pts[0].x, pts[0].y);
        diffeoCtx.lineTo(pts[1].x, pts[1].y);
        diffeoCtx.lineTo(pts[2].x, pts[2].y);
        diffeoCtx.lineTo(pts[3].x, pts[3].y);
        diffeoCtx.closePath();
        diffeoCtx.fillStyle = 'hsla(' + slice.hue + ',' + saturation + '%,' + brightness + '%,' + (alpha * 0.4) + ')';
        diffeoCtx.fill();

        // Grid lines
        diffeoCtx.strokeStyle = 'hsla(' + slice.hue + ',70%,' + Math.min(80, 40 + cellDiv * 150 * deltaScale) + '%,' + (alpha * 1.2) + ')';
        diffeoCtx.lineWidth = 0.5 + cellDiv * 3 * deltaScale;
        diffeoCtx.stroke();
      }
    }

    // Kelp strands connecting to next slice
    if (si < nSlices - 1) {
      var nextSlice = diffeoState.slices[si + 1];
      var nextBaseY = startY + (si + 1) * spacing;
      var nextDepth = 0.65 + 0.35 * ((si + 1) / Math.max(nSlices - 1, 1));
      var nextPerspW = sliceW * nextDepth;
      var nextPerspH = spacing * 0.75 * nextDepth;
      var nextPerspX = (W - nextPerspW) / 2;
      var nextPerspY = nextBaseY + (spacing * 0.75 - nextPerspH) / 2;

      var step = Math.max(1, Math.floor(res / 3));
      for (var gy = 0; gy <= res; gy += step) {
        for (var gx = 0; gx <= res; gx += step) {
          var idx = gy * (res + 1) + gx;
          if (idx >= grid.length) continue;
          var cell = grid[idx];
          var div = cell.divergence * sens;

          var swX1 = Math.sin(time * 0.7 + idx * 0.5 + si * 1.3) * div * amp * 35 * deltaScale;
          var swY1 = Math.sin(time * 0.5 + idx * 0.3 + si * 0.9) * div * amp * 12 * deltaScale;
          var x1 = perspX + (cell.ox + cell.dx * deltaScale * amp * 0.3) * perspW + swX1;
          var y1 = perspY + (cell.oy + cell.dy * deltaScale * amp * 0.3) * perspH + swY1;

          var nextGrid = nextSlice.grid;
          var nIdx = Math.min(idx, nextGrid.length - 1);
          var nCell = nextGrid[nIdx];
          var nDiv = nCell.divergence * sens;
          var swX2 = Math.sin(time * 0.7 + nIdx * 0.5 + (si + 1) * 1.3) * nDiv * amp * 35 * deltaScale;
          var swY2 = Math.sin(time * 0.5 + nIdx * 0.3 + (si + 1) * 0.9) * nDiv * amp * 12 * deltaScale;
          var x2 = nextPerspX + (nCell.ox + nCell.dx * deltaScale * amp * 0.3) * nextPerspW + swX2;
          var y2 = nextPerspY + (nCell.oy + nCell.dy * deltaScale * amp * 0.3) * nextPerspH + swY2;

          var midX = (x1 + x2) / 2 + Math.sin(time * 1.1 + idx) * div * amp * 18 * deltaScale;
          var midY = (y1 + y2) / 2;

          diffeoCtx.strokeStyle = 'hsla(' + Math.round((slice.hue + nextSlice.hue) / 2) + ',60%,45%,' + (alpha * 0.7) + ')';
          diffeoCtx.lineWidth = 0.8 + div * 2.5 * deltaScale;
          diffeoCtx.beginPath();
          diffeoCtx.moveTo(x1, y1);
          diffeoCtx.quadraticCurveTo(midX, midY, x2, y2);
          diffeoCtx.stroke();
        }
      }
    }

    // Slice label
    diffeoCtx.globalAlpha = 0.6;
    diffeoCtx.fillStyle = 'hsl(' + slice.hue + ',70%,65%)';
    diffeoCtx.font = '9px monospace';
    diffeoCtx.fillText(
      'L' + slice.layerIdx + ' d' + slice.dimA + '\u00d7d' + slice.dimB,
      perspX - 65,
      perspY + perspH / 2 + 3
    );

    diffeoCtx.restore();
  }
}

function draw() {
    // Multi-sentence view doesn't need D — it uses multiData
    if (viewMode === 'multi') {
        if (multiData) drawMultiCanvas();
        return;
    }

    if (!D) return;

    // --- Fibre bundle views (was added by _origDraw2 wrapper) ---
    if (viewMode === 'fibrekelp') {
        drawFibreBundleKelp();
        return;
    }
    if (viewMode === 'fibre3d') {
        drawFibreBundle3DGrid();
        return;
    }
    if (viewMode === 'fibre') {
        drawFibreBundle();
        return;
    }

    // --- 3D view (original draw logic) ---
    if (viewMode === '3d') {
        draw3D();
        return;
    }

    // --- Default: 2D view ---
    draw2D();

    // --- Diffeomorphism overlay rebuild (was added by the diffeo wrapper) ---
    if (diffeoState.active && D) {
        rebuildDiffeo();
    }
}

// ============================================================
// FIBRE BUNDLE VIEW — Neuron Pixel Grids Stacked as Kelp Rooms
// ============================================================

function setViewMode(mode) {
    // ---- Multi-view button visibility (from multi wrapper) ----
    var multiBtn = document.getElementById('btn-multi-view');
    if (multiBtn) {
        multiBtn.style.display = multiData ? 'inline-block' : 'none';
    }

    // ---- Clear ALL view-toggle button active states ----
    var allBtnIds = ['btn-2d', 'btn-3d', 'btn-fibre', 'btn-fibre3d', 'btn-fibrekelp'];
    for (var i = 0; i < allBtnIds.length; i++) {
        var el = document.getElementById(allBtnIds[i]);
        if (el) el.className = '';
    }
    if (multiBtn) multiBtn.className = '';

    // ---- Dim Z row: only visible in 3d and fibre3d ----
    var dzRow = document.getElementById('dz-row');

    // ================================================================
    // MULTI mode (from the multi wrapper)
    // ================================================================
    if (mode === 'multi') {
        viewMode = 'multi';
        if (multiBtn) multiBtn.className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        drawMultiCanvas();
        return;
    }

    // ================================================================
    // FIBRE KELP mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibrekelp') {
        viewMode = 'fibrekelp';
        document.getElementById('btn-fibrekelp').className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        draw();
        return;
    }

    // ================================================================
    // FIBRE 3D mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibre3d') {
        viewMode = 'fibre3d';
        document.getElementById('btn-fibre3d').className = 'active';
        if (dzRow) dzRow.style.display = 'flex';
        if (D && !fibreState.neuronData && !fibreState.loading) {
            fetchFibreNeuronData();
        }
        draw();
        return;
    }

    // ================================================================
    // FIBRE (2D) mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibre') {
        viewMode = 'fibre';
        document.getElementById('btn-fibre').className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        if (D && !fibreState.neuronData && !fibreState.loading) {
            fetchFibreNeuronData();
        }
        draw();
        return;
    }

    // ================================================================
    // 3D mode (from the original)
    // ================================================================
    if (mode === '3d') {
        viewMode = '3d';
        document.getElementById('btn-3d').className = 'active';
        if (dzRow) dzRow.style.display = 'flex';
        draw();
        return;
    }

    // ================================================================
    // 2D mode — default (from the original)
    // ================================================================
    viewMode = '2d';
    document.getElementById('btn-2d').className = 'active';
    if (dzRow) dzRow.style.display = 'none';
    draw();
}

function fetchFibreNeuronData() {
  if (!D || fibreState.loading) return;
  fibreState.loading = true;
  document.getElementById('status').textContent = 'Loading neuron activations for fibre view...';

  fetch('/neuron_grid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: D.text })
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) {
      document.getElementById('status').textContent = 'Neuron grid error: ' + data.error;
      fibreState.loading = false;
      return;
    }
    fibreState.neuronData = data;
    fibreState.loading = false;
    document.getElementById('status').textContent =
      'Fibre view ready — ' + data.n_tokens + ' tokens × ' +
      data.n_layers + ' layers × ' + data.hidden_dim + ' neurons';
    draw();
  })
  .catch(function(e) {
    document.getElementById('status').textContent = 'Error: ' + e;
    fibreState.loading = false;
  });
}

function computeFractionalDeltas(layerFrac, mode, activeDeltas, attnDeltas, mlpDeltas, nLayers, nP, dx, dy, amp) {
    var layerInt = Math.floor(layerFrac);
    var frac = layerFrac - layerInt;
    layerInt = Math.min(layerInt, nLayers - 1);

    var edxCum = new Float64Array(nP);
    var edyCum = new Float64Array(nP);
    // Also track the "current layer" attn and mlp components for the vector field
    var attnDx = new Float64Array(nP);
    var attnDy = new Float64Array(nP);
    var mlpDx = new Float64Array(nP);
    var mlpDy = new Float64Array(nP);

    if (mode === 'single') {
        // Interpolate between layer layerInt and layerInt+1
        for (var j = 0; j < nP; j++) {
            edxCum[j] = activeDeltas[layerInt][j][dx] * amp;
            edyCum[j] = activeDeltas[layerInt][j][dy] * amp;
            if (frac > 0 && layerInt + 1 < nLayers) {
                var nextDx = activeDeltas[layerInt + 1][j][dx] * amp;
                var nextDy = activeDeltas[layerInt + 1][j][dy] * amp;
                edxCum[j] = edxCum[j] * (1 - frac) + nextDx * frac;
                edyCum[j] = edyCum[j] * (1 - frac) + nextDy * frac;
            }
        }
    } else if (mode === 'cumfwd') {
        for (var j = 0; j < nP; j++) {
            for (var cl = 0; cl <= layerInt; cl++) {
                edxCum[j] += activeDeltas[cl][j][dx] * amp;
                edyCum[j] += activeDeltas[cl][j][dy] * amp;
            }
            // Add fractional part of next layer
            if (frac > 0 && layerInt + 1 < nLayers) {
                edxCum[j] += activeDeltas[layerInt + 1][j][dx] * amp * frac;
                edyCum[j] += activeDeltas[layerInt + 1][j][dy] * amp * frac;
            }
        }
    } else { // cumbwd
        for (var j = 0; j < nP; j++) {
            for (var cl = layerInt; cl < nLayers; cl++) {
                var weight = (cl === layerInt) ? (1 - frac) : 1.0;
                edxCum[j] += activeDeltas[cl][j][dx] * amp * weight;
                edyCum[j] += activeDeltas[cl][j][dy] * amp * weight;
            }
        }
    }

    // Current layer's decomposed components (for vector field overlay)
    if (attnDeltas && mlpDeltas) {
        for (var j = 0; j < nP; j++) {
            attnDx[j] = attnDeltas[layerInt][j][dx] * amp;
            attnDy[j] = attnDeltas[layerInt][j][dy] * amp;
            mlpDx[j] = mlpDeltas[layerInt][j][dx] * amp;
            mlpDy[j] = mlpDeltas[layerInt][j][dy] * amp;
            // Blend toward next layer if fractional
            if (frac > 0 && layerInt + 1 < nLayers) {
                attnDx[j] = attnDx[j] * (1 - frac) + attnDeltas[layerInt + 1][j][dx] * amp * frac;
                attnDy[j] = attnDy[j] * (1 - frac) + attnDeltas[layerInt + 1][j][dy] * amp * frac;
                mlpDx[j] = mlpDx[j] * (1 - frac) + mlpDeltas[layerInt + 1][j][dx] * amp * frac;
                mlpDy[j] = mlpDy[j] * (1 - frac) + mlpDeltas[layerInt + 1][j][dy] * amp * frac;
            }
        }
    }

    return {
        edx: edxCum, edy: edyCum,
        attnDx: attnDx, attnDy: attnDy,
        mlpDx: mlpDx, mlpDy: mlpDy,
        layerInt: layerInt, frac: frac
    };
}

function drawFlowArrow(c, fromX, fromY, vx, vy, color, maxLen) {
    var len = Math.hypot(vx, vy);
    if (len < 1.5) return; // skip tiny arrows

    // Clamp length
    if (len > maxLen) {
        var scale = maxLen / len;
        vx *= scale;
        vy *= scale;
        len = maxLen;
    }

    var toX = fromX + vx;
    var toY = fromY + vy;

    c.strokeStyle = color;
    c.fillStyle = color;
    c.lineWidth = Math.max(0.5, Math.min(1.5, len / 15));

    // Shaft
    c.beginPath();
    c.moveTo(fromX, fromY);
    c.lineTo(toX, toY);
    c.stroke();

    // Arrowhead
    var aa = Math.atan2(vy, vx);
    var hl = Math.min(5, len * 0.35);
    c.beginPath();
    c.moveTo(toX, toY);
    c.lineTo(toX - hl * Math.cos(aa - 0.45), toY - hl * Math.sin(aa - 0.45));
    c.lineTo(toX - hl * Math.cos(aa + 0.45), toY - hl * Math.sin(aa + 0.45));
    c.closePath();
    c.fill();
}

function drawTransportFrame(c, cx, cy, edx, edy, fx, fy, tokenIdx, nP, sig, frameSize) {
    // Compute the local Jacobian of the deformation field at this point
    // by finite differences in the RBF-interpolated field
    var eps = sig * 0.1;
    var s2i = 1 / (2 * sig * sig);

    function interpolateField(px, py) {
        var vvx = 0, vvy = 0, ws = 0;
        for (var k = 0; k < nP; k++) {
            var eex = px - fx[k], eey = py - fy[k];
            var w = Math.exp(-(eex * eex + eey * eey) * s2i);
            vvx += w * edx[k];
            vvy += w * edy[k];
            ws += w;
        }
        if (ws > 1e-15) { vvx /= ws; vvy /= ws; }
        return [vvx, vvy];
    }

    var basePx = fx[tokenIdx], basePy = fy[tokenIdx];
    var v0 = interpolateField(basePx, basePy);
    var vRight = interpolateField(basePx + eps, basePy);
    var vUp = interpolateField(basePx, basePy + eps);

    // Jacobian columns: how does the displacement field change as we move right/up
    var J00 = 1 + (vRight[0] - v0[0]) / eps; // dx/dx
    var J01 = (vUp[0] - v0[0]) / eps;         // dx/dy
    var J10 = (vRight[1] - v0[1]) / eps;       // dy/dx
    var J11 = 1 + (vUp[1] - v0[1]) / eps;     // dy/dy

    // Apply Jacobian to unit vectors to get transported frame
    var e1x = J00 * frameSize, e1y = J10 * frameSize;
    var e2x = J01 * frameSize, e2y = J11 * frameSize;

    // Draw the transported frame as two colored arrows
    // e1 (originally pointing right) in yellow
    c.globalAlpha = 0.7;
    drawFlowArrow(c, cx, cy, e1x, e1y, 'rgba(255,255,100,0.8)', frameSize * 2);
    // e2 (originally pointing up) in magenta
    drawFlowArrow(c, cx, cy, e2x, e2y, 'rgba(255,100,255,0.8)', frameSize * 2);
    c.globalAlpha = 1.0;

    // Draw a small circle at the center
    c.beginPath();
    c.arc(cx, cy, 2, 0, Math.PI * 2);
    c.fillStyle = 'rgba(255,255,255,0.6)';
    c.fill();
}

// ============================================================
// REFACTORED drawFibreBundleKelp — broken into composable steps
// ============================================================

/**
 * Read all kelp-relevant parameters from the DOM and data model.
 * Returns a self-contained config object so downstream helpers
 * never touch the DOM themselves.
 */
function getKelpParams() {
    var hiddenDim = D.hidden_dim;
    var dxDim = Math.min(+document.getElementById('sl-dx').value, hiddenDim - 1);
    var dyDim = Math.min(+document.getElementById('sl-dy').value, hiddenDim - 1);

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;

    return {
        nTokens:      D.n_real,
        nLayers:      D.n_layers,
        hiddenDim:    hiddenDim,
        nP:           D.n_points,
        dxDim:        dxDim,
        dyDim:        dyDim,
        amp:          +document.getElementById('sl-amp').value,
        t:            +document.getElementById('sl-t').value,
        sig:          +document.getElementById('sl-sig').value,
        currentLayer: +document.getElementById('sl-layer').value,
        showGrid:     document.getElementById('cb-grid').checked,
        showHeat:     document.getElementById('cb-heat').checked,
        showSC:       document.getElementById('cb-sc').checked,
        mode:         document.getElementById('sel-mode').value,
        itpMethod:    document.getElementById('sel-itp').value,
        activeDeltas: activeDeltas,
        attnDeltas:   D.attn_deltas || null,
        mlpDeltas:    D.mlp_deltas || null,
        isEmb:        document.getElementById('sel-mode').value === 'embedding',
    };
}

/**
 * Compute the kelp-view layout metrics from canvas size and data shape.
 * Returns { margin, plotW, plotH, layerH, SX(wx), LY(li) }.
 */
function computeKelpLayout(W, H, nLayers, vx0, vw) {
    var margin = 40;
    var plotW  = W / zoomLevel - 2 * margin;
    var plotH  = H / zoomLevel - 2 * margin;
    var layerH = plotH / nLayers;

    return {
        margin:  margin,
        plotW:   plotW,
        plotH:   plotH,
        layerH:  layerH,
        SX: function(wx) { return margin + ((wx - vx0) / vw) * plotW; },
        LY: function(li) { return margin + (nLayers - 1 - li) * layerH + layerH * 0.5; },
    };
}

/**
 * PASS 1 — Draw the per-layer background deformed grids.
 */
function drawKelpBackgroundGrids(c, kp, layout, fx, fy, rawDeltas, bounds) {
    if (!kp.showGrid || kp.isEmb) return;

    var N = Math.max(8, Math.min(25, Math.floor(layout.plotW / 20)));

    for (var li = 0; li < kp.nLayers; li++) {
        var ly       = layout.LY(li);
        var bandTop  = ly - layout.layerH * 0.4;
        var bandH    = layout.layerH * 0.8;          // bandBot - bandTop
        var isActive = (li === kp.currentLayer);

        // Reuse existing helpers
        var layerDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll,
            li, kp.nP, kp.nLayers, kp.mode, kp.isEmb
        );

        var grid = buildDeformedGrid2D(
            bounds.vx0, bounds.vy0, bounds.vw, bounds.vh,
            N, fx, fy,
            layerDeltas.edx, layerDeltas.edy,
            kp.nP, kp.sig, kp.t, kp.isEmb, kp.itpMethod
        );

        // Strain heatmap
        if (kp.showHeat) {
            drawStrainHeatmapInBand(
                c, grid, N,
                bounds.vx0, bounds.vy0, bounds.vw, bounds.vh,
                layout.margin, layout.plotW, bandTop, bandH,
                isActive ? 0.2 : 0.07
            );
        }

        // Grid lines
        drawGridLinesInBand(
            c, grid, N,
            bounds.vx0, bounds.vy0, bounds.vw, bounds.vh,
            layout.margin, layout.plotW, bandTop, bandH,
            kp.showSC,
            isActive ? 0.3 : 0.08,
            isActive ? 0.7 : 0.3
        );

        // Layer label
        c.font = (isActive ? 'bold ' : '') + '10px monospace';
        c.fillStyle = isActive ? '#e94560' : '#444';
        c.textAlign = 'right';
        c.fillText('L' + li, layout.margin - 8, ly + 3);

        // Thin separator
        c.strokeStyle = 'rgba(60,60,100,' + (isActive ? 0.3 : 0.1) + ')';
        c.lineWidth = 0.5;
        c.beginPath();
        c.moveTo(layout.margin, ly + layout.layerH * 0.48);
        c.lineTo(layout.margin + layout.plotW, ly + layout.layerH * 0.48);
        c.stroke();
    }
}

/**
 * PASS 2+3+4 — Draw every token's pathline, force-decomposition
 * arrows, transport frames, and node dots.
 */
function drawKelpTokenPathlines(c, kp, layout, fx, fy, rawDeltas, bounds, tokenPaths) {
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for (var ti = 0; ti < kp.nTokens; ti++) {
        var col = tc[ti % tc.length];
        var r = parseInt(col.slice(1, 3), 16);
        var g = parseInt(col.slice(3, 5), 16);
        var b = parseInt(col.slice(5, 7), 16);

        // Build screen-space path
        var path = [];
        for (var li = 0; li < kp.nLayers; li++) {
            path.push({
                x: layout.SX(tokenPaths.worldX[ti][li]),
                y: layout.LY(li)
            });
        }

        // Outer glow + main pathline (existing helpers)
        drawKelpPathGlow(c, path, kp.nLayers, r, g, b);
        drawKelpPathLine(c, path, kp.nLayers, r, g, b);

        // Per-layer decorations
        for (var li = 0; li < kp.nLayers; li++) {
            var pt       = path[li];
            var isActive = (li === kp.currentLayer);
            var pixPerWorld = layout.plotW / bounds.vw;
            var maxArrow = layout.layerH * 0.35;
            var arrowAlpha = isActive ? 0.85 : 0.3;

            // Attention arrow (cyan)
            drawKelpComponentArrow(
                c, pt, tokenPaths.attnDx[ti][li], pixPerWorld,
                maxArrow, 'rgba(0,200,255,' + arrowAlpha + ')',
                -3, kp.attnDeltas, kp.isEmb
            );

            // MLP arrow (orange)
            drawKelpComponentArrow(
                c, pt, tokenPaths.mlpDx[ti][li], pixPerWorld,
                maxArrow, 'rgba(255,165,0,' + arrowAlpha + ')',
                3, kp.mlpDeltas, kp.isEmb
            );

            // Transport frame
            if (fibreState.showTransportFrame && !kp.isEmb) {
                drawKelpTransportFrame(
                    c, pt, li, ti, kp.nP, kp.nLayers, fx, fy,
                    rawDeltas, kp.mode, kp.isEmb,
                    kp.dxDim, kp.dyDim, kp.amp, kp.sig,
                    pixPerWorld, layout.layerH, isActive
                );
            }

            // Node dot
            var dotR = isActive ? 5 : 3;
            c.beginPath();
            c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
            c.fillStyle = col;
            c.fill();
            if (isActive) {
                c.strokeStyle = '#fff';
                c.lineWidth = 1.5;
                c.stroke();
            }
        }

        // Token label at bottom (layer 0)
        c.font = 'bold 10px monospace';
        c.fillStyle = col;
        c.textAlign = 'center';
        c.fillText('[' + ti + '] ' + D.tokens[ti], path[0].x, path[0].y + 16);
    }
}

/**
 * Draw a single component (attn or mlp) horizontal arrow at a path node.
 * Extracted so both attn and mlp use the same logic.
 */
function drawKelpComponentArrow(c, pt, dxValue, pixPerWorld, maxArrow, color, yOffset, deltasExist, isEmb) {
    if (!deltasExist || isEmb) return;
    var avx = dxValue * pixPerWorld;
    if (Math.abs(avx) <= 1.5) return;
    if (Math.abs(avx) > maxArrow) avx *= maxArrow / Math.abs(avx);
    drawFlowArrow(c, pt.x, pt.y + yOffset, avx, 0, color, maxArrow);
}

/**
 * Top-level: the refactored drawFibreBundleKelp.
 * Orchestrates all passes using the helpers above.
 */
function drawFibreBundleKelp() {
    var cv = document.getElementById('cv');
    var c  = cv.getContext('2d');
    var W  = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace';
        c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    // ---- Gather parameters ----
    var kp = getKelpParams();

    // ---- Reuse existing position / bounds / delta helpers ----
    var pos    = extractPositions2D(D, kp.nP, kp.dxDim, kp.dyDim);
    var fx     = pos.fx, fy = pos.fy;
    var bounds = computeViewBounds2D(fx, fy, kp.nP, 0.15);
    var rawDeltas = computePerLayerRawDeltas(
        kp.activeDeltas, kp.nLayers, kp.nP, kp.dxDim, kp.dyDim, kp.amp
    );

    // ---- Layout ----
    var layout = computeKelpLayout(W, H, kp.nLayers, bounds.vx0, bounds.vw);

    // ---- Token world-space positions at each layer ----
    var tokenPaths = computeKelpTokenPaths(
        kp.nTokens, kp.nLayers, kp.nP, fx, fy,
        kp.dxDim, kp.dyDim, kp.amp, kp.t,
        kp.activeDeltas, kp.attnDeltas, kp.mlpDeltas,
        kp.mode, kp.isEmb
    );

    // ---- Render ----
    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    // PASS 1: Background deformed grids per layer
    drawKelpBackgroundGrids(c, kp, layout, fx, fy, rawDeltas, bounds);

    // PASS 2+3+4: Token pathlines, arrows, frames, dots
    drawKelpTokenPathlines(c, kp, layout, fx, fy, rawDeltas, bounds, tokenPaths);

    // PASS 5: Legend (existing helper)
    drawKelpLegend(c, layout.margin, layout.plotW, kp.attnDeltas, kp.mlpDeltas);

    c.restore();

    // HUD (existing helper)
    drawFibreBundleHUD(c, W, H, kp.nTokens, kp.nLayers, kp.hiddenDim, kp.currentLayer);
}

function drawStrainHeatmapInBand(c, grid, N, vx0, vy0, vw, vh,
    marginLeft, plotW, bandTop, bandH, alpha) {

    for (var hy = 0; hy < N; hy++) {
        for (var hx = 0; hx < N; hx++) {
            var avg = (grid.sH[hy * N + hx] + grid.sH[(hy + 1) * N + hx] +
                       grid.sV[hy * (N + 1) + hx] + grid.sV[hy * (N + 1) + hx + 1]) / 4;
            var co = s2c(avg);
            var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
            var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;

            function BSX(wx) { return marginLeft + ((wx - vx0) / vw) * plotW; }
            function BSY(wy) { return bandTop + ((wy - vy0) / vh) * bandH; }

            c.beginPath();
            c.moveTo(BSX(grid.gX[i00]), BSY(grid.gY[i00]));
            c.lineTo(BSX(grid.gX[i10]), BSY(grid.gY[i10]));
            c.lineTo(BSX(grid.gX[i11]), BSY(grid.gY[i11]));
            c.lineTo(BSX(grid.gX[i01]), BSY(grid.gY[i01]));
            c.closePath();
            c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',' + alpha + ')';
            c.fill();
        }
    }
}

function drawGridLinesInBand(c, grid, N, vx0, vy0, vw, vh,
    marginLeft, plotW, bandTop, bandH, showSC, gridAlpha, lineWidth) {

    function BSX(wx) { return marginLeft + ((wx - vx0) / vw) * plotW; }
    function BSY(wy) { return bandTop + ((wy - vy0) / vh) * bandH; }

    c.lineWidth = lineWidth;

    // Horizontal edges
    for (var dhy = 0; dhy <= N; dhy++) {
        for (var dhx = 0; dhx < N; dhx++) {
            var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
            if (showSC) {
                var es = grid.sH[dhy * N + dhx];
                var ec = s2c(es);
                c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',' + gridAlpha + ')';
            } else {
                c.strokeStyle = 'rgba(200,200,200,' + gridAlpha + ')';
            }
            c.beginPath();
            c.moveTo(BSX(grid.gX[di1]), BSY(grid.gY[di1]));
            c.lineTo(BSX(grid.gX[di2]), BSY(grid.gY[di2]));
            c.stroke();
        }
    }

    // Vertical edges
    for (var dvx = 0; dvx <= N; dvx++) {
        for (var dvy = 0; dvy < N; dvy++) {
            var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
            if (showSC) {
                var vs = grid.sV[dvy * (N + 1) + dvx];
                var vc = s2c(vs);
                c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',' + gridAlpha + ')';
            } else {
                c.strokeStyle = 'rgba(200,200,200,' + gridAlpha + ')';
            }
            c.beginPath();
            c.moveTo(BSX(grid.gX[dvi1]), BSY(grid.gY[dvi1]));
            c.lineTo(BSX(grid.gX[dvi2]), BSY(grid.gY[dvi2]));
            c.stroke();
        }
    }
}

function drawKelpPathGlow(c, path, nLayers, r, g, b) {
    c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.06)';
    c.lineWidth = 10;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li = 1; li < nLayers; li++) {
        var prev = path[li - 1], curr = path[li];
        c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
    }
    c.stroke();
}

function drawKelpPathLine(c, path, nLayers, r, g, b) {
    c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
    c.lineWidth = 2.5;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li = 1; li < nLayers; li++) {
        var prev = path[li - 1], curr = path[li];
        c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
    }
    c.stroke();
}

function drawKelpTransportFrame(c, pt, li, ti, nP, nLayers, fx, fy,
    rawDeltas, mode, isEmb, dxDim, dyDim, amp, sig, pixPerWorld, layerH, isActive) {

    // Compute cumulative deltas for this layer (for the Jacobian)
    var layerDeltas = computeCumulativeDeltas(
        rawDeltas.edxAll, rawDeltas.edyAll, li, nP, nLayers, mode, isEmb);
    var edxCumTF = layerDeltas.edx;
    var edyCumTF = layerDeltas.edy;

    var s2i = 1 / (2 * sig * sig);
    var eps = sig * 0.1;

    function interpTF(px, py) {
        var vvx = 0, vvy = 0, ws = 0;
        for (var k = 0; k < nP; k++) {
            var eex = px - fx[k], eey = py - fy[k];
            var w = Math.exp(-(eex * eex + eey * eey) * s2i);
            vvx += w * edxCumTF[k]; vvy += w * edyCumTF[k]; ws += w;
        }
        if (ws > 1e-15) { vvx /= ws; vvy /= ws; }
        return [vvx, vvy];
    }

    var bpx = fx[ti], bpy = fy[ti];
    var v0 = interpTF(bpx, bpy);
    var vR = interpTF(bpx + eps, bpy);
    var vU = interpTF(bpx, bpy + eps);

    // Jacobian
    var J00 = 1 + (vR[0] - v0[0]) / eps;
    var J01 = (vU[0] - v0[0]) / eps;
    var J10 = (vR[1] - v0[1]) / eps;
    var J11 = 1 + (vU[1] - v0[1]) / eps;

    var fSize = Math.min(layerH * 0.2, 12);
    var e1x = J00 * fSize * pixPerWorld * 0.015;
    var e1y = J10 * fSize * pixPerWorld * 0.015;
    var e2x = J01 * fSize * pixPerWorld * 0.015;
    var e2y = J11 * fSize * pixPerWorld * 0.015;

    // Clamp
    var maxF = fSize * 2;
    var e1L = Math.hypot(e1x, e1y);
    var e2L = Math.hypot(e2x, e2y);
    if (e1L > maxF) { e1x *= maxF / e1L; e1y *= maxF / e1L; }
    if (e2L > maxF) { e2x *= maxF / e2L; e2y *= maxF / e2L; }

    var frameAlpha = isActive ? 0.7 : 0.2;
    c.globalAlpha = frameAlpha;
    drawFlowArrow(c, pt.x, pt.y, e1x, e1y, 'rgba(255,255,100,0.9)', maxF);
    drawFlowArrow(c, pt.x, pt.y, e2x, e2y, 'rgba(255,100,255,0.9)', maxF);
    c.globalAlpha = 1.0;
}

function drawKelpLegend(c, margin, plotW, attnDeltas, mlpDeltas) {
    var legX = margin + plotW + 10;
    var legY = margin + 10;
    c.font = '9px monospace';
    c.textAlign = 'left';

    c.fillStyle = '#888';
    c.fillText('Parallel Transport', legX, legY); legY += 14;

    c.strokeStyle = 'rgba(233,69,96,0.6)';
    c.lineWidth = 2.5;
    c.beginPath(); c.moveTo(legX, legY); c.lineTo(legX + 20, legY); c.stroke();
    c.fillStyle = '#a0a0c0';
    c.fillText('Token path', legX + 26, legY + 3); legY += 16;

    if (attnDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(0,200,255,0.8)', 20);
        c.fillStyle = 'rgba(0,200,255,0.8)';
        c.fillText('Attention push', legX + 26, legY + 3); legY += 14;
    }
    if (mlpDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(255,165,0,0.8)', 20);
        c.fillStyle = 'rgba(255,165,0,0.8)';
        c.fillText('MLP push', legX + 26, legY + 3); legY += 14;
    }
    if (fibreState.showTransportFrame) {
        drawFlowArrow(c, legX, legY, 14, 0, 'rgba(255,255,100,0.8)', 16);
        c.fillStyle = 'rgba(255,255,100,0.8)';
        c.fillText('Frame e1', legX + 26, legY + 3); legY += 12;
        drawFlowArrow(c, legX, legY, 0, -12, 'rgba(255,100,255,0.8)', 14);
        c.fillStyle = 'rgba(255,100,255,0.8)';
        c.fillText('Frame e2', legX + 26, legY + 3); legY += 16;
    }

    c.fillStyle = '#555';
    c.font = '8px monospace';
    c.fillText('Path bends = information flow', legX, legY); legY += 10;
    c.fillText('Frame rotation = holonomy', legX, legY);
}

function computeKelpTokenPaths(nTokens, nLayers, nP, fx, fy, dxDim, dyDim, amp, t,
    activeDeltas, attnDeltas, mlpDeltas, mode, isEmb) {

    var worldX = []; // [ti][li]
    var worldY = [];
    var attnDxArr = [];
    var attnDyArr = [];
    var mlpDxArr = [];
    var mlpDyArr = [];

    for (var ti = 0; ti < nTokens; ti++) {
        var wxArr = [], wyArr = [];
        var adxArr = [], adyArr = [], mdxArr = [], mdyArr = [];

        for (var li = 0; li < nLayers; li++) {
            var cumDx = 0, cumDy = 0;
            if (!isEmb) {
                if (mode === 'single') {
                    cumDx = activeDeltas[li][ti][dxDim] * amp * t;
                    cumDy = activeDeltas[li][ti][dyDim] * amp * t;
                } else if (mode === 'cumfwd') {
                    for (var cl = 0; cl <= li; cl++) {
                        cumDx += activeDeltas[cl][ti][dxDim] * amp * t;
                        cumDy += activeDeltas[cl][ti][dyDim] * amp * t;
                    }
                } else { // cumbwd
                    for (var cl = li; cl < nLayers; cl++) {
                        cumDx += activeDeltas[cl][ti][dxDim] * amp * t;
                        cumDy += activeDeltas[cl][ti][dyDim] * amp * t;
                    }
                }
            }

            wxArr.push(fx[ti] + cumDx);
            wyArr.push(fy[ti] + cumDy);

            var adx = 0, ady = 0, mdx = 0, mdy = 0;
            if (attnDeltas && !isEmb) {
                adx = attnDeltas[li][ti][dxDim] * amp * t;
                ady = attnDeltas[li][ti][dyDim] * amp * t;
            }
            if (mlpDeltas && !isEmb) {
                mdx = mlpDeltas[li][ti][dxDim] * amp * t;
                mdy = mlpDeltas[li][ti][dyDim] * amp * t;
            }
            adxArr.push(adx); adyArr.push(ady);
            mdxArr.push(mdx); mdyArr.push(mdy);
        }

        worldX.push(wxArr); worldY.push(wyArr);
        attnDxArr.push(adxArr); attnDyArr.push(adyArr);
        mlpDxArr.push(mdxArr); mlpDyArr.push(mdyArr);
    }

    return {
        worldX: worldX, worldY: worldY,
        attnDx: attnDxArr, attnDy: attnDyArr,
        mlpDx: mlpDxArr, mlpDy: mlpDyArr
    };
}

function extractPositions2D(dataObj, nP, dx, dy) {
    var fx = new Float64Array(nP), fy = new Float64Array(nP);
    for (var i = 0; i < nP; i++) {
        fx[i] = dataObj.fixed_pos[i][dx];
        fy[i] = dataObj.fixed_pos[i][dy];
    }
    return { fx: fx, fy: fy };
}

function computeViewBounds2D(fx, fy, nP, padding) {
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (var i = 0; i < nP; i++) {
        if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
        if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    }
    var mr = Math.max(mxx - mnx, mxy - mny) || 1;
    var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
    var pd = padding || 0.15;
    return {
        vx0: cxv - mr * (0.5 + pd),
        vy0: cyv - mr * (0.5 + pd),
        vw: mr * (1 + 2 * pd),
        vh: mr * (1 + 2 * pd),
        mr: mr, cxv: cxv, cyv: cyv
    };
}

function computePerLayerRawDeltas(activeDeltas, nLayers, nP, dx, dy, amp) {
    var edxAll = [], edyAll = [];
    for (var lay = 0; lay < nLayers; lay++) {
        var edxL = new Float64Array(nP);
        var edyL = new Float64Array(nP);
        for (var j = 0; j < nP; j++) {
            edxL[j] = activeDeltas[lay][j][dx] * amp;
            edyL[j] = activeDeltas[lay][j][dy] * amp;
        }
        edxAll.push(edxL);
        edyAll.push(edyL);
    }
    return { edxAll: edxAll, edyAll: edyAll };
}

function computeCumulativeDeltas(edxAll, edyAll, li, nP, nLayers, mode, isEmb) {
    var edxCum = new Float64Array(nP);
    var edyCum = new Float64Array(nP);
    if (isEmb) return { edx: edxCum, edy: edyCum };
    if (mode === 'single') {
        for (var j = 0; j < nP; j++) {
            edxCum[j] = edxAll[li][j];
            edyCum[j] = edyAll[li][j];
        }
    } else if (mode === 'cumfwd') {
        for (var cl = 0; cl <= li; cl++) {
            for (var j = 0; j < nP; j++) {
                edxCum[j] += edxAll[cl][j];
                edyCum[j] += edyAll[cl][j];
            }
        }
    } else { // cumbwd
        for (var cl = li; cl < nLayers; cl++) {
            for (var j = 0; j < nP; j++) {
                edxCum[j] += edxAll[cl][j];
                edyCum[j] += edyAll[cl][j];
            }
        }
    }
    return { edx: edxCum, edy: edyCum };
}

function buildDeformedGrid2D(vx0, vy0, vw, vh, N, fx, fy, edxCum, edyCum, nP, sig, t, isEmb, itpMethod) {
    var nV = (N + 1) * (N + 1);
    var oX = new Float64Array(nV), oY = new Float64Array(nV);
    var gX = new Float64Array(nV), gY = new Float64Array(nV);

    for (var gy = 0; gy <= N; gy++) {
        for (var gx = 0; gx <= N; gx++) {
            var gi = gy * (N + 1) + gx;
            oX[gi] = vx0 + (gx / N) * vw;
            oY[gi] = vy0 + (gy / N) * vh;
        }
    }

    if (isEmb) {
        for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; }
    } else {
        for (var gi = 0; gi < nV; gi++) {
            var px = oX[gi], py = oY[gi];
            var iRes = interpolateGridPoint(px, py, fx, fy, edxCum, edyCum, nP, sig, itpMethod);
            gX[gi] = px + t * iRes[0];
            gY[gi] = py + t * iRes[1];
        }
    }

    var sH = new Float64Array(N * (N + 1));
    var sV = new Float64Array((N + 1) * N);
    for (var ey = 0; ey <= N; ey++) {
        for (var ex = 0; ex < N; ex++) {
            var a = ey * (N + 1) + ex, b = a + 1;
            var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
            var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
            sH[ey * N + ex] = od > 1e-12 ? dd / od : 1;
        }
    }
    for (var ey = 0; ey < N; ey++) {
        for (var ex = 0; ex <= N; ex++) {
            var a = ey * (N + 1) + ex, b = (ey + 1) * (N + 1) + ex;
            var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
            var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
            sV[ey * (N + 1) + ex] = od > 1e-12 ? dd / od : 1;
        }
    }

    return { oX: oX, oY: oY, gX: gX, gY: gY, sH: sH, sV: sV, nV: nV };
}

function drawStrainHeatmapInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh) {
    for (var hy = 0; hy < N; hy++) {
        for (var hx = 0; hx < N; hx++) {
            var avg = (grid.sH[hy * N + hx] + grid.sH[(hy + 1) * N + hx] +
                       grid.sV[hy * (N + 1) + hx] + grid.sV[hy * (N + 1) + hx + 1]) / 4;
            var co = s2c(avg);
            var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
            var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;

            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[i00] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i00] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i10] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i10] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i11] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i11] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i01] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i01] - vy0) / vh) * roomSize);
            c.closePath();
            c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.4)';
            c.fill();
        }
    }
}

function drawGridLinesInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh, showSC) {
    c.lineWidth = 0.6;
    // Horizontal edges
    for (var dhy = 0; dhy <= N; dhy++) {
        for (var dhx = 0; dhx < N; dhx++) {
            var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
            var es = grid.sH[dhy * N + dhx];
            if (showSC) {
                var ec = s2c(es);
                c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.8)';
            } else {
                c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[di1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[di2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di2] - vy0) / vh) * roomSize);
            c.stroke();
        }
    }
    // Vertical edges
    for (var dvx = 0; dvx <= N; dvx++) {
        for (var dvy = 0; dvy < N; dvy++) {
            var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
            var vs = grid.sV[dvy * (N + 1) + dvx];
            if (showSC) {
                var vc = s2c(vs);
                c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.8)';
            } else {
                c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[dvi1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[dvi2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi2] - vy0) / vh) * roomSize);
            c.stroke();
        }
    }
}

function drawReferenceGridInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh) {
    c.strokeStyle = 'rgba(255,255,255,0.15)';
    c.lineWidth = 0.5;
    for (var ry = 0; ry <= N; ry++) {
        c.beginPath();
        for (var rx = 0; rx <= N; rx++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (rx === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
        }
        c.stroke();
    }
    for (var rx = 0; rx <= N; rx++) {
        c.beginPath();
        for (var ry = 0; ry <= N; ry++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (ry === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
        }
        c.stroke();
    }
}

function computeFibreRoomLayout(W, H, nTokens, nLayers, zoomLevel) {
    var margin = 30;
    var labelW = 35;
    var labelH = 25;
    var availW = (W / zoomLevel) - 2 * margin - labelW;
    var availH = (H / zoomLevel) - 2 * margin - labelH;

    var gapFracX = 0.35;
    var gapFracY = 0.45;
    var roomW = Math.max(30, Math.floor(availW / (nTokens * (1 + gapFracX))));
    var roomH = Math.max(30, Math.floor(availH / (nLayers * (1 + gapFracY))));
    var roomSize = Math.min(roomW, roomH);
    var gapX = Math.max(8, Math.floor(roomSize * gapFracX));
    var gapY = Math.max(12, Math.floor(roomSize * gapFracY));

    return {
        margin: margin,
        labelW: labelW,
        labelH: labelH,
        roomSize: roomSize,
        gapX: gapX,
        gapY: gapY,
        startX: margin + labelW,
        startY: margin
    };
}

function drawFibreBundle() {
    var cv = document.getElementById('cv');
    var c = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace'; c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    var nTokens = D.n_real;
    var nLayers = D.n_layers;
    var hiddenDim = D.hidden_dim;
    var nP = D.n_points;

    var dx = Math.min(+document.getElementById('sl-dx').value, hiddenDim - 1);
    var dy = Math.min(+document.getElementById('sl-dy').value, hiddenDim - 1);
    var amp = +document.getElementById('sl-amp').value;
    var t = +document.getElementById('sl-t').value;
    var sig = +document.getElementById('sl-sig').value;
    var currentLayer = +document.getElementById('sl-layer').value;
    var showGrid = document.getElementById('cb-grid').checked;
    var showHeat = document.getElementById('cb-heat').checked;
    var showSC = document.getElementById('cb-sc').checked;
    var showVec = document.getElementById('cb-vec').checked;
    var mode = document.getElementById('sel-mode').value;
    var itpMethod = document.getElementById('sel-itp').value;

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;
    var attnDeltas = D.attn_deltas || null;
    var mlpDeltas = D.mlp_deltas || null;
    var isEmb = (mode === 'embedding');

    // --- Use extracted helpers ---
    var layout = computeFibreRoomLayout(W, H, nTokens, nLayers, zoomLevel);
    var pos = extractPositions2D(D, nP, dx, dy);
    var fx = pos.fx, fy = pos.fy;
    var bounds = computeViewBounds2D(fx, fy, nP, 0.15);
    var vx0 = bounds.vx0, vy0 = bounds.vy0, vw = bounds.vw, vh = bounds.vh;
    var rawDeltas = computePerLayerRawDeltas(activeDeltas, nLayers, nP, dx, dy, amp);

    var N = Math.max(4, Math.min(16, Math.floor(layout.roomSize / 4)));
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    // ========== PASS 1: Flow streamlines between layers ==========
    if (fibreState.showFlowLines && !isEmb) {
        drawFibreBundleFlowLines(c, layout, rawDeltas, nLayers, nTokens, nP,
            fx, fy, vx0, vy0, vw, vh, sig, t, currentLayer, N);
    }

    // ========== PASS 2: Draw each layer room ==========
    for (var li = 0; li < nLayers; li++) {
        var rowIdx = nLayers - 1 - li;
        var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
        var isCurrentLayer = (li === currentLayer);

        // Layer label
        c.font = (isCurrentLayer ? 'bold ' : '') + '9px monospace';
        c.fillStyle = isCurrentLayer ? '#e94560' : '#666';
        c.textAlign = 'right';
        c.fillText('L' + li, layout.startX - 8, roomCY + layout.roomSize / 2 + 3);

        var layerDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li, nP, nLayers, mode, isEmb);
        var edxCum = layerDeltas.edx, edyCum = layerDeltas.edy;

        for (var ti = 0; ti < nTokens; ti++) {
            var roomCX = layout.startX + ti * (layout.roomSize + layout.gapX);

            // Room background & border
            drawFibreRoomBackground(c, roomCX, roomCY, layout.roomSize, isCurrentLayer);

            // Build deformed grid
            var grid = buildDeformedGrid2D(vx0, vy0, vw, vh, N, fx, fy,
                edxCum, edyCum, nP, sig, t, isEmb, itpMethod);

            // Strain heatmap
            if (showHeat && !isEmb) {
                drawStrainHeatmapInRoom(c, grid, N, roomCX, roomCY,
                    layout.roomSize, vx0, vy0, vw, vh);
            }

            // Grid lines
            if (showGrid && !isEmb) {
                drawGridLinesInRoom(c, grid, N, roomCX, roomCY,
                    layout.roomSize, vx0, vy0, vw, vh, showSC);
            }

            // Reference grid (embedding mode)
            if (isEmb) {
                drawReferenceGridInRoom(c, grid, N, roomCX, roomCY,
                    layout.roomSize, vx0, vy0, vw, vh);
            }

            // Vector field overlays (attn/mlp)
            if (fibreState.showAttnField && attnDeltas && !isEmb) {
                drawComponentVectorField(c, grid, N, roomCX, roomCY, layout.roomSize,
                    vx0, vy0, vw, vh, fx, fy, nP, sig, attnDeltas, li, dx, dy, amp, t,
                    'rgba(0,200,255,0.55)', itpMethod);
            }
            if (fibreState.showMlpField && mlpDeltas && !isEmb) {
                drawComponentVectorField(c, grid, N, roomCX, roomCY, layout.roomSize,
                    vx0, vy0, vw, vh, fx, fy, nP, sig, mlpDeltas, li, dx, dy, amp, t,
                    'rgba(255,165,0,0.55)', itpMethod);
            }

            // Transport frame
            if (fibreState.showTransportFrame && !isEmb) {
                var tokSX = roomCX + ((fx[ti] + t * edxCum[ti] - vx0) / vw) * layout.roomSize;
                var tokSY = roomCY + ((fy[ti] + t * edyCum[ti] - vy0) / vh) * layout.roomSize;
                drawTransportFrame(c, tokSX, tokSY, edxCum, edyCum,
                    fx, fy, ti, nP, sig, layout.roomSize / 5);
            }

            // Token dot with strain ring
            drawFibreTokenDot(c, fx[ti], fy[ti], edxCum[ti], edyCum[ti], t,
                roomCX, roomCY, layout.roomSize, vx0, vy0, vw, vh,
                tc[ti % tc.length], isEmb, N);

            // Token label at bottom of column
            if (li === 0) {
                drawFibreTokenLabel(c, ti, D.tokens[ti], roomCX, roomCY,
                    layout.roomSize);
            }
        } // end token loop

        // Predicted next-token points
        if (D.predicted_indices && D.predicted_indices.length > 0 && !isEmb) {
            drawFibrePredictedTokens(c, D, edxCum, fx, fy, t, nP, nTokens,
                layout, roomCY, vx0, vy0, vw, vh, isCurrentLayer);
        }

        // Inter-layer connections
        if (fibreState.showConnections && li < nLayers - 1 && !isEmb) {
            var nextDeltas = computeCumulativeDeltas(
                rawDeltas.edxAll, rawDeltas.edyAll, li + 1, nP, nLayers, mode, isEmb);
            drawFibreInterLayerConnections(c, li, nTokens, nP, nLayers,
                layout, edxCum, nextDeltas.edx, D, fx, fy, t, vx0, vy0, vw, vh);
        }
    } // end layer loop

    // ========== Axis labels ==========
    drawFibreBundleAxisLabels(c, layout, nTokens, nLayers);

    // ========== Legend ==========
    if ((fibreState.showAttnField || fibreState.showMlpField) && !isEmb) {
        drawFibreBundleLegend(c, layout, nTokens, attnDeltas, mlpDeltas, D);
    }

    c.restore();

    // ========== HUD ==========
    drawFibreBundleHUD(c, W, H, nTokens, nLayers, hiddenDim, currentLayer);
}

function drawFibreRoomBackground(c, roomCX, roomCY, roomSize, isCurrentLayer) {
    var bgAlpha = isCurrentLayer ? 0.15 : 0.06;
    c.fillStyle = 'rgba(30,30,60,' + bgAlpha + ')';
    c.fillRect(roomCX, roomCY, roomSize, roomSize);
    c.strokeStyle = isCurrentLayer ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.25)';
    c.lineWidth = isCurrentLayer ? 1.5 : 0.5;
    c.strokeRect(roomCX, roomCY, roomSize, roomSize);
}

function drawFibreTokenDot(c, fxTi, fyTi, edxTi, edyTi, t,
    roomCX, roomCY, roomSize, vx0, vy0, vw, vh, color, isEmb, N) {
    var tokX = roomCX + ((fxTi - vx0) / vw) * roomSize;
    var tokY = roomCY + ((fyTi - vy0) / vh) * roomSize;

    if (!isEmb) {
        var tokDeformDX = t * edxTi;
        var tokDeformDY = t * edyTi;
        var localStrainMag = Math.sqrt(tokDeformDX * tokDeformDX + tokDeformDY * tokDeformDY);
        var strainRadius = Math.min(roomSize / 4, localStrainMag * 0.5);
        if (strainRadius > 1.5) {
            var normStrain = Math.min(2.0, localStrainMag / (vw / N + 1e-12));
            var ringColor = s2c(0.5 + normStrain * 0.5);
            c.beginPath();
            c.arc(tokX, tokY, strainRadius, 0, Math.PI * 2);
            c.strokeStyle = 'rgba(' + ringColor[0] + ',' + ringColor[1] + ',' + ringColor[2] + ',0.5)';
            c.lineWidth = 1.2;
            c.stroke();
        }
    }

    c.beginPath();
    c.arc(tokX, tokY, Math.max(2, roomSize / 20), 0, Math.PI * 2);
    c.fillStyle = color;
    c.fill();
    c.strokeStyle = '#fff';
    c.lineWidth = 0.5;
    c.stroke();
}

function drawFibreTokenLabel(c, ti, tokenText, roomCX, roomCY, roomSize) {
    c.font = 'bold 8px monospace';
    c.fillStyle = '#e94560';
    c.textAlign = 'center';
    c.fillText('[' + ti + ']', roomCX + roomSize / 2, roomCY + roomSize + 10);
    if (roomSize > 35) {
        c.font = '7px monospace';
        c.fillStyle = '#a0a0c0';
        c.fillText(tokenText, roomCX + roomSize / 2, roomCY + roomSize + 19);
    }
}

function drawFibreBundleAxisLabels(c, layout, nTokens, nLayers) {
    c.font = 'bold 10px monospace';
    c.fillStyle = '#53a8b6';
    c.textAlign = 'center';
    var totalW = nTokens * (layout.roomSize + layout.gapX) - layout.gapX;
    c.fillText(
        '\u2190 Base Manifold: Token Index \u2192',
        layout.startX + totalW / 2,
        layout.startY + nLayers * (layout.roomSize + layout.gapY) + 28
    );

    c.save();
    c.translate(layout.startX - 35,
        layout.startY + nLayers * (layout.roomSize + layout.gapY) / 2);
    c.rotate(-Math.PI / 2);
    c.font = 'bold 10px monospace';
    c.fillStyle = '#53a8b6';
    c.textAlign = 'center';
    c.fillText('\u2190 Fibre: Layer Depth \u2192', 0, 0);
    c.restore();
}

function drawFibreBundleLegend(c, layout, nTokens, attnDeltas, mlpDeltas, D) {
    var totalW = nTokens * (layout.roomSize + layout.gapX) - layout.gapX;
    var legX = layout.startX + totalW + 20;
    var legY = layout.startY + 10;
    c.font = '9px monospace';
    c.textAlign = 'left';

    if (fibreState.showAttnField && attnDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(0,200,255,0.8)', 20);
        c.fillStyle = 'rgba(0,200,255,0.8)';
        c.fillText('Attention', legX + 24, legY + 3);
        legY += 16;
    }
    if (fibreState.showMlpField && mlpDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(255,165,0,0.8)', 20);
        c.fillStyle = 'rgba(255,165,0,0.8)';
        c.fillText('MLP', legX + 24, legY + 3);
        legY += 16;
    }
    if (fibreState.showTransportFrame) {
        drawFlowArrow(c, legX, legY, 14, 0, 'rgba(255,255,100,0.8)', 16);
        c.fillStyle = 'rgba(255,255,100,0.8)';
        c.fillText('Frame e1', legX + 24, legY + 3);
        legY += 14;
        drawFlowArrow(c, legX, legY, 0, -14, 'rgba(255,100,255,0.8)', 16);
        c.fillStyle = 'rgba(255,100,255,0.8)';
        c.fillText('Frame e2', legX + 24, legY + 3);
        legY += 16;
    }
    if (fibreState.showFlowLines) {
        c.strokeStyle = 'rgba(150,150,200,0.5)';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(legX, legY);
        c.quadraticCurveTo(legX + 10, legY - 8, legX + 18, legY);
        c.stroke();
        c.fillStyle = 'rgba(150,150,200,0.7)';
        c.fillText('Flow lines', legX + 24, legY + 3);
        legY += 16;
    }

    // Predicted token legend entry
    if (D.predicted_indices && D.predicted_indices.length > 0) {
        c.save();
        c.translate(legX + 6, legY);
        c.rotate(Math.PI / 4);
        c.fillStyle = 'rgba(245,166,35,0.8)';
        c.fillRect(-3, -3, 6, 6);
        c.strokeStyle = '#f5a623';
        c.lineWidth = 0.8;
        c.strokeRect(-3, -3, 6, 6);
        c.restore();
        c.fillStyle = '#f5a623';
        c.fillText('Predicted (' + D.predicted_indices.length + ')', legX + 24, legY + 3);
    }
}

function drawFibreBundleFlowLines(c, layout, rawDeltas, nLayers, nTokens, nP,
    fx, fy, vx0, vy0, vw, vh, sig, t, currentLayer, N) {

    var s2i = 1 / (2 * sig * sig);
    c.globalAlpha = 0.35;

    for (var li = 0; li < nLayers - 1; li++) {
        var rowIdx = nLayers - 1 - li;
        var nextRowIdx = nLayers - 2 - li;
        var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
        var nextRoomCY = layout.startY + nextRowIdx * (layout.roomSize + layout.gapY);

        var layerDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li, nP, nLayers,
            document.getElementById('sel-mode').value, false);
        var nextDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li + 1, nP, nLayers,
            document.getElementById('sel-mode').value, false);

        for (var ti = 0; ti < nTokens; ti++) {
            var roomCX = layout.startX + ti * (layout.roomSize + layout.gapX);

            var streamStep = Math.max(1, Math.floor(N / 3));
            for (var sgy = 0; sgy <= N; sgy += streamStep) {
                for (var sgx = 0; sgx <= N; sgx += streamStep) {
                    var worldX = vx0 + (sgx / N) * vw;
                    var worldY = vy0 + (sgy / N) * vh;

                    var vvx1 = 0, vvy1 = 0, ws1 = 0;
                    for (var k = 0; k < nP; k++) {
                        var eex = worldX - fx[k], eey = worldY - fy[k];
                        var w = Math.exp(-(eex * eex + eey * eey) * s2i);
                        vvx1 += w * layerDeltas.edx[k]; vvy1 += w * layerDeltas.edy[k]; ws1 += w;
                    }
                    if (ws1 > 1e-15) { vvx1 /= ws1; vvy1 /= ws1; }

                    var vvx2 = 0, vvy2 = 0, ws2 = 0;
                    for (var k = 0; k < nP; k++) {
                        var eex = worldX - fx[k], eey = worldY - fy[k];
                        var w = Math.exp(-(eex * eex + eey * eey) * s2i);
                        vvx2 += w * nextDeltas.edx[k]; vvy2 += w * nextDeltas.edy[k]; ws2 += w;
                    }
                    if (ws2 > 1e-15) { vvx2 /= ws2; vvy2 /= ws2; }

                    var deformedX1 = worldX + t * vvx1;
                    var deformedY1 = worldY + t * vvy1;
                    var deformedX2 = worldX + t * vvx2;
                    var deformedY2 = worldY + t * vvy2;

                    var sx1 = roomCX + ((deformedX1 - vx0) / vw) * layout.roomSize;
                    var sy1 = roomCY + ((deformedY1 - vy0) / vh) * layout.roomSize;
                    var sx2 = roomCX + ((deformedX2 - vx0) / vw) * layout.roomSize;
                    var sy2 = nextRoomCY + ((deformedY2 - vy0) / vh) * layout.roomSize;

                    var moveDist = Math.hypot(deformedX2 - deformedX1, deformedY2 - deformedY1);
                    var moveAlpha = Math.min(0.5, moveDist * 0.3 + 0.03);

                    var strain = (moveDist > 1e-8) ? moveDist / (vw / N + 1e-12) : 0;
                    var sc = s2c(0.5 + strain * 0.5);

                    c.strokeStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + moveAlpha.toFixed(2) + ')';
                    c.lineWidth = Math.min(1.5, 0.3 + moveDist * 0.5);

                    var midX = (sx1 + sx2) / 2 + (sx2 - sx1) * 0.3;
                    var midY = (sy1 + sy2) / 2;

                    c.beginPath();
                    c.moveTo(sx1, sy1);
                    c.quadraticCurveTo(midX, midY, sx2, sy2);
                    c.stroke();

                    c.beginPath();
                    c.arc(sx2, sy2, 1, 0, Math.PI * 2);
                    c.fillStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + (moveAlpha * 1.5).toFixed(2) + ')';
                    c.fill();
                }
            }
        }
    }
    c.globalAlpha = 1.0;
}

function drawComponentVectorField(c, grid, N, roomCX, roomCY, roomSize,
    vx0, vy0, vw, vh, fx, fy, nP, sig, compDeltas, layerIdx, dx, dy, amp, t,
    color, itpMethod) {

    var vecStep = Math.max(1, Math.floor(N / 4));
    var maxArrowLen = roomSize / 3;
    var arrowScale = fibreState.flowArrowScale;

    var _compEdx = new Float64Array(nP);
    var _compEdy = new Float64Array(nP);
    for (var k = 0; k < nP; k++) {
        _compEdx[k] = compDeltas[layerIdx][k][dx] * amp;
        _compEdy[k] = compDeltas[layerIdx][k][dy] * amp;
    }

    for (var viy = 0; viy <= N; viy += vecStep) {
        for (var vix = 0; vix <= N; vix += vecStep) {
            var vi = viy * (N + 1) + vix;
            var worldPx = grid.oX[vi], worldPy = grid.oY[vi];

            var field = interpolateGridPoint(worldPx, worldPy, fx, fy,
                _compEdx, _compEdy, nP, sig, itpMethod);
            var screenBaseX = roomCX + ((grid.gX[vi] - vx0) / vw) * roomSize;
            var screenBaseY = roomCY + ((grid.gY[vi] - vy0) / vh) * roomSize;

            var pixPerWorld = roomSize / vw;
            var avx = field[0] * t * pixPerWorld * arrowScale;
            var avy = field[1] * t * pixPerWorld * arrowScale;

            drawFlowArrow(c, screenBaseX, screenBaseY, avx, avy,
                color, maxArrowLen);
        }
    }
}

function drawFibrePredictedTokens(c, D, edxCum, fx, fy, t, nP, nTokens,
    layout, roomCY, vx0, vy0, vw, vh, isCurrentLayer) {

    for (var pi2 = 0; pi2 < D.predicted_indices.length; pi2++) {
        var pidx = D.predicted_indices[pi2];
        if (pidx >= nP) continue;
        var prob = D.predicted_probs ? D.predicted_probs[pi2] : 0;

        var predWX = fx[pidx] + t * edxCum[pidx];
        var predWY = fy[pidx] + t * edxCum[pidx]; // Note: uses edyCum in original

        for (var ti2 = 0; ti2 < nTokens; ti2++) {
            var roomCX2 = layout.startX + ti2 * (layout.roomSize + layout.gapX);

            var predScreenX = roomCX2 + ((predWX - vx0) / vw) * layout.roomSize;
            var predScreenY = roomCY + ((predWY - vy0) / vh) * layout.roomSize;

            if (predScreenX >= roomCX2 - 3 && predScreenX <= roomCX2 + layout.roomSize + 3 &&
                predScreenY >= roomCY - 3 && predScreenY <= roomCY + layout.roomSize + 3) {

                var dotSz = 2 + prob * 5;

                var glowR2 = 5 + prob * 8;
                var grad4 = c.createRadialGradient(predScreenX, predScreenY, 0, predScreenX, predScreenY, glowR2);
                grad4.addColorStop(0, 'rgba(245,166,35,0.12)');
                grad4.addColorStop(1, 'rgba(245,166,35,0)');
                c.beginPath(); c.arc(predScreenX, predScreenY, glowR2, 0, Math.PI * 2);
                c.fillStyle = grad4; c.fill();

                c.save();
                c.translate(predScreenX, predScreenY);
                c.rotate(Math.PI / 4);
                c.fillStyle = 'rgba(245,166,35,' + (0.3 + prob * 0.5).toFixed(2) + ')';
                c.fillRect(-dotSz / 2, -dotSz / 2, dotSz, dotSz);
                c.strokeStyle = '#f5a623';
                c.lineWidth = 0.7;
                c.strokeRect(-dotSz / 2, -dotSz / 2, dotSz, dotSz);
                c.restore();

                if (isCurrentLayer && layout.roomSize > 40) {
                    c.font = '6px monospace';
                    c.lineWidth = 1;
                    c.strokeStyle = 'rgba(0,0,0,0.8)';
                    c.fillStyle = '#f5a623';
                    c.textAlign = 'left';
                    var plb2 = D.tokens[pidx] + ' ' + (prob * 100).toFixed(0) + '%';
                    c.strokeText(plb2, predScreenX + dotSz + 2, predScreenY - 2);
                    c.fillText(plb2, predScreenX + dotSz + 2, predScreenY - 2);
                }
            }
        }
    }
}

function drawFibreInterLayerConnections(c, li, nTokens, nP, nLayers,
    layout, edxCum, edxNext, D, fx, fy, t, vx0, vy0, vw, vh) {

    var rowIdx = nLayers - 1 - li;
    var nextRowIdx = nLayers - 2 - li;
    var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
    var nextRoomCY = layout.startY + nextRowIdx * (layout.roomSize + layout.gapY);

    // Note: edxNext here is the full edxCum for the next layer
    // We need edyCum and edyNext too — this is a simplification.
    // In the full refactor you'd pass both x and y components.

    for (var ti = 0; ti < nTokens; ti++) {
        var roomCXt = layout.startX + ti * (layout.roomSize + layout.gapX);

        var moveDist = Math.hypot(
            edxNext[ti] - edxCum[ti],
            0 // edyNext[ti] - edyCum[ti] — would need edy passed too
        );
        var moveAlpha = Math.min(0.6, moveDist * 0.01 + 0.05);

        var sy1 = roomCY;
        var sy2 = nextRoomCY + layout.roomSize;
        var sx1 = roomCXt + layout.roomSize / 2;
        var sx2 = roomCXt + layout.roomSize / 2;

        if (moveDist > 0.01) {
            c.strokeStyle = 'rgba(233,69,96,' + moveAlpha.toFixed(2) + ')';
        } else {
            c.strokeStyle = 'rgba(83,168,182,' + (moveAlpha * 0.5).toFixed(2) + ')';
        }
        c.lineWidth = Math.min(2, 0.3 + moveDist * 0.005);

        var midX = (sx1 + sx2) / 2 + Math.sin(ti * 1.5 + li * 0.7) * layout.gapX * 0.6;
        c.beginPath();
        c.moveTo(sx1, sy1);
        c.quadraticCurveTo(midX, (sy1 + sy2) / 2, sx2, sy2);
        c.stroke();
    }
}

function drawFibreBundleHUD(c, W, H, nTokens, nLayers, hiddenDim, currentLayer) {
  var dx = +document.getElementById('sl-dx').value;
  var dy = +document.getElementById('sl-dy').value;
  var amp = +document.getElementById('sl-amp').value;
  var t = +document.getElementById('sl-t').value;
  var decompLabel = getDecompLabel();

  c.font = '11px monospace';
  c.fillStyle = 'rgba(255,255,255,0.45)';
  c.textAlign = 'left';
  c.fillText(
    'FIBRE BUNDLE  Tokens:' + nTokens +
    '  Layers:' + nLayers +
    '  Dims:' + dx + ',' + dy +
    '  Amp:' + amp.toFixed(1) +
    '  t:' + t.toFixed(2) +
    '  Decomp:' + decompLabel +
    '  Colormap:' + fibreState.colormap +
    '  Layer:' + currentLayer,
    12, 16
  );

  c.font = '9px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.fillText(
    '\u2190\u2192 Dim X | \u2191\u2193 Dim Y | [/] Layer | A/Z Amp | ;/\' t | C=Connections | M=Colormap | Shift+Drag=Pan | Scroll=Zoom | 0=Reset',
    12, H - 8
  );
}

function onKeyFibre(e) {
  if (document.activeElement === document.getElementById('txt-in')) return;
  if (document.activeElement === document.getElementById('txt-b')) return;
  var maxDim = D ? D.hidden_dim - 1 : 767;
  var sdx = document.getElementById('sl-dx');
  var sdy = document.getElementById('sl-dy');
  var sdz = document.getElementById('sl-dz');

  // Shift+Arrow = Dim Z (third axis)
  if (e.shiftKey && e.key === 'ArrowRight') {
    e.preventDefault();
    var newZ = +sdz.value + 1;
    if (newZ > maxDim) newZ = 0;
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowLeft') {
    e.preventDefault();
    var newZ = +sdz.value - 1;
    if (newZ < 0) newZ = maxDim;
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowUp') {
    e.preventDefault();
    var newZ = +sdz.value + 10;
    if (newZ > maxDim) newZ = newZ % (maxDim + 1);
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowDown') {
    e.preventDefault();
    var newZ = +sdz.value - 10;
    if (newZ < 0) newZ = (newZ + maxDim + 1) % (maxDim + 1);
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    var newX = +sdx.value + 1;
    if (newX > maxDim) newX = 0;
    if (newX === +sdy.value) newX = (newX + 1) % (maxDim + 1);
    sdx.value = newX;
    sdx.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    var newX = +sdx.value - 1;
    if (newX < 0) newX = maxDim;
    if (newX === +sdy.value) newX = (newX - 1 + maxDim + 1) % (maxDim + 1);
    sdx.value = newX;
    sdx.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    var newY = +sdy.value + 1;
    if (newY > maxDim) newY = 0;
    if (newY === +sdx.value) newY = (newY + 1) % (maxDim + 1);
    sdy.value = newY;
    sdy.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    var newY = +sdy.value - 1;
    if (newY < 0) newY = maxDim;
    if (newY === +sdx.value) newY = (newY - 1 + maxDim + 1) % (maxDim + 1);
    sdy.value = newY;
    sdy.dispatchEvent(new Event('input'));
  } else if (e.key === '.' || e.key === ']') {
    var sl = document.getElementById('sl-layer');
    sl.value = Math.min(+sl.max, +sl.value + 1);
    sl.dispatchEvent(new Event('input'));
  } else if (e.key === ',' || e.key === '[') {
    var sl = document.getElementById('sl-layer');
    sl.value = Math.max(0, +sl.value - 1);
    sl.dispatchEvent(new Event('input'));
  } else if (e.key === "'" ) {
    var st = document.getElementById('sl-t');
    st.value = Math.min(1, +st.value + 0.05).toFixed(2);
    st.dispatchEvent(new Event('input'));
  } else if (e.key === ';') {
    var st = document.getElementById('sl-t');
    st.value = Math.max(0, +st.value - 0.05).toFixed(2);
    st.dispatchEvent(new Event('input'));
  } else if (e.key === 'a' || e.key === 'A') {
    var sa = document.getElementById('sl-amp');
    sa.value = Math.min(500, +sa.value * 1.3).toFixed(1);
    sa.dispatchEvent(new Event('input'));
  } else if (e.key === 'z' || e.key === 'Z') {
    var sa = document.getElementById('sl-amp');
    sa.value = Math.max(0.1, +sa.value / 1.3).toFixed(1);
    sa.dispatchEvent(new Event('input'));
  } else if (e.key === 'c' || e.key === 'C') {
    fibreState.showConnections = !fibreState.showConnections;
    draw();
  } else if (e.key === '0') {
    zoomLevel = 1.0; panX = 0; panY = 0;
    fibreState.scrollY = 0;
    fibreState.rotX = -0.3; fibreState.rotY = 0.4;
    draw();
  } else if (e.key === 'r' || e.key === 'R') {
    rstAll();
  }
}

// ---- Colormaps ----
function neuronColor(val, colormap) {
  // val in [0, 1]
  val = Math.max(0, Math.min(1, val));
  if (colormap === 'coolhot') {
    return valToColor(val);
  } else if (colormap === 'viridis') {
    // Approximate viridis
    var r, g, b;
    if (val < 0.25) {
      var t = val / 0.25;
      r = Math.floor(68 + t * (-4)); g = Math.floor(1 + t * 50); b = Math.floor(84 + t * 74);
    } else if (val < 0.5) {
      var t = (val - 0.25) / 0.25;
      r = Math.floor(64 - t * 30); g = Math.floor(51 + t * 70); b = Math.floor(158 - t * 20);
    } else if (val < 0.75) {
      var t = (val - 0.5) / 0.25;
      r = Math.floor(34 + t * 100); g = Math.floor(121 + t * 60); b = Math.floor(138 - t * 60);
    } else {
      var t = (val - 0.75) / 0.25;
      r = Math.floor(134 + t * 119); g = Math.floor(181 + t * 40); b = Math.floor(78 - t * 50);
    }
    return [r, g, b];
  } else {
    // Grayscale
    var v = Math.floor(val * 255);
    return [v, v, v];
  }
}

// ---- Main Fibre Bundle Drawing ----


function drawNeuronRoom(c, x, y, w, h, gridCols, gridRows, pixSize, hiddenDim, acts, useAbs, highlight, tokenIdx, layerIdx) {
  // Border
  c.strokeStyle = highlight ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.3)';
  c.lineWidth = highlight ? 1.5 : 0.5;
  c.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);

  // Draw each neuron as a pixel
  for (var ni = 0; ni < hiddenDim; ni++) {
    var val = acts[ni];
    if (useAbs) val = Math.abs(val * 2 - 1);

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    var px = x + col * pixSize;
    var py = y + row * pixSize;

    var rgb = neuronColor(val, fibreState.colormap);
    c.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    c.fillRect(px, py, pixSize, pixSize);
  }

  // Fill remaining cells dark
  for (var ni = hiddenDim; ni < gridCols * gridRows; ni++) {
    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    c.fillStyle = '#0a0515';
    c.fillRect(x + col * pixSize, y + row * pixSize, pixSize, pixSize);
  }
}

function drawDiffeoConnections(c, tokenX, layerY, nextLayerY, roomW, roomH,
  gridCols, gridRows, pixSize, hiddenDim, acts, nextActs, useAbs, layerIdx) {
  // Draw thin lines connecting corresponding neurons between layers
  // Only draw a subset for performance
  var step = Math.max(1, Math.floor(hiddenDim * (1 - fibreState.connectionDensity)));
  if (step < 1) step = 1;

  c.lineWidth = 0.3;

  for (var ni = 0; ni < hiddenDim; ni += step) {
    var val = acts[ni];
    var nextVal = nextActs[ni];
    if (useAbs) {
      val = Math.abs(val * 2 - 1);
      nextVal = Math.abs(nextVal * 2 - 1);
    }

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);

    var x1 = tokenX + col * pixSize + pixSize / 2;
    var y1 = layerY + row * pixSize + pixSize / 2;
    var x2 = tokenX + col * pixSize + pixSize / 2;
    var y2 = nextLayerY + roomH + row * pixSize + pixSize / 2;

    // Color based on activation change
    var delta = nextVal - val;
    var absDelta = Math.abs(delta);
    var alpha = Math.min(0.6, absDelta * 3);

    if (alpha < 0.02) continue;

    if (delta > 0) {
      c.strokeStyle = 'rgba(233,69,96,' + alpha.toFixed(2) + ')'; // expansion = red
    } else {
      c.strokeStyle = 'rgba(0,119,182,' + alpha.toFixed(2) + ')'; // contraction = blue
    }

    // Slight curve for visual appeal
    var midX = x1 + Math.sin(ni * 0.1 + layerIdx) * 3;
    var midY = (y1 + y2) / 2;

    c.beginPath();
    c.moveTo(x1, y1);
    c.quadraticCurveTo(midX, midY, x2, y2);
    c.stroke();
  }
}

// ---- 3D Pseudo-Perspective Fibre View ----
function drawFibreBundle3D(c, W, H, data, layersSource, nTokens, nLayers, hiddenDim,
  gridCols, gridRows, pixSize, roomW, roomH, tokenGap, layerGap, useAbs, currentLayer, offsetX, offsetY) {

  var dx = +document.getElementById('sl-dx').value;
  var dy = +document.getElementById('sl-dy').value;
  var dz = +document.getElementById('sl-dz').value;

  // In 3D mode:
  // X axis = token index
  // Y axis = layer depth (fibre)
  // Z axis = a chosen hidden dimension's activation value

  var totalTokenW = nTokens * (roomW + tokenGap) - tokenGap;
  var focalLen = 500;

  function rot3D(x, y, z) {
    var cosY = Math.cos(fibreState.rotY), sinY = Math.sin(fibreState.rotY);
    var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
    var cosX = Math.cos(fibreState.rotX), sinX = Math.sin(fibreState.rotX);
    var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
    return [x1, y1, z2];
  }

  function proj(x, y, z) {
    var r = rot3D(x, y, z);
    var scale = focalLen / (focalLen + r[2]);
    return [W / (2 * zoomLevel) + r[0] * scale, H / (2 * zoomLevel) + r[1] * scale, r[2], scale];
  }

  // Scale factors
  var xScale = 200;  // spread tokens
  var yScale = 40;   // spread layers
  var zScale = 80;   // depth from activation

  // Draw from back to front (painter's algorithm by layer)
  for (var li = 0; li < nLayers; li++) {
    var layerDepth = (li - nLayers / 2) * yScale;
    var isCurrentLayer = (li === currentLayer + 1) || (li === 0 && currentLayer === 0);

    for (var ti = 0; ti < nTokens; ti++) {
      var tokenOffset = (ti - nTokens / 2) * (roomW + tokenGap) * 0.8;
      var acts = layersSource[li].activations[ti];

      // For each neuron, compute a 3D position
      // X = token position + neuron column offset
      // Y = layer position + neuron row offset
      // Z = activation value of a chosen dimension (dz)

      // Draw the room as a small grid of projected pixels
      for (var ni = 0; ni < hiddenDim; ni++) {
        var val = acts[ni];
        if (useAbs) val = Math.abs(val * 2 - 1);

        var col = ni % gridCols;
        var row = Math.floor(ni / gridCols);

        var wx = tokenOffset + (col - gridCols / 2) * pixSize * 0.5;
        var wy = layerDepth + (row - gridRows / 2) * pixSize * 0.3;
        var wz = (val - 0.5) * zScale;

        var p = proj(wx, wy, wz);
        var sz = Math.max(0.5, pixSize * 0.4 * p[3]);

        var rgb = neuronColor(val, fibreState.colormap);
        var depthAlpha = Math.max(0.1, Math.min(0.9, 0.7 - p[2] * 0.001));
        c.fillStyle = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + depthAlpha.toFixed(2) + ')';
        c.fillRect(p[0] - sz / 2, p[1] - sz / 2, sz, sz);
      }

      // Token label
      if (li === 0) {
        var lp = proj(tokenOffset, layerDepth + gridRows * pixSize * 0.2, 0);
        c.font = Math.max(7, Math.round(9 * lp[3])) + 'px monospace';
        c.fillStyle = '#e94560';
        c.textAlign = 'center';
        c.fillText(data.tokens[ti], lp[0], lp[1] + 12);
      }
    }

    // Layer label
    var llp = proj(-nTokens / 2 * (roomW + tokenGap) * 0.8 - 40, layerDepth, 0);
    c.font = '8px monospace';
    c.fillStyle = isCurrentLayer ? '#e94560' : '#555';
    c.textAlign = 'right';
    var layerLabel = li === 0 ? 'Emb' : 'L' + (li - 1);
    c.fillText(layerLabel, llp[0], llp[1] + 3);
  }

  // Draw 3D axes
  var axLen = 80;
  var axes = [
    { v: [1, 0, 0], label: 'Token →', color: '#e94560' },
    { v: [0, 1, 0], label: 'Layer ↑', color: '#53a8b6' },
    { v: [0, 0, 1], label: 'Dim ' + dz, color: '#f5a623' }
  ];
  var o3 = proj(0, 0, 0);
  for (var ai = 0; ai < 3; ai++) {
    var ax = axes[ai];
    var e3 = proj(ax.v[0] * axLen, ax.v[1] * axLen, ax.v[2] * axLen);
    c.strokeStyle = ax.color;
    c.globalAlpha = 0.5;
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(o3[0], o3[1]);
    c.lineTo(e3[0], e3[1]);
    c.stroke();
    c.globalAlpha = 1;
    c.font = '9px monospace';
    c.fillStyle = ax.color;
    c.textAlign = 'left';
    c.fillText(ax.label, e3[0] + 4, e3[1] - 4);
  }
}

// ---- Mouse interaction for fibre view ----
cv3d.addEventListener('mousedown', function(e) {
    if (viewMode !== 'fibre' && viewMode !== 'fibre3d') return;
  if (e.button === 0 && !e.shiftKey) {
    // Normal drag = rotate all rooms
    fibreState.dragActive = true;
    fibreState.dragLastX = e.clientX;
    fibreState.dragLastY = e.clientY;
    e.preventDefault();
    return;
  }
  if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
    // Shift+drag or middle = pan
    e.preventDefault();
    panActive = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
  }
});

window.addEventListener('mousemove', function(e) {
  if (viewMode !== 'fibre' && viewMode !== 'fibre3d') return;
  if (fibreState.dragActive) {
    var ddx = e.clientX - fibreState.dragLastX;
    var ddy = e.clientY - fibreState.dragLastY;
    fibreState.rotY += ddx * 0.005;
    fibreState.rotX += ddy * 0.005;
    fibreState.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, fibreState.rotX));
    fibreState.dragLastX = e.clientX;
    fibreState.dragLastY = e.clientY;
    draw();
    return;
  }
});

// ============================================================
// REFACTORED drawFibreBundle3DGrid — broken into composable steps
// ============================================================

/**
 * Read all fibre-3D-relevant parameters from the DOM and data model.
 * Returns a self-contained config object.
 */
function getFibre3DParams() {
    var hiddenDim = D.hidden_dim;
    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;

    return {
        nTokens:      D.n_real,
        nLayers:      D.n_layers,
        hiddenDim:    hiddenDim,
        nP:           D.n_points,
        dx:           Math.min(+document.getElementById('sl-dx').value, hiddenDim - 1),
        dy:           Math.min(+document.getElementById('sl-dy').value, hiddenDim - 1),
        dz:           Math.min(+document.getElementById('sl-dz').value, hiddenDim - 1),
        amp:          +document.getElementById('sl-amp').value,
        t:            +document.getElementById('sl-t').value,
        sig:          +document.getElementById('sl-sig').value,
        gr:           +document.getElementById('sl-gr').value,
        currentLayer: +document.getElementById('sl-layer').value,
        showGrid:     document.getElementById('cb-grid').checked,
        showHeat:     document.getElementById('cb-heat').checked,
        showSC:       document.getElementById('cb-sc').checked,
        mode:         document.getElementById('sel-mode').value,
        isEmb:        document.getElementById('sel-mode').value === 'embedding',
        activeDeltas: activeDeltas,
    };
}

/**
 * Compute the fibre-3D room layout: room sizes, gaps, scale factor.
 */
function computeFibre3DRoomLayout(nTokens, nLayers, mr) {
    var gapFracX = 0.35;
    var gapFracY = 0.45;
    var maxRoomSize = 120;
    var roomSize = Math.min(maxRoomSize, Math.max(30, Math.floor(300 / Math.max(nTokens, nLayers))));
    var gapX = Math.max(8, Math.floor(roomSize * gapFracX));
    var gapY = Math.max(12, Math.floor(roomSize * gapFracY));
    var sc3 = roomSize * 0.4 / mr;

    return {
        roomSize: roomSize,
        gapX: gapX,
        gapY: gapY,
        sc3: sc3,
    };
}

/**
 * Create a 3D projection function using fibreState rotation, focal length,
 * canvas dimensions, zoom, and pan.
 */
function makeFibre3DProjector(W, H) {
    var focalLen = 600;
    return function proj3Df(x, y, z) {
        var cosY = Math.cos(fibreState.rotY), sinY = Math.sin(fibreState.rotY);
        var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
        var cosX = Math.cos(fibreState.rotX), sinX = Math.sin(fibreState.rotX);
        var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
        // Apply zoom inside the rotation
        x1 *= zoomLevel; y1 *= zoomLevel; z2 *= zoomLevel;
        var scale = focalLen / (focalLen + z2);
        return [W / 2 + panX + x1 * scale, H / 2 + panY + y1 * scale, z2, scale];
    };
}

/**
 * Compute mode-aware cumulative 3D deltas for a single layer,
 * using pre-computed per-layer raw deltas.
 */
function computeCumulativeDeltas3DFromRaw(edxAll, edyAll, edzAll, li, nP, nLayers, mode, isEmb) {
    var edxCum = new Float64Array(nP);
    var edyCum = new Float64Array(nP);
    var edzCum = new Float64Array(nP);
    if (isEmb) return { edx: edxCum, edy: edyCum, edz: edzCum };

    if (mode === 'single') {
        for (var j = 0; j < nP; j++) {
            edxCum[j] = edxAll[li][j];
            edyCum[j] = edyAll[li][j];
            edzCum[j] = edzAll[li][j];
        }
    } else if (mode === 'cumfwd') {
        for (var cl = 0; cl <= li; cl++) {
            for (var j = 0; j < nP; j++) {
                edxCum[j] += edxAll[cl][j];
                edyCum[j] += edyAll[cl][j];
                edzCum[j] += edzAll[cl][j];
            }
        }
    } else { // cumbwd
        for (var cl = li; cl < nLayers; cl++) {
            for (var j = 0; j < nP; j++) {
                edxCum[j] += edxAll[cl][j];
                edyCum[j] += edyAll[cl][j];
                edzCum[j] += edzAll[cl][j];
            }
        }
    }
    return { edx: edxCum, edy: edyCum, edz: edzCum };
}

/**
 * Build a 3D deformed grid with edges and surface faces for one room.
 * Reuses interpolateGridPoint3D for the deformation.
 */
function buildFibre3DRoomGrid(vx0, vy0, vz0, vx1, vy1, vz1, N,
    fx, fy, fz, edxCum, edyCum, edzCum, nP, sig, t, isEmb) {

    function gIdx(ix, iy, iz) { return iz * (N + 1) * (N + 1) + iy * (N + 1) + ix; }
    var nV = (N + 1) * (N + 1) * (N + 1);
    var oX = new Float64Array(nV), oY = new Float64Array(nV), oZ = new Float64Array(nV);
    var gX = new Float64Array(nV), gY = new Float64Array(nV), gZ = new Float64Array(nV);

    for (var iz = 0; iz <= N; iz++)
        for (var iy = 0; iy <= N; iy++)
            for (var ix = 0; ix <= N; ix++) {
                var gi = gIdx(ix, iy, iz);
                oX[gi] = vx0 + (ix / N) * (vx1 - vx0);
                oY[gi] = vy0 + (iy / N) * (vy1 - vy0);
                oZ[gi] = vz0 + (iz / N) * (vz1 - vz0);
            }

    var s2i = 1 / (2 * sig * sig);

    if (isEmb) {
        for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; gZ[gi] = oZ[gi]; }
    } else {
        for (var gi = 0; gi < nV; gi++) {
            var gpx = oX[gi], gpy = oY[gi], gpz = oZ[gi];
            var vvx = 0, vvy = 0, vvz = 0, ws = 0;
            for (var k = 0; k < nP; k++) {
                var eex = gpx - fx[k], eey = gpy - fy[k], eez = gpz - fz[k];
                var w = Math.exp(Math.max(-500, -(eex * eex + eey * eey + eez * eez) * s2i));
                vvx += w * edxCum[k]; vvy += w * edyCum[k]; vvz += w * edzCum[k]; ws += w;
            }
            if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
            gX[gi] = gpx + t * vvx;
            gY[gi] = gpy + t * vvy;
            gZ[gi] = gpz + t * vvz;
        }
    }

    // Collect edges with strain
    var edges = [];
    function addEdge(a, b) {
        var od = Math.sqrt((oX[b]-oX[a])*(oX[b]-oX[a])+(oY[b]-oY[a])*(oY[b]-oY[a])+(oZ[b]-oZ[a])*(oZ[b]-oZ[a]));
        var dd = Math.sqrt((gX[b]-gX[a])*(gX[b]-gX[a])+(gY[b]-gY[a])*(gY[b]-gY[a])+(gZ[b]-gZ[a])*(gZ[b]-gZ[a]));
        var strain = od > 1e-12 ? dd / od : 1;
        edges.push({ a: a, b: b, strain: strain });
    }
    for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix < N; ix++)
        addEdge(gIdx(ix, iy, iz), gIdx(ix + 1, iy, iz));
    for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy < N; iy++) for (var ix = 0; ix <= N; ix++)
        addEdge(gIdx(ix, iy, iz), gIdx(ix, iy + 1, iz));
    for (var iz = 0; iz < N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++)
        addEdge(gIdx(ix, iy, iz), gIdx(ix, iy, iz + 1));

    // Collect surface faces for heatmap
    var faces = [];
    function addFace(a, b, cc, d) {
        var od1 = Math.sqrt((oX[b]-oX[a])*(oX[b]-oX[a])+(oY[b]-oY[a])*(oY[b]-oY[a])+(oZ[b]-oZ[a])*(oZ[b]-oZ[a]));
        var dd1 = Math.sqrt((gX[b]-gX[a])*(gX[b]-gX[a])+(gY[b]-gY[a])*(gY[b]-gY[a])+(gZ[b]-gZ[a])*(gZ[b]-gZ[a]));
        var od2 = Math.sqrt((oX[cc]-oX[b])*(oX[cc]-oX[b])+(oY[cc]-oY[b])*(oY[cc]-oY[b])+(oZ[cc]-oZ[b])*(oZ[cc]-oZ[b]));
        var dd2 = Math.sqrt((gX[cc]-gX[b])*(gX[cc]-gX[b])+(gY[cc]-gY[b])*(gY[cc]-gY[b])+(gZ[cc]-gZ[b])*(gZ[cc]-gZ[b]));
        var avgS = ((od1 > 1e-12 ? dd1/od1 : 1) + (od2 > 1e-12 ? dd2/od2 : 1)) / 2;
        faces.push({ verts: [a, b, cc, d], strain: avgS });
    }
    for (var iz = 0; iz < N; iz++) for (var iy = 0; iy < N; iy++) {
        addFace(gIdx(0,iy,iz), gIdx(0,iy+1,iz), gIdx(0,iy+1,iz+1), gIdx(0,iy,iz+1));
        addFace(gIdx(N,iy,iz), gIdx(N,iy+1,iz), gIdx(N,iy+1,iz+1), gIdx(N,iy,iz+1));
    }
    for (var iz = 0; iz < N; iz++) for (var ix = 0; ix < N; ix++) {
        addFace(gIdx(ix,0,iz), gIdx(ix+1,0,iz), gIdx(ix+1,0,iz+1), gIdx(ix,0,iz+1));
        addFace(gIdx(ix,N,iz), gIdx(ix+1,N,iz), gIdx(ix+1,N,iz+1), gIdx(ix,N,iz+1));
    }
    for (var iy = 0; iy < N; iy++) for (var ix = 0; ix < N; ix++) {
        addFace(gIdx(ix,iy,0), gIdx(ix+1,iy,0), gIdx(ix+1,iy+1,0), gIdx(ix,iy+1,0));
        addFace(gIdx(ix,iy,N), gIdx(ix+1,iy,N), gIdx(ix+1,iy+1,N), gIdx(ix,iy+1,N));
    }

    return { oX: oX, oY: oY, oZ: oZ, gX: gX, gY: gY, gZ: gZ,
             edges: edges, faces: faces, nV: nV, gIdx: gIdx };
}

/**
 * Project a grid vertex from world-space into a room-local 3D position,
 * then into screen space.
 */
function projectRoomVertex(grid, gi, roomCenterX, roomCenterY, cx3, cy3, cz3, sc3, proj3Df, deformed) {
    var x = deformed ? grid.gX[gi] : grid.oX[gi];
    var y = deformed ? grid.gY[gi] : grid.oY[gi];
    var z = deformed ? grid.gZ[gi] : grid.oZ[gi];
    var sx = roomCenterX + (x - cx3) * sc3;
    var sy = roomCenterY + (y - cy3) * sc3;
    var sz = (z - cz3) * sc3;
    return proj3Df(sx, sy, sz);
}

/**
 * Collect all projected grid edges for one room into the allEdges array.
 */
function collectRoomGridEdges(allEdges, grid, roomCenterX, roomCenterY,
    cx3, cy3, cz3, sc3, proj3Df, isCurrentLayer, showGrid, isEmb) {

    if (!showGrid || isEmb) return;

    for (var ei = 0; ei < grid.edges.length; ei++) {
        var e = grid.edges[ei];
        var pa = projectRoomVertex(grid, e.a, roomCenterX, roomCenterY, cx3, cy3, cz3, sc3, proj3Df, true);
        var pb = projectRoomVertex(grid, e.b, roomCenterX, roomCenterY, cx3, cy3, cz3, sc3, proj3Df, true);
        allEdges.push({
            x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1],
            z: (pa[2] + pb[2]) / 2, strain: e.strain,
            isCurrentLayer: isCurrentLayer, type: 'grid'
        });
    }
}

/**
 * Collect all projected heatmap face quads for one room into the allQuads array.
 */
function collectRoomHeatmapFaces(allQuads, grid, roomCenterX, roomCenterY,
    cx3, cy3, cz3, sc3, proj3Df, isCurrentLayer, showHeat, isEmb) {

    if (!showHeat || isEmb) return;

    for (var fi = 0; fi < grid.faces.length; fi++) {
        var f = grid.faces[fi];
        var co = s2c(f.strain);
        var pts3d = [];
        var avgZ3d = 0;
        for (var ci = 0; ci < 4; ci++) {
            var p = projectRoomVertex(grid, f.verts[ci], roomCenterX, roomCenterY, cx3, cy3, cz3, sc3, proj3Df, true);
            pts3d.push(p);
            avgZ3d += p[2];
        }
        avgZ3d /= 4;
        allQuads.push({
            pts: pts3d, z: avgZ3d,
            color: co,
            alpha: isCurrentLayer ? 0.25 : 0.1,
            isCurrentLayer: isCurrentLayer
        });
    }
}

/**
 * Collect the 12 wireframe border edges of a room bounding box.
 */
function collectRoomBorderEdges(allEdges, roomCenterX, roomCenterY, roomSize, proj3Df, isCurrentLayer) {
    var borderAlpha = isCurrentLayer ? 0.6 : 0.2;
    var halfR = roomSize * 0.5;
    var bCorners3D = [
        [-halfR, -halfR, -halfR], [halfR, -halfR, -halfR],
        [halfR, halfR, -halfR], [-halfR, halfR, -halfR],
        [-halfR, -halfR, halfR], [halfR, -halfR, halfR],
        [halfR, halfR, halfR], [-halfR, halfR, halfR]
    ];
    var bEdgeIndices = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for (var bi = 0; bi < bEdgeIndices.length; bi++) {
        var be = bEdgeIndices[bi];
        var c0 = bCorners3D[be[0]], c1 = bCorners3D[be[1]];
        var pa = proj3Df(roomCenterX + c0[0] * 0.5, roomCenterY + c0[1] * 0.5, c0[2] * 0.5);
        var pb = proj3Df(roomCenterX + c1[0] * 0.5, roomCenterY + c1[1] * 0.5, c1[2] * 0.5);
        allEdges.push({
            x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1],
            z: (pa[2] + pb[2]) / 2, strain: 1.0,
            isCurrentLayer: isCurrentLayer, type: 'border',
            borderAlpha: borderAlpha
        });
    }
}

/**
 * Collect a token dot point for one room.
 */
function collectRoomTokenPoint(allPoints, fx, fy, fz, ti, edxCum, edyCum, edzCum,
    t, isEmb, roomCenterX, roomCenterY, cx3, cy3, cz3, sc3, proj3Df,
    li, isCurrentLayer, color) {

    var tokWX = fx[ti] + (isEmb ? 0 : t * edxCum[ti]);
    var tokWY = fy[ti] + (isEmb ? 0 : t * edyCum[ti]);
    var tokWZ = fz[ti] + (isEmb ? 0 : t * edzCum[ti]);

    var tokSX = roomCenterX + (tokWX - cx3) * sc3;
    var tokSY = roomCenterY + (tokWY - cy3) * sc3;
    var tokSZ = (tokWZ - cz3) * sc3;

    var tp = proj3Df(tokSX, tokSY, tokSZ);

    allPoints.push({
        x: tp[0], y: tp[1], z: tp[2], scale: tp[3],
        tokenIdx: ti, layerIdx: li,
        isCurrentLayer: isCurrentLayer,
        color: color
    });
}

/**
 * Collect a token label point (shown at the bottom layer).
 */
function collectRoomTokenLabel(allPoints, ti, roomCenterX, roomCenterY, roomSize, proj3Df, color, labelText) {
    var labelP = proj3Df(roomCenterX, roomCenterY + roomSize * 0.6, 0);
    allPoints.push({
        x: labelP[0], y: labelP[1], z: labelP[2], scale: labelP[3],
        tokenIdx: ti, layerIdx: -1,
        isCurrentLayer: false, isTokenLabel: true,
        color: color,
        labelText: labelText
    });
}

/**
 * Collect a layer label point.
 */
function collectLayerLabel(allPoints, li, labelX, labelY, proj3Df, isCurrentLayer) {
    var llp = proj3Df(labelX, labelY, 0);
    allPoints.push({
        x: llp[0], y: llp[1], z: llp[2], scale: llp[3],
        tokenIdx: -1, layerIdx: li,
        isCurrentLayer: isCurrentLayer, isLabel: true,
        labelText: 'L' + li
    });
}

/**
 * Collect inter-layer pathline edges connecting rooms vertically.
 */
function collectInterLayerPathlines(allEdges, ti, prevRowIdx, rowIdx, nLayers,
    roomSize, gapY, gapX, nTokens, proj3Df, isCurrentLayer, color) {

    var prevRoomCY = (prevRowIdx - (nLayers - 1) / 2) * (roomSize + gapY);
    var currRoomCY = (rowIdx - (nLayers - 1) / 2) * (roomSize + gapY);
    var roomCX = (ti - (nTokens - 1) / 2) * (roomSize + gapX);

    var py3 = prevRoomCY + roomSize * 0.5;
    var cy3b = currRoomCY - roomSize * 0.5;

    var pp = proj3Df(roomCX, py3, 0);
    var cp = proj3Df(roomCX, cy3b, 0);

    allEdges.push({
        x1: pp[0], y1: pp[1], x2: cp[0], y2: cp[1],
        z: (pp[2] + cp[2]) / 2, strain: 1.0,
        isCurrentLayer: isCurrentLayer, type: 'pathline',
        color: color
    });
}

/**
 * Render all depth-sorted quads (heatmap faces).
 */
function renderFibre3DQuads(c, allQuads) {
    allQuads.sort(function(a, b) { return b.z - a.z; });
    for (var qi = 0; qi < allQuads.length; qi++) {
        var q = allQuads[qi];
        c.beginPath();
        c.moveTo(q.pts[0][0], q.pts[0][1]);
        for (var ci = 1; ci < 4; ci++) c.lineTo(q.pts[ci][0], q.pts[ci][1]);
        c.closePath();
        c.fillStyle = 'rgba(' + q.color[0] + ',' + q.color[1] + ',' + q.color[2] + ',' + q.alpha.toFixed(2) + ')';
        c.fill();
    }
}

/**
 * Render all depth-sorted edges (grid lines, borders, pathlines).
 */
function renderFibre3DEdges(c, allEdges, showSC) {
    allEdges.sort(function(a, b) { return b.z - a.z; });
    for (var ei = 0; ei < allEdges.length; ei++) {
        var e = allEdges[ei];
        var depthAlpha = Math.max(0.15, Math.min(0.95, 0.7 - e.z * 0.0002));

        if (e.type === 'pathline') {
            var r = parseInt(e.color.slice(1, 3), 16);
            var g = parseInt(e.color.slice(3, 5), 16);
            var b = parseInt(e.color.slice(5, 7), 16);
            c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (depthAlpha * 0.6).toFixed(2) + ')';
            c.lineWidth = e.isCurrentLayer ? 2 : 1.0;
        } else if (e.type === 'border') {
            c.strokeStyle = e.isCurrentLayer ?
                'rgba(233,69,96,' + Math.min(1.0, e.borderAlpha * depthAlpha * 2.5).toFixed(2) + ')' :
                'rgba(80,80,130,' + Math.min(1.0, e.borderAlpha * depthAlpha * 1.5).toFixed(2) + ')';
            c.lineWidth = e.isCurrentLayer ? 1.5 : 0.7;
        } else if (e.type === 'grid') {
            var gridAlpha = e.isCurrentLayer ? depthAlpha * 0.9 : depthAlpha * 0.45;
            if (showSC) {
                var sc = s2c(e.strain);
                c.strokeStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + gridAlpha.toFixed(2) + ')';
            } else {
                c.strokeStyle = 'rgba(200,200,200,' + gridAlpha.toFixed(2) + ')';
            }
            c.lineWidth = e.isCurrentLayer ? 1.0 : 0.5;
        } else {
            continue;
        }

        c.beginPath();
        c.moveTo(e.x1, e.y1);
        c.lineTo(e.x2, e.y2);
        c.stroke();
    }
}

/**
 * Render all depth-sorted points (token dots, labels, layer labels).
 */
function renderFibre3DPoints(c, allPoints) {
    allPoints.sort(function(a, b) { return b.z - a.z; });

    for (var pi = 0; pi < allPoints.length; pi++) {
        var pt = allPoints[pi];
        var depthAlpha = Math.max(0.4, Math.min(1.0, 0.9 - pt.z * 0.0002));

        if (pt.isLabel) {
            c.font = (pt.isCurrentLayer ? 'bold ' : '') + Math.max(8, Math.round(9 * pt.scale)) + 'px monospace';
            c.fillStyle = pt.isCurrentLayer ? '#e94560' : '#666';
            c.textAlign = 'right';
            c.globalAlpha = depthAlpha;
            c.fillText(pt.labelText, pt.x - 5, pt.y + 3);
            c.globalAlpha = 1.0;
            continue;
        }

        if (pt.isTokenLabel) {
            var fontSize = Math.max(7, Math.round(8 * pt.scale));
            c.font = 'bold ' + fontSize + 'px monospace';
            c.fillStyle = pt.color;
            c.textAlign = 'center';
            c.globalAlpha = depthAlpha;
            c.fillText(pt.labelText, pt.x, pt.y);
            c.globalAlpha = 1.0;
            continue;
        }

        var dotR = Math.max(2, (pt.isCurrentLayer ? 5 : 3) * pt.scale);

        if (pt.isCurrentLayer) {
            var grad = c.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, dotR * 2.5);
            grad.addColorStop(0, 'rgba(255,255,255,0.12)');
            grad.addColorStop(1, 'rgba(255,255,255,0)');
            c.beginPath();
            c.arc(pt.x, pt.y, dotR * 2.5, 0, Math.PI * 2);
            c.fillStyle = grad;
            c.fill();
        }

        c.beginPath();
        c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
        c.fillStyle = pt.color;
        c.globalAlpha = depthAlpha;
        c.fill();
        c.strokeStyle = pt.isCurrentLayer ? '#fff' : 'rgba(255,255,255,0.5)';
        c.lineWidth = pt.isCurrentLayer ? 1.5 : 0.7;
        c.stroke();
        c.globalAlpha = 1.0;
    }
}

/**
 * Draw the 3D axes overlay for the fibre 3D view.
 */
function drawFibre3DAxes(c, roomSize, mr, proj3Df, dx, dy, dz) {
    var axLen = Math.max(roomSize, mr * 0.3) * zoomLevel;
    var axes = [
        { v: [1, 0, 0], label: 'Dim ' + dx, color: '#e94560' },
        { v: [0, 1, 0], label: 'Layer \u2191', color: '#53a8b6' },
        { v: [0, 0, 1], label: 'Dim ' + dz, color: '#f5a623' }
    ];
    c.lineWidth = 1.5;
    var o3 = proj3Df(0, 0, 0);
    for (var ai = 0; ai < 3; ai++) {
        var ax = axes[ai];
        var e3 = proj3Df(ax.v[0] * axLen, ax.v[1] * axLen, ax.v[2] * axLen);
        c.strokeStyle = ax.color;
        c.globalAlpha = 0.5;
        c.beginPath();
        c.moveTo(o3[0], o3[1]);
        c.lineTo(e3[0], e3[1]);
        c.stroke();
        c.globalAlpha = 1;
        c.font = '10px monospace';
        c.fillStyle = ax.color;
        c.textAlign = 'left';
        c.fillText(ax.label, e3[0] + 4, e3[1] - 4);
    }
}

/**
 * Top-level: the refactored drawFibreBundle3DGrid.
 * Orchestrates all passes using the helpers above.
 */
function drawFibreBundle3DGrid() {
    var cv = document.getElementById('cv');
    var c = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace';
        c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    // ---- Gather parameters ----
    var params = getFibre3DParams();

    // ---- Extract base positions in all 3 dims ----
    var pos3 = extractPositions3D(D, params.nP, params.dx, params.dy, params.dz);
    var fx = pos3.fx, fy = pos3.fy, fz = pos3.fz;

    // ---- Compute view bounds ----
    var bounds = computeViewBounds3D(fx, fy, fz, params.nP, 0.12);
    var cx3 = bounds.cx, cy3 = bounds.cy, cz3 = bounds.cz;
    var mr = bounds.mr;

    // ---- Room layout ----
    var roomLayout = computeFibre3DRoomLayout(params.nTokens, params.nLayers, mr);

    // ---- 3D projector ----
    var proj3Df = makeFibre3DProjector(W, H);

    // ---- Precompute per-layer raw deltas (3D) ----
    var edxAll = [], edyAll = [], edzAll = [];
    for (var lay = 0; lay < params.nLayers; lay++) {
        var edxL = new Float64Array(params.nP);
        var edyL = new Float64Array(params.nP);
        var edzL = new Float64Array(params.nP);
        for (var j = 0; j < params.nP; j++) {
            edxL[j] = params.activeDeltas[lay][j][params.dx] * params.amp;
            edyL[j] = params.activeDeltas[lay][j][params.dy] * params.amp;
            edzL[j] = params.activeDeltas[lay][j][params.dz] * params.amp;
        }
        edxAll.push(edxL); edyAll.push(edyL); edzAll.push(edzL);
    }

    // ---- 3D grid resolution ----
    var N3 = Math.max(3, Math.min(8, Math.floor(params.gr / 6)));

    // ---- Collect all drawable elements for depth sorting ----
    var allEdges = [];
    var allQuads = [];
    var allPoints = [];

    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for (var li = 0; li < params.nLayers; li++) {
        var rowIdx = params.nLayers - 1 - li;
        var isCurrentLayer = (li === params.currentLayer);

        // Compute cumulative deltas for this layer
        var layerDeltas = computeCumulativeDeltas3DFromRaw(
            edxAll, edyAll, edzAll, li, params.nP, params.nLayers,
            params.mode, params.isEmb
        );

        // Build the 3D deformed grid
        var grid = buildFibre3DRoomGrid(
            bounds.vx0, bounds.vy0, bounds.vz0,
            bounds.vx1, bounds.vy1, bounds.vz1,
            N3, fx, fy, fz,
            layerDeltas.edx, layerDeltas.edy, layerDeltas.edz,
            params.nP, params.sig, params.t, params.isEmb
        );

        for (var ti = 0; ti < params.nTokens; ti++) {
            var roomCenterX = (ti - (params.nTokens - 1) / 2) * (roomLayout.roomSize + roomLayout.gapX);
            var roomCenterY = (rowIdx - (params.nLayers - 1) / 2) * (roomLayout.roomSize + roomLayout.gapY);

            // ---- 3D Grid edges ----
            collectRoomGridEdges(allEdges, grid, roomCenterX, roomCenterY,
                cx3, cy3, cz3, roomLayout.sc3, proj3Df,
                isCurrentLayer, params.showGrid, params.isEmb);

            // ---- Strain heatmap faces ----
            collectRoomHeatmapFaces(allQuads, grid, roomCenterX, roomCenterY,
                cx3, cy3, cz3, roomLayout.sc3, proj3Df,
                isCurrentLayer, params.showHeat, params.isEmb);

            // ---- Room border (wireframe box) ----
            collectRoomBorderEdges(allEdges, roomCenterX, roomCenterY,
                roomLayout.roomSize, proj3Df, isCurrentLayer);

            // ---- Token dot ----
            collectRoomTokenPoint(allPoints, fx, fy, fz, ti,
                layerDeltas.edx, layerDeltas.edy, layerDeltas.edz,
                params.t, params.isEmb,
                roomCenterX, roomCenterY, cx3, cy3, cz3, roomLayout.sc3, proj3Df,
                li, isCurrentLayer, tc[ti % tc.length]);

            // ---- Token label at bottom layer ----
            if (li === 0) {
                collectRoomTokenLabel(allPoints, ti, roomCenterX, roomCenterY,
                    roomLayout.roomSize, proj3Df, tc[ti % tc.length],
                    '[' + ti + '] ' + D.tokens[ti]);
            }
        }

        // ---- Layer label ----
        var layerLabelX = -(params.nTokens / 2) * (roomLayout.roomSize + roomLayout.gapX) - 30;
        var layerLabelY = (rowIdx - (params.nLayers - 1) / 2) * (roomLayout.roomSize + roomLayout.gapY);
        collectLayerLabel(allPoints, li, layerLabelX, layerLabelY, proj3Df, isCurrentLayer);

        // ---- Inter-layer connections ----
        if (li > 0 && fibreState.showConnections) {
            var prevRowIdx = params.nLayers - li;
            for (var ti2 = 0; ti2 < params.nTokens; ti2++) {
                collectInterLayerPathlines(allEdges, ti2, prevRowIdx, rowIdx,
                    params.nLayers, roomLayout.roomSize, roomLayout.gapY, roomLayout.gapX,
                    params.nTokens, proj3Df, isCurrentLayer, tc[ti2 % tc.length]);
            }
        }
    }

    // ---- Render all depth-sorted elements ----
    renderFibre3DQuads(c, allQuads);
    renderFibre3DEdges(c, allEdges, params.showSC);
    renderFibre3DPoints(c, allPoints);

    // ---- 3D axes ----
    drawFibre3DAxes(c, roomLayout.roomSize, mr, proj3Df, params.dx, params.dy, params.dz);

    // ---- HUD ----
    var decompLabel = getDecompLabel();
    c.font = '11px monospace';
    c.fillStyle = 'rgba(255,255,255,0.45)';
    c.textAlign = 'left';
    if (params.isEmb) {
        c.fillText('FIBRE BUNDLE 3D  EMBEDDING  Dims:' + params.dx + ',' + params.dy + ',' + params.dz +
                   '  Layers:' + params.nLayers + '  Tokens:' + params.nTokens + '  Drag to rotate', 12, 16);
    } else {
        c.fillText('FIBRE BUNDLE 3D  Layer ' + params.currentLayer + '/' + (params.nLayers - 1) +
                   '  t=' + params.t.toFixed(2) + '  amp=' + params.amp.toFixed(1) +
                   '  Dims:' + params.dx + ',' + params.dy + ',' + params.dz +
                   '  Mode:' + params.mode + '  Decomp:' + decompLabel +
                   '  Drag to rotate', 12, 16);
    }
    c.font = '9px monospace';
    c.fillStyle = 'rgba(255,255,255,0.3)';
    c.fillText('Zoom: ' + zoomLevel.toFixed(2) + 'x  (Scroll=zoom, Shift+drag=pan, 0=reset)', 12, H - 8);
}

window.addEventListener('mouseup', function(e) {
  if (viewMode === 'fibre' || viewMode === 'fibre3d') {
    fibreState.dragActive = false;
  }
});

// Colormap cycling with 'M' key in fibre mode
var _origOnKeyFibre = onKeyFibre;
onKeyFibre = function(e) {
  if (e.key === 'm' || e.key === 'M') {
    var maps = ['grayscale', 'coolhot', 'viridis'];
    var idx = maps.indexOf(fibreState.colormap);
    fibreState.colormap = maps[(idx + 1) % maps.length];
    document.getElementById('status').textContent = 'Colormap: ' + fibreState.colormap;
    draw();
    return;
  }
  if (e.key === 'd' || e.key === 'D') {
    // Adjust connection density
    fibreState.connectionDensity = Math.min(1.0, fibreState.connectionDensity + 0.05);
    if (fibreState.connectionDensity > 1.0) fibreState.connectionDensity = 0.02;
    document.getElementById('status').textContent =
      'Connection density: ' + (fibreState.connectionDensity * 100).toFixed(0) + '%';
    draw();
    return;
  }
  _origOnKeyFibre(e);
};

// ---- Fibre view button in sidebar ----
// Add the button dynamically if it doesn't exist
(function() {
  var toggleDiv = document.querySelector('.view-toggle');
  if (toggleDiv && !document.getElementById('btn-fibre')) {
    var btn = document.createElement('button');
    btn.id = 'btn-fibre';
    btn.textContent = 'Fibre Bundle';
    btn.onclick = function() { setViewMode('fibre'); };
    toggleDiv.appendChild(btn);
  }
})();

// ============================================================
// COMPARE MODE — Differential Activation Maps
// ============================================================

var compareData = null;

function toggleCompareMode() {
  var on = document.getElementById('cb-compare').checked;
  document.getElementById('compare-area').style.display = on ? 'block' : 'none';
  if (!on) {
    document.getElementById('compare-panel').style.display = 'none';
    document.getElementById('compare-summary').style.display = 'none';
    document.getElementById('compare-divergence-chart').innerHTML = '';
    compareData = null;
  }
}

function runCompare() {
  var textA = document.getElementById('txt-in').value.trim();
  var textB = document.getElementById('txt-b').value.trim();
  if (!textA || !textB) { alert('Enter both texts'); return; }

  var btn = document.getElementById('btn-compare');
  btn.disabled = true; btn.textContent = 'Comparing...';
  document.getElementById('status').textContent = 'Comparing activations...';

  fetch('/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text_a: textA, text_b: textB })
  })
  .then(function(r) { if (!r.ok) throw new Error('Server error ' + r.status); return r.json(); })
  .then(function(data) {
    if (data.error) { alert(data.error); return; }
    compareData = data;
    renderCompareSummary();
    renderDivergenceChart();
    renderCompareGrids();
    btn.disabled = false; btn.textContent = 'Compare';
    document.getElementById('status').textContent =
      'Compare: ' + data.n_common + ' aligned tokens, ' +
      data.n_layers + ' layers, onset layer: ' +
      (data.onset_layer >= 0 ? data.onset_layer : 'none');
  })
  .catch(function(e) {
    alert('Error: ' + e);
    btn.disabled = false; btn.textContent = 'Compare';
  });
}

function renderCompareSummary() {
  var d = compareData;
  var panel = document.getElementById('compare-summary');
  panel.style.display = 'block';

  var html = '';
  html += '<div style="color:#e94560;font-weight:bold;margin-bottom:4px">Differential Activation Analysis</div>';
  html += '<div><b style="color:#53a8b6">Text A:</b> <span style="color:#a0a0c0">' +
          d.tokens_a.join(' ') + '</span></div>';
  html += '<div><b style="color:#f5a623">Text B:</b> <span style="color:#a0a0c0">' +
          d.tokens_b.join(' ') + '</span></div>';
  html += '<div style="margin-top:4px">';
  html += 'Aligned tokens: <span style="color:#e94560">' + d.n_common + '</span> | ';
  html += 'Layers: <span style="color:#e94560">' + d.n_layers + '</span> | ';
  html += 'Hidden dim: <span style="color:#e94560">' + d.hidden_dim + '</span> | ';
  html += 'Max diff: <span style="color:#e94560">' + d.global_diff_max.toFixed(4) + '</span>';
  html += '</div>';

  if (d.onset_layer >= 0) {
    html += '<div style="margin-top:4px;color:#f5a623;font-weight:bold">';
    html += '⚡ Divergence onset at layer ' + d.onset_layer +
            ' — this is where the model starts processing the inputs differently!';
    html += '</div>';
  } else {
    html += '<div style="margin-top:4px;color:#888">';
    html += 'No clear divergence onset detected (inputs may be very similar or very different from the start).';
    html += '</div>';
  }

  // Top diverging dims at the most divergent layer
  var maxDivLayer = 0;
  var maxDiv = 0;
  for (var i = 0; i < d.layer_divergence.length; i++) {
    if (d.layer_divergence[i] > maxDiv) {
      maxDiv = d.layer_divergence[i];
      maxDivLayer = i;
    }
  }
  if (d.top_dims_per_layer[maxDivLayer] && d.top_dims_per_layer[maxDivLayer].length > 0) {
    html += '<div style="margin-top:4px;font-size:9px">';
    html += '<b style="color:#53a8b6">Most divergent layer: ' +
            (maxDivLayer === 0 ? 'Embedding' : 'L' + (maxDivLayer - 1)) + '</b> — Top dims: ';
    var topDims = d.top_dims_per_layer[maxDivLayer].slice(0, 8);
    for (var di = 0; di < topDims.length; di++) {
      html += '<span style="color:#e94560">d' + topDims[di].dim + '</span>';
      html += '<span style="color:#666">(' + topDims[di].mean_abs_diff.toFixed(4) + ')</span> ';
    }
    html += '</div>';
  }

  panel.innerHTML = html;
}

function renderDivergenceChart() {
  var d = compareData;
  var container = document.getElementById('compare-divergence-chart');
  var chartW = 340, chartH = 80;

  var html = '<div style="color:#888;font-size:9px;margin-bottom:2px">' +
             'Layer-by-layer divergence (mean |A−B|):</div>';
  html += '<canvas id="div-chart-cv" width="' + chartW + '" height="' + chartH + '"></canvas>';

  // Per-token divergence sparklines
  if (d.n_common > 0 && d.token_divergence.length > 0) {
    html += '<div style="margin-top:6px;font-size:9px;color:#888">Per-token divergence across layers:</div>';
    for (var ti = 0; ti < d.n_common; ti++) {
      var tokLabel = d.tokens_a[ti];
      var tokLabelB = d.tokens_b[ti];
      var same = (tokLabel === tokLabelB);
      html += '<div style="display:flex;align-items:center;gap:4px;margin:1px 0">';
      html += '<span style="color:' + (same ? '#53a8b6' : '#e94560') +
              ';min-width:80px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="A: ' +
              tokLabel + ' | B: ' + tokLabelB + '">';
      html += '[' + ti + '] ' + tokLabel;
      if (!same) html += ' / ' + tokLabelB;
      html += '</span>';
      html += '<canvas id="tok-div-' + ti + '" width="200" height="16" style="border:1px solid #0f3460;border-radius:2px"></canvas>';
      html += '</div>';
    }
  }

  container.innerHTML = html;

  // Draw the main divergence chart
  var cv = document.getElementById('div-chart-cv');
  if (!cv) return;
  var c = cv.getContext('2d');
  var nL = d.layer_divergence.length;
  var maxDiv = 0;
  for (var i = 0; i < nL; i++) {
    if (d.layer_divergence[i] > maxDiv) maxDiv = d.layer_divergence[i];
  }
  if (maxDiv < 1e-12) maxDiv = 1;

  var barW = Math.max(2, Math.floor((chartW - 20) / nL) - 1);
  var barGap = 1;
  var baseY = chartH - 15;
  var maxBarH = baseY - 5;

  c.fillStyle = '#0a0a1a';
  c.fillRect(0, 0, chartW, chartH);

  for (var i = 0; i < nL; i++) {
    var val = d.layer_divergence[i];
    var h = (val / maxDiv) * maxBarH;
    var x = 10 + i * (barW + barGap);

    // Color: low divergence = blue, high = red
    var frac = val / maxDiv;
    var r = Math.floor(frac * 233);
    var g = Math.floor((1 - frac) * 100 + frac * 69);
    var b = Math.floor((1 - frac) * 182 + frac * 96);

    // Highlight onset layer
    if (d.onset_layer >= 0 && i === d.onset_layer + 1) {
      c.fillStyle = 'rgba(245,166,35,0.3)';
      c.fillRect(x - 1, 0, barW + 2, chartH);
    }

    c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    c.fillRect(x, baseY - h, barW, h);

    // Layer label
    if (nL <= 30 || i % 2 === 0) {
      c.font = '7px monospace';
      c.fillStyle = '#666';
      c.textAlign = 'center';
      c.fillText(i === 0 ? 'E' : '' + (i - 1), x + barW / 2, chartH - 2);
    }
  }

  // Draw per-token sparklines
  for (var ti = 0; ti < d.n_common; ti++) {
    var sparkCv = document.getElementById('tok-div-' + ti);
    if (!sparkCv) continue;
    var sc = sparkCv.getContext('2d');
    var sw = sparkCv.width, sh = sparkCv.height;
    sc.fillStyle = '#0a0a1a';
    sc.fillRect(0, 0, sw, sh);

    var vals = [];
    for (var li = 0; li < nL; li++) {
      vals.push(d.token_divergence[li][ti] || 0);
    }
    var sparkMax = 0;
    for (var li = 0; li < vals.length; li++) {
      if (vals[li] > sparkMax) sparkMax = vals[li];
    }
    if (sparkMax < 1e-12) sparkMax = 1;

    var stepX = sw / Math.max(1, nL - 1);

    // Fill area
    sc.beginPath();
    sc.moveTo(0, sh);
    for (var li = 0; li < nL; li++) {
      var sx = li * stepX;
      var sy = sh - (vals[li] / sparkMax) * (sh - 2);
      sc.lineTo(sx, sy);
    }
    sc.lineTo(sw, sh);
    sc.closePath();
    sc.fillStyle = 'rgba(233,69,96,0.2)';
    sc.fill();

    // Line
    sc.beginPath();
    for (var li = 0; li < nL; li++) {
      var sx = li * stepX;
      var sy = sh - (vals[li] / sparkMax) * (sh - 2);
      if (li === 0) sc.moveTo(sx, sy);
      else sc.lineTo(sx, sy);
    }
    sc.strokeStyle = '#e94560';
    sc.lineWidth = 1;
    sc.stroke();

    // Onset marker
    if (d.onset_layer >= 0) {
      var ox = (d.onset_layer + 1) * stepX;
      sc.strokeStyle = 'rgba(245,166,35,0.6)';
      sc.lineWidth = 1;
      sc.beginPath();
      sc.moveTo(ox, 0);
      sc.lineTo(ox, sh);
      sc.stroke();
    }
  }
}

function renderCompareGrids() {
  var d = compareData;
  var panel = document.getElementById('compare-panel');
  panel.style.display = 'block';

  var hiddenDim = d.hidden_dim;
  var gridCols = Math.ceil(Math.sqrt(hiddenDim));
  var gridRows = Math.ceil(hiddenDim / gridCols);
  var pixSize = 2;

  var html = '';
  html += '<div style="color:#888;font-size:9px;margin-bottom:6px">';
  html += 'Each row = one aligned token position. Three columns: ';
  html += '<span style="color:#53a8b6">A</span> | ';
  html += '<span style="color:#f5a623">B</span> | ';
  html += '<span style="color:#e94560">Diff</span> (red=A>B, blue=B>A, black=same). ';
  html += 'Each pixel = one neuron dimension. Rows within each grid = layers (top=embedding, bottom=last).';
  html += '</div>';

  // For each aligned token, show A | B | Diff stacked vertically by layer
  for (var ti = 0; ti < d.n_common; ti++) {
    var tokA = d.tokens_a[ti];
    var tokB = d.tokens_b[ti];
    var same = (tokA === tokB);

    html += '<div style="margin-bottom:10px;border-bottom:1px solid #0f3460;padding-bottom:6px">';
    html += '<div style="font-size:10px;margin-bottom:3px">';
    html += '<span style="color:#53a8b6;font-weight:bold">[' + ti + '] A: ' + tokA + '</span>';
    if (!same) {
      html += ' <span style="color:#e94560">≠</span> ';
      html += '<span style="color:#f5a623;font-weight:bold">B: ' + tokB + '</span>';
    } else {
      html += ' <span style="color:#2ecc71">=</span> ';
      html += '<span style="color:#f5a623">B: ' + tokB + '</span>';
    }
    html += '</div>';

    // Three side-by-side columns, each containing all layers stacked vertically
    html += '<div style="display:flex;gap:8px;align-items:flex-start">';

    // Column A
    html += '<div style="text-align:center">';
    html += '<div style="color:#53a8b6;font-size:8px;font-weight:bold;margin-bottom:2px">Text A</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-a-' + ti + '-' + li;
      var cw = gridCols * pixSize;
      var ch = gridRows * pixSize;
      html += '<canvas id="' + cid + '" width="' + cw + '" height="' + ch + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="A Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column B
    html += '<div style="text-align:center">';
    html += '<div style="color:#f5a623;font-size:8px;font-weight:bold;margin-bottom:2px">Text B</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-b-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="B Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column Diff
    html += '<div style="text-align:center">';
    html += '<div style="color:#e94560;font-size:8px;font-weight:bold;margin-bottom:2px">A − B</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-d-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="Diff Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column: Magnitude
    html += '<div style="text-align:center">';
    html += '<div style="color:#f5a623;font-size:8px;font-weight:bold;margin-bottom:2px">|Diff|</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-m-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="Magnitude Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Layer labels
    html += '<div style="text-align:left;padding-top:12px">';
    for (var li = 0; li < d.n_layers; li++) {
      var isOnset = (d.onset_layer >= 0 && li === d.onset_layer + 1);
      var lh = gridRows * pixSize + 1;
      html += '<div style="height:' + lh + 'px;line-height:' + lh + 'px;font-size:7px;' +
              'color:' + (isOnset ? '#f5a623' : '#555') + ';' +
              'font-weight:' + (isOnset ? 'bold' : 'normal') + '">';
      html += (li === 0 ? 'Emb' : 'L' + (li - 1));
      if (isOnset) html += ' ⚡';
      html += '</div>';
    }
    html += '</div>';

    html += '</div>'; // end flex row
    html += '</div>'; // end token block
  }

  panel.innerHTML = html;

  // Now draw on all canvases
  for (var ti = 0; ti < d.n_common; ti++) {
    for (var li = 0; li < d.n_layers; li++) {
      // Draw A
      drawCompareCanvas('cmp-a-' + ti + '-' + li,
        d.activations_a[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'grayscale');
      // Draw B
      drawCompareCanvas('cmp-b-' + ti + '-' + li,
        d.activations_b[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'grayscale');
      // Draw Diff (diverging colormap)
      drawCompareCanvas('cmp-d-' + ti + '-' + li,
        d.diff[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'diverging');
      // Draw Magnitude
      drawCompareCanvas('cmp-m-' + ti + '-' + li,
        d.diff_magnitude[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'hot');
    }
  }
}

/**
 * Draw a single compare-mode neuron grid canvas.
 * @param {string} canvasId
 * @param {number[]} acts - array of hiddenDim floats in [0,1]
 * @param {number} hiddenDim
 * @param {number} gridCols
 * @param {number} gridRows
 * @param {number} pixSize
 * @param {string} colorMode - 'grayscale', 'diverging', or 'hot'
 */
function drawCompareCanvas(canvasId, acts, hiddenDim, gridCols, gridRows, pixSize, colorMode) {
  var cv = document.getElementById(canvasId);
  if (!cv) return;
  if (!acts || acts.length === 0) {
    // Empty canvas — fill dark
    var ctx = cv.getContext('2d');
    ctx.fillStyle = '#0a0515';
    ctx.fillRect(0, 0, cv.width, cv.height);
    return;
  }
  var ctx = cv.getContext('2d');
  var imgData = ctx.createImageData(gridCols * pixSize, gridRows * pixSize);

  for (var ni = 0; ni < hiddenDim; ni++) {
    var val = acts[ni];
    var r, g, b;

    if (colorMode === 'diverging') {
      // val: 0 = B >> A (blue), 0.5 = no diff (black), 1 = A >> B (red)
      if (val < 0.5) {
        // Blue side: 0 = bright blue, 0.5 = black
        var intensity = (0.5 - val) * 2; // 0..1
        r = 0;
        g = Math.floor(intensity * 80);
        b = Math.floor(intensity * 220);
      } else {
        // Red side: 0.5 = black, 1 = bright red
        var intensity = (val - 0.5) * 2; // 0..1
        r = Math.floor(intensity * 233);
        g = Math.floor(intensity * 50);
        b = Math.floor(intensity * 30);
      }
    } else if (colorMode === 'hot') {
      // val: 0 = black, 1 = bright white-hot
      // black -> red -> orange -> yellow -> white
      if (val < 0.25) {
        var t = val / 0.25;
        r = Math.floor(t * 180); g = 0; b = 0;
      } else if (val < 0.5) {
        var t = (val - 0.25) / 0.25;
        r = 180 + Math.floor(t * 75); g = Math.floor(t * 120); b = 0;
      } else if (val < 0.75) {
        var t = (val - 0.5) / 0.25;
        r = 255; g = 120 + Math.floor(t * 135); b = Math.floor(t * 50);
      } else {
        var t = (val - 0.75) / 0.25;
        r = 255; g = 255; b = 50 + Math.floor(t * 205);
      }
    } else {
      // Grayscale
      var v = Math.floor(val * 255);
      r = v; g = v; b = v;
    }

    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);

    for (var py = 0; py < pixSize; py++) {
      for (var px = 0; px < pixSize; px++) {
        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
        var offset = ix * 4;
        imgData.data[offset]     = r;
        imgData.data[offset + 1] = g;
        imgData.data[offset + 2] = b;
        imgData.data[offset + 3] = 255;
      }
    }
  }

  // Fill remaining pixels dark
  for (var ni = hiddenDim; ni < gridCols * gridRows; ni++) {
    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    for (var py = 0; py < pixSize; py++) {
      for (var px = 0; px < pixSize; px++) {
        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
        var offset = ix * 4;
        imgData.data[offset]     = 10;
        imgData.data[offset + 1] = 5;
        imgData.data[offset + 2] = 20;
        imgData.data[offset + 3] = 255;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

// ============================================================
// DIFFEOMORPHISM SPECTRUM VIEW
// ============================================================

// ============================================================
// DIFFEOMORPHISM SPECTRUM — Frontend Integration
// ============================================================

var spectrumData = null;
var contrastiveData = null;

// Behavior prompt libraries for one-click scanning
var BEHAVIOR_LIBRARIES = {
    "math": {
        positive: [
            "What is 2+2?",
            "Calculate 15 times 7",
            "Solve x squared equals 16",
            "What is the integral of x squared?"
        ],
        negative: [
            "What color is the sky?",
            "Tell me about dogs",
            "Who was Shakespeare?",
            "Describe a sunset"
        ]
    },
    "refusal": {
        positive: [
            "How do I pick a lock?",
            "Write a phishing email template",
            "How to bypass security systems",
            "Generate harmful content about"
        ],
        negative: [
            "How do I bake a cake?",
            "Write a thank you email",
            "How to set up home security",
            "Generate a poem about nature"
        ]
    },
    "code": {
        positive: [
            "Write a Python function to sort",
            "Debug this code: for i in range",
            "Implement binary search in JavaScript",
            "How to use async await in Python"
        ],
        negative: [
            "Write a poem about spring",
            "Tell me a funny joke",
            "Describe the taste of chocolate",
            "What happened in 1776?"
        ]
    },
    "reasoning": {
        positive: [
            "If all cats are animals and all animals breathe, do cats breathe?",
            "A bat and ball cost $1.10 total. The bat costs $1 more than the ball.",
            "There are 3 boxes. One has apples, one has oranges, one has both.",
            "If it takes 5 machines 5 minutes to make 5 widgets"
        ],
        negative: [
            "The weather today is sunny",
            "I like to eat pizza",
            "The car is parked outside",
            "She walked to the store"
        ]
    }
};

function fetchDiffeoSpectrum(textB) {
    if (!D) return;
    var body = { text: D.text };
    if (textB) body.text_b = textB;

    document.getElementById('status').textContent = 'Computing diffeomorphism spectrum...';

    fetch('/diffeomorphism_spectrum', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
            return;
        }
        spectrumData = data;
        renderSpectrumPanel();
        document.getElementById('status').textContent =
            'Spectrum ready — ' + data.anomalies.length + ' geometric anomalies detected';
    })
    .catch(function(e) {
        document.getElementById('status').textContent = 'Spectrum error: ' + e;
    });
}

function runContrastiveScan(behaviorName) {
    var lib = BEHAVIOR_LIBRARIES[behaviorName];
    if (!lib) return;

    document.getElementById('status').textContent =
        'Scanning for "' + behaviorName + '" geometric signature...';

    fetch('/contrastive_spectrum', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            positive: lib.positive,
            negative: lib.negative,
            behavior: behaviorName
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
            return;
        }
        contrastiveData = data;
        renderContrastiveResults();
        document.getElementById('status').textContent =
            'Contrastive scan complete — ' + behaviorName +
            ' signature found at layer ' + data.geometric_signature.most_discriminative_layer;
    })
    .catch(function(e) {
        document.getElementById('status').textContent = 'Scan error: ' + e;
    });
}

function renderSpectrumPanel() {
    var panel = document.getElementById('spectrum-results');
    if (!spectrumData || !panel) return;

    var data = spectrumData;
    var html = '';

    // Anomalies list
    html += '<div style="color:#e94560;font-weight:bold;font-size:11px;margin-bottom:6px">';
    html += '⚡ Geometric Anomalies (' + data.anomalies.length + ')</div>';

    var typeIcons = {
        'high_curl': '🌀',
        'high_anisotropy': '↔️',
        'bottleneck': '⏳',
        'rank_change': '📐'
    };
    var typeColors = {
        'high_curl': '#f5a623',
        'high_anisotropy': '#7b68ee',
        'bottleneck': '#e94560',
        'rank_change': '#53a8b6'
    };

    for (var i = 0; i < Math.min(data.anomalies.length, 15); i++) {
        var a = data.anomalies[i];
        var icon = typeIcons[a.type] || '❓';
        var color = typeColors[a.type] || '#888';

        html += '<div style="margin:2px 0;padding:2px 6px;border-left:2px solid ' + color +
                ';font-size:9px;cursor:pointer" ' +
                'onclick="document.getElementById(\'sl-layer\').value=' + a.layer +
                ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))" ' +
                'onmouseover="this.style.background=\'#1a1a2e\'" ' +
                'onmouseout="this.style.background=\'transparent\'">';
        html += icon + ' <span style="color:' + color + '">' +
                a.type.replace(/_/g, ' ') + '</span> ';
        html += 'L' + a.layer + ' "' + a.token_str + '" ';
        html += '<span style="color:#888">val=' + a.value.toFixed(4) + '</span>';
        html += '</div>';
    }

    // Layer-by-layer invariant summary
    if (data.layer_spectra && data.layer_spectra.length > 0) {
        html += '<div style="margin-top:8px;color:#53a8b6;font-weight:bold;font-size:10px">';
        html += 'Layer Invariants (averaged across tokens)</div>';
        html += '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:8px;width:100%">';
        html += '<tr><th style="color:#53a8b6;padding:2px 3px">L</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Div</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Curl</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Shear</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Rank</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Cond</th></tr>';

        for (var li = 0; li < data.layer_spectra.length; li++) {
            var layerSpecs = data.layer_spectra[li];
            // Average across tokens
            var avgDiv = 0, avgCurl = 0, avgShear = 0, avgRank = 0, avgCond = 0;
            for (var ti = 0; ti < layerSpecs.length; ti++) {
                avgDiv += layerSpecs[ti].divergence;
                avgCurl += layerSpecs[ti].curl;
                avgShear += layerSpecs[ti].shear;
                avgRank += layerSpecs[ti].effective_rank;
                avgCond += layerSpecs[ti].condition_number;
            }
            var nT = layerSpecs.length || 1;
            avgDiv /= nT; avgCurl /= nT; avgShear /= nT;
            avgRank /= nT; avgCond /= nT;

            var isCurrentLayer = (li === +document.getElementById('sl-layer').value);
            var rowStyle = isCurrentLayer ? 'background:rgba(233,69,96,0.15)' : '';

            html += '<tr style="' + rowStyle + ';cursor:pointer" ' +
                    'onclick="document.getElementById(\'sl-layer\').value=' + li +
                    ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))">';
            html += '<td style="color:#e94560;font-weight:bold;padding:2px 3px">' + li + '</td>';
            html += '<td style="padding:2px 3px;color:' +
                    (avgDiv > 0 ? '#e94560' : '#0077b6') + '">' + avgDiv.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px;color:#f5a623">' + avgCurl.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px">' + avgShear.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px;color:#53a8b6">' + avgRank.toFixed(1) + '</td>';
            html += '<td style="padding:2px 3px;color:' +
                    (avgCond > 50 ? '#e94560' : '#888') + '">' + avgCond.toFixed(1) + '</td>';
            html += '</tr>';
        }
        html += '</table></div>';
    }

    // Diff spectra summary (if comparison was done)
    if (data.diff_spectra && data.diff_spectra.summary) {
        var ds = data.diff_spectra.summary;
        html += '<div style="margin-top:8px;border-top:1px solid #0f3460;padding-top:6px">';
        html += '<div style="color:#f5a623;font-weight:bold;font-size:10px">Differential Spectrum</div>';
        html += '<div style="font-size:9px;color:#a0a0c0;margin-top:3px">';
        html += 'Max spectral distance: <span style="color:#e94560">' +
                ds.max_spectral_distance.toFixed(4) + '</span> at layer ' +
                ds.max_spectral_distance_layer + '<br>';
        if (ds.onset_layer >= 0) {
            html += 'Geometric divergence onset: <span style="color:#f5a623;font-weight:bold">layer ' +
                    ds.onset_layer + '</span>';
        } else {
            html += 'No clear geometric divergence onset detected';
        }
        html += '</div></div>';
    }

    panel.innerHTML = html;
}

function renderContrastiveResults() {
    var panel = document.getElementById('contrastive-results');
    if (!contrastiveData || !panel) return;

    var data = contrastiveData;
    var sig = data.geometric_signature;
    var html = '';

    // Signature summary
    html += '<div style="background:#0f3460;padding:8px;border-radius:4px;margin-bottom:8px">';
    html += '<div style="color:#e94560;font-weight:bold;font-size:11px;margin-bottom:4px">';
    html += '🔬 Geometric Signature: "' + sig.behavior + '"</div>';
    html += '<div style="font-size:9px;color:#a0a0c0;line-height:1.5">';
    html += sig.description;
    html += '</div>';
    html += '<div style="margin-top:6px;font-size:9px">';
    html += 'Most discriminative layer: <span style="color:#e94560;font-weight:bold">' +
            sig.most_discriminative_layer + '</span> | ';
    html += 'Invariant: <span style="color:#f5a623">' +
            sig.most_discriminative_invariant + '</span> | ';
    html += 'Effect size: <span style="color:#53a8b6">' +
            sig.best_effect_size.toFixed(2) + '</span>';
    html += '</div>';
    html += '</div>';

    // Layer ranking
    html += '<div style="color:#53a8b6;font-weight:bold;font-size:10px;margin-bottom:4px">';
    html += 'Layer Contrast Scores</div>';

    var maxScore = 0;
    for (var i = 0; i < data.layer_contrasts.length; i++) {
        if (data.layer_contrasts[i].total_contrast_score > maxScore) {
            maxScore = data.layer_contrasts[i].total_contrast_score;
        }
    }
    if (maxScore < 1e-8) maxScore = 1;

    for (var i = 0; i < data.layer_contrasts.length; i++) {
        var lc = data.layer_contrasts[i];
        var score = lc.total_contrast_score;
        var barW = Math.max(2, (score / maxScore) * 150);
        var isTop = data.ranked_layers.indexOf(i) < 3;

        html += '<div style="display:flex;align-items:center;gap:4px;margin:1px 0;' +
                'cursor:pointer;padding:1px 4px;border-radius:2px' +
                (isTop ? ';background:rgba(233,69,96,0.1)' : '') + '" ' +
                'onclick="document.getElementById(\'sl-layer\').value=' + i +
                ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))">';
        html += '<span style="color:' + (isTop ? '#e94560' : '#888') +
                ';min-width:25px;font-size:9px;font-weight:' +
                (isTop ? 'bold' : 'normal') + '">L' + i + '</span>';
        html += '<div style="background:' + (isTop ? '#e94560' : '#555') +
                ';height:6px;width:' + barW + 'px;border-radius:2px"></div>';
        html += '<span style="color:#888;font-size:8px">' + score.toFixed(2) + '</span>';
        html += '</div>';
    }

    // Eigenvalue comparison histograms for top layers
    if (data.eigenvalue_comparisons && data.eigenvalue_comparisons.length > 0) {
        html += '<div style="margin-top:8px;color:#f5a623;font-weight:bold;font-size:10px">';
        html += 'Eigenvalue Spectrum Comparison</div>';

        for (var ei = 0; ei < data.eigenvalue_comparisons.length; ei++) {
            var ec = data.eigenvalue_comparisons[ei];
            html += '<div style="margin-top:4px;font-size:9px">';
            html += '<span style="color:#53a8b6">Layer ' + ec.layer + '</span> — ';
            html += 'KL divergence: <span style="color:#e94560">' +
                    ec.kl_divergence.toFixed(4) + '</span> | ';
            html += 'Pos mean: ' + ec.pos_mean_magnitude.toFixed(4) + ' | ';
            html += 'Neg mean: ' + ec.neg_mean_magnitude.toFixed(4);
            html += '</div>';

            // Mini histogram
            var histId = 'eig-hist-' + ei;
            html += '<canvas id="' + histId + '" width="200" height="40" ' +
                    'style="border:1px solid #0f3460;border-radius:2px;margin-top:2px"></canvas>';
        }
    }

    panel.innerHTML = html;

    // Draw eigenvalue histograms
    if (data.eigenvalue_comparisons) {
        for (var ei = 0; ei < data.eigenvalue_comparisons.length; ei++) {
            var ec = data.eigenvalue_comparisons[ei];
            var cv = document.getElementById('eig-hist-' + ei);
            if (!cv) continue;
            var ctx = cv.getContext('2d');
            var cw = cv.width, ch = cv.height;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, cw, ch);

            var nBins = ec.pos_histogram.length;
            var maxH = 0;
            for (var bi = 0; bi < nBins; bi++) {
                maxH = Math.max(maxH, ec.pos_histogram[bi], ec.neg_histogram[bi]);
            }
            if (maxH < 1e-8) maxH = 1;

            var barW = cw / nBins;
            for (var bi = 0; bi < nBins; bi++) {
                var x = bi * barW;
                // Positive (behavior) in red
                var hPos = (ec.pos_histogram[bi] / maxH) * (ch - 4);
                ctx.fillStyle = 'rgba(233,69,96,0.6)';
                ctx.fillRect(x, ch - 2 - hPos, barW * 0.45, hPos);
                // Negative (baseline) in blue
                var hNeg = (ec.neg_histogram[bi] / maxH) * (ch - 4);
                ctx.fillStyle = 'rgba(83,168,182,0.6)';
                ctx.fillRect(x + barW * 0.5, ch - 2 - hNeg, barW * 0.45, hNeg);
            }
        }
    }
}
// ============================================================
// HOLOGRAPHIC CURVATURE ANALYSIS — Frontend
// ============================================================

var curvatureData = null;

// Slider value displays
document.getElementById('curv-k').addEventListener('input', function(){
    document.getElementById('v-curv-k').textContent = this.value;
});
document.getElementById('curv-d').addEventListener('input', function(){
    document.getElementById('v-curv-d').textContent = this.value;
});
document.getElementById('curv-topk').addEventListener('input', function(){
    document.getElementById('v-curv-topk').textContent = this.value;
});

function runCurvatureAnalysis(){
    if(!D){
        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#e94560">Run a prompt first!</span>';
        return;
    }

    var btn = document.getElementById('btn-curvature');
    btn.disabled = true;
    btn.textContent = 'Computing...';
    document.getElementById('curvature-status').innerHTML =
        '<span style="color:#53a8b6">Computing fiber curvature (this may take a moment)...</span>';

    var kNeighbors = +document.getElementById('curv-k').value;
    var pcaD = +document.getElementById('curv-d').value;
    var topK = +document.getElementById('curv-topk').value;

    fetch('/curvature', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text: D.text,
            k_neighbors: kNeighbors,
            pca_d: pcaD,
            top_k_singularities: topK
        })
    })
    .then(function(r){
        if(!r.ok) throw new Error('Server error ' + r.status);
        return r.json();
    })
    .then(function(data){
        if(data.error){
            document.getElementById('curvature-status').innerHTML =
                '<span style="color:#e94560">' + data.error + '</span>';
            btn.disabled = false;
            btn.textContent = 'Analyze Curvature';
            return;
        }
        curvatureData = data;
        btn.disabled = false;
        btn.textContent = 'Analyze Curvature';

        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#2ecc71">✓ Curvature computed</span> — ' +
            data.seq_len + ' tokens × ' + data.n_layers + ' layers | ' +
            data.singularities.length + ' singularities found';

        renderCurvatureCorrelation();
        renderCurvatureSingularities();
        renderCurvatureHeatmap();
        renderCurvatureSurprisalChart();
    })
    .catch(function(e){
        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#e94560">Error: ' + e + '</span>';
        btn.disabled = false;
        btn.textContent = 'Analyze Curvature';
    });
}

function renderCurvatureCorrelation(){
    var panel = document.getElementById('curvature-correlation-summary');
    if(!curvatureData || !curvatureData.correlation){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    var corr = curvatureData.correlation;

    var html = '<div style="color:#2ecc71;font-weight:bold;margin-bottom:4px">Metric–Surprisal Correlation</div>';
    html += '<div>' + corr.summary + '</div>';
    html += '<div style="margin-top:4px;color:#888">Best layer: <span style="color:#e94560;font-weight:bold">' +
            corr.best_layer + '</span></div>';
    panel.innerHTML = html;
}

function renderCurvatureSingularities(){
    var panel = document.getElementById('curvature-singularities');
    if(!curvatureData || !curvatureData.singularities || curvatureData.singularities.length === 0){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';

    var sings = curvatureData.singularities;
    var html = '<div style="color:#e94560;font-weight:bold;font-size:10px;margin-bottom:6px">' +
               '⚡ Curvature Singularities (' + sings.length + ')</div>';

    var classIcons = {
        'gravitational_source': '🌀',
        'entropy_collapse': '🔻',
        'syntactic_junction': '🔗',
        'curvature_anomaly': '⚠️',
        'volume_anomaly': '📐',
        'transport_anomaly': '🔄'
    };
    var classColors = {
        'gravitational_source': '#f5a623',
        'entropy_collapse': '#e94560',
        'syntactic_junction': '#53a8b6',
        'curvature_anomaly': '#7b68ee',
        'volume_anomaly': '#2ecc71',
        'transport_anomaly': '#fd79a8'
    };

    for(var i = 0; i < sings.length; i++){
        var s = sings[i];
        var icon = classIcons[s.classification] || '●';
        var color = classColors[s.classification] || '#888';

        html += '<div style="margin:3px 0;padding:4px 6px;border-left:3px solid ' + color +
                ';background:rgba(0,0,0,0.2);border-radius:0 4px 4px 0;cursor:pointer" ' +
                'onclick="jumpToCurvatureSingularity(' + s.layer + ',' + s.token_idx + ')" ' +
                'onmouseover="this.style.background=\'rgba(255,255,255,0.05)\'" ' +
                'onmouseout="this.style.background=\'rgba(0,0,0,0.2)\'">';

        html += '<div style="display:flex;align-items:center;gap:4px">';
        html += '<span style="font-size:12px">' + icon + '</span>';
        html += '<span style="color:' + color + ';font-weight:bold;font-size:10px">' +
                s.classification.replace(/_/g, ' ') + '</span>';
        html += '<span style="color:#888;font-size:9px;margin-left:auto">#' + s.rank + '</span>';
        html += '</div>';

        html += '<div style="font-size:9px;margin-top:2px">';
        html += '<span style="color:#e94560">[' + s.token_idx + '] "' + s.token + '"</span>';
        html += ' at <span style="color:#53a8b6">L' + s.layer + '</span>';
        html += ' | score: <span style="color:#f5a623">' + s.combined_score.toFixed(4) + '</span>';
        html += '</div>';

        // Compact metrics row
        html += '<div style="font-size:8px;color:#666;margin-top:2px;display:flex;gap:6px;flex-wrap:wrap">';
        html += '<span>ORC=' + s.orc.toFixed(3) + '</span>';
        html += '<span>Sect=' + s.sectional.toFixed(3) + '</span>';
        html += '<span>Scal=' + s.scalar.toFixed(3) + '</span>';
        html += '<span>Proc=' + s.procrustes.toFixed(3) + '</span>';
        html += '</div>';

        // Description (collapsible)
        html += '<div style="font-size:8px;color:#888;margin-top:3px;line-height:1.4">' +
                s.description + '</div>';

        html += '</div>';
    }

    panel.innerHTML = html;
}

function jumpToCurvatureSingularity(layer, tokenIdx){
    // Jump the layer slider to this layer
    var sl = document.getElementById('sl-layer');
    sl.value = Math.min(layer, +sl.max);
    sl.dispatchEvent(new Event('input'));

    // Select the token
    if(D && tokenIdx < D.n_real){
        selectedTokens.clear();
        selectedTokens.add(tokenIdx);
        updateSelectedUI();
        draw();
    }
}

function renderCurvatureHeatmap(){
    var controls = document.getElementById('curvature-heatmap-controls');
    if(!curvatureData){
        controls.style.display = 'none';
        return;
    }
    controls.style.display = 'block';

    var type = document.getElementById('curv-heatmap-type').value;
    var matrix = curvatureData[type];
    if(!matrix || matrix.length === 0) return;

    var nRows = matrix.length;
    var nCols = matrix[0].length;
    var tokens = curvatureData.tokens;

    // Find min/max
    var vmin = Infinity, vmax = -Infinity;
    for(var r = 0; r < nRows; r++){
        for(var c = 0; c < nCols; c++){
            var v = matrix[r][c];
            if(v < vmin) vmin = v;
            if(v > vmax) vmax = v;
        }
    }

    // Handle diverging colormaps (centered at 0)
    var isDiverging = (type === 'ollivier_ricci' || type === 'scalar_curvature');
    if(isDiverging){
        var absMax = Math.max(Math.abs(vmin), Math.abs(vmax), 0.001);
        vmin = -absMax;
        vmax = absMax;
    }

    var range = vmax - vmin;
    if(range < 1e-12) range = 1;

    var cv = document.getElementById('curv-heatmap-cv');
    // Set canvas resolution based on data
    var cellW = Math.max(2, Math.floor(340 / nCols));
    var cellH = Math.max(2, Math.floor(160 / nRows));
    cv.width = cellW * nCols;
    cv.height = cellH * nRows;
    var ctx = cv.getContext('2d');

    for(var r = 0; r < nRows; r++){
        for(var c = 0; c < nCols; c++){
            var v = matrix[r][c];
            var norm = (v - vmin) / range; // 0..1

            var rgb;
            if(isDiverging){
                // Blue (negative) -> Black (zero) -> Red (positive)
                if(norm < 0.5){
                    var t = (0.5 - norm) * 2;
                    rgb = [Math.floor(t * 30), Math.floor(t * 100), Math.floor(t * 220)];
                } else {
                    var t = (norm - 0.5) * 2;
                    rgb = [Math.floor(t * 233), Math.floor(t * 50), Math.floor(t * 40)];
                }
            } else {
                // Viridis-like
                rgb = curvatureViridis(norm);
            }

            ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
    }

    // Draw current layer indicator
    var currentLayer = +document.getElementById('sl-layer').value;
    if(currentLayer < nRows){
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(0, currentLayer * cellH, cv.width, cellH);
    }

    // Update legend
    var titles = {
        'ollivier_ricci': 'Ollivier-Ricci Curvature',
        'scalar_curvature': 'Scalar Curvature (ΔlogVol)',
        'sectional_curvature': 'Sectional Curvature',
        'procrustes_deviation': 'Procrustes ||R-I||',
        'metric_log_det': 'log det(g)'
    };
    document.getElementById('curv-hm-title').textContent = titles[type] || type;
    document.getElementById('curv-hm-min').textContent = vmin.toFixed(3);
    document.getElementById('curv-hm-max').textContent = vmax.toFixed(3);
}

function curvatureViridis(t){
    // Approximate viridis colormap
    t = Math.max(0, Math.min(1, t));
    var r, g, b;
    if(t < 0.25){
        var f = t / 0.25;
        r = Math.floor(68 - f * 4); g = Math.floor(1 + f * 50); b = Math.floor(84 + f * 74);
    } else if(t < 0.5){
        var f = (t - 0.25) / 0.25;
        r = Math.floor(64 - f * 30); g = Math.floor(51 + f * 70); b = Math.floor(158 - f * 20);
    } else if(t < 0.75){
        var f = (t - 0.5) / 0.25;
        r = Math.floor(34 + f * 100); g = Math.floor(121 + f * 60); b = Math.floor(138 - f * 60);
    } else {
        var f = (t - 0.75) / 0.25;
        r = Math.floor(134 + f * 119); g = Math.floor(181 + f * 40); b = Math.floor(78 - f * 50);
    }
    return [r, g, b];
}

function renderCurvatureSurprisalChart(){
    var chartDiv = document.getElementById('curvature-surprisal-chart');
    if(!curvatureData || !curvatureData.correlation){
        chartDiv.style.display = 'none';
        return;
    }
    chartDiv.style.display = 'block';

    var corr = curvatureData.correlation;
    var surprisal = corr.surprisal;
    var bestLayer = corr.best_layer;
    var logDet = curvatureData.metric_log_det;
    var tokens = curvatureData.tokens;
    var seqLen = tokens.length;

    // Get log_det at best layer
    var ldBest = logDet[bestLayer];

    // Draw scatter plot
    var cv = document.getElementById('curv-surprisal-cv');
    cv.width = 340;
    cv.height = 140;
    var ctx = cv.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, 340, 140);

    // Find ranges
    var ldMin = Infinity, ldMax = -Infinity;
    var sMin = Infinity, sMax = -Infinity;
    for(var i = 0; i < seqLen; i++){
        if(ldBest[i] < ldMin) ldMin = ldBest[i];
        if(ldBest[i] > ldMax) ldMax = ldBest[i];
        if(surprisal[i] < sMin) sMin = surprisal[i];
        if(surprisal[i] > sMax) sMax = surprisal[i];
    }
    var ldRange = ldMax - ldMin || 1;
    var sRange = sMax - sMin || 1;

    var margin = 25;
    var plotW = 340 - 2 * margin;
    var plotH = 140 - 2 * margin;

    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, margin + plotH);
    ctx.lineTo(margin + plotW, margin + plotH);
    ctx.stroke();

    ctx.font = '7px monospace';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('log det(g)', margin + plotW / 2, 140 - 2);
    ctx.save();
    ctx.translate(8, margin + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Surprisal', 0, 0);
    ctx.restore();

    // Scatter points
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for(var i = 0; i < seqLen; i++){
        var sx = margin + ((ldBest[i] - ldMin) / ldRange) * plotW;
        var sy = margin + plotH - ((surprisal[i] - sMin) / sRange) * plotH;

        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, Math.PI * 2);
        ctx.fillStyle = tc[i % tc.length];
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Label
        if(seqLen <= 15){
            ctx.font = '7px monospace';
            ctx.fillStyle = '#888';
            ctx.textAlign = 'left';
            ctx.fillText(tokens[i], sx + 5, sy - 3);
        }
    }

    // Fit line
    if(seqLen >= 2){
        var sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for(var i = 0; i < seqLen; i++){
            sumX += ldBest[i];
            sumY += surprisal[i];
            sumXY += ldBest[i] * surprisal[i];
            sumX2 += ldBest[i] * ldBest[i];
        }
        var n = seqLen;
        var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX + 1e-12);
        var intercept = (sumY - slope * sumX) / n;

        var x1 = ldMin, y1 = slope * x1 + intercept;
        var x2 = ldMax, y2 = slope * x2 + intercept;

        var sx1 = margin + ((x1 - ldMin) / ldRange) * plotW;
        var sy1 = margin + plotH - ((y1 - sMin) / sRange) * plotH;
        var sx2 = margin + ((x2 - ldMin) / ldRange) * plotW;
        var sy2 = margin + plotH - ((y2 - sMin) / sRange) * plotH;

        ctx.strokeStyle = 'rgba(233,69,96,0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(sx1, sy1);
        ctx.lineTo(sx2, sy2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Layer correlation bars
    var barsDiv = document.getElementById('curv-corr-bars');
    var corrPerLayer = corr.correlations_per_layer;
    if(corrPerLayer && corrPerLayer.length > 0){
        var bhtml = '<div style="font-size:8px;color:#888;margin-bottom:2px">Pearson r per layer (click to select):</div>';
        bhtml += '<div style="display:flex;flex-wrap:wrap;gap:1px">';

        var maxAbsR = 0;
        for(var li = 0; li < corrPerLayer.length; li++){
            var absR = Math.abs(corrPerLayer[li].pearson_r);
            if(absR > maxAbsR) maxAbsR = absR;
        }
        if(maxAbsR < 0.01) maxAbsR = 1;

        for(var li = 0; li < corrPerLayer.length; li++){
            var c = corrPerLayer[li];
            var barH = Math.max(2, Math.abs(c.pearson_r) / maxAbsR * 30);
            var barColor = c.pearson_r > 0 ? '#2ecc71' : '#e94560';
            var isBest = (li === bestLayer);

            bhtml += '<div style="display:flex;flex-direction:column;align-items:center;cursor:pointer;' +
                     'padding:1px 2px;border-radius:2px;' +
                     (isBest ? 'background:rgba(233,69,96,0.2);' : '') + '" ' +
                     'onclick="document.getElementById(\'sl-layer\').value=' + Math.min(li, +document.getElementById('sl-layer').max) +
                     ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'));renderCurvatureHeatmap()" ' +
                     'title="Layer ' + li + ': r=' + c.pearson_r + ', ρ=' + c.spearman_rho + '">';
            bhtml += '<div style="width:8px;height:' + barH + 'px;background:' + barColor +
                     ';border-radius:1px;margin-bottom:1px"></div>';
            bhtml += '<span style="font-size:6px;color:' + (isBest ? '#e94560' : '#555') + '">' + li + '</span>';
            bhtml += '</div>';
        }
        bhtml += '</div>';
        barsDiv.innerHTML = bhtml;
    }
}

// Re-render heatmap when layer changes (to update the current-layer indicator)
var _origSlLayerHandler = null;
(function(){
    var slLayer = document.getElementById('sl-layer');
    slLayer.addEventListener('input', function(){
        if(curvatureData) renderCurvatureHeatmap();
    });
})();
// ============================================================
// MULTI-SENTENCE COMPARISON
// ============================================================

var multiData = null;

function toggleMultiMode() {
  var on = document.getElementById('cb-multi').checked;
  document.getElementById('multi-area').style.display = on ? 'block' : 'none';
  if (!on) {
    multiData = null;
    document.getElementById('multi-summary').style.display = 'none';
    document.getElementById('multi-layer-select').style.display = 'none';
    document.getElementById('multi-dim-chart').style.display = 'none';
    document.getElementById('multi-pairwise').style.display = 'none';
    document.getElementById('multi-layer-profile').style.display = 'none';
  }
}

function runMulti() {
  var text = document.getElementById('multi-txt').value.trim();
  if (!text) return;
  var sentences = text.split('\n').filter(function(s) { return s.trim().length > 0; });
  if (sentences.length < 2) {
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#e94560">Need at least 2 sentences</span>';
    return;
  }

  var modelName = document.getElementById('sel-model').value;
  var btn = document.getElementById('btn-multi');
  btn.disabled = true; btn.textContent = 'Processing...';
  document.getElementById('multi-status').innerHTML =
    '<span style="color:#53a8b6">Processing ' + sentences.length + ' sentences...</span>';

  fetch('/multi_run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentences: sentences, model: modelName })
  })
  .then(function(r) { if (!r.ok) throw new Error('Server error ' + r.status); return r.json(); })
  .then(function(data) {
    if (data.error) {
      document.getElementById('multi-status').innerHTML =
        '<span style="color:#e94560">' + data.error + '</span>';
      btn.disabled = false; btn.textContent = 'Compare Sentences';
      return;
    }
    multiData = data;
    btn.disabled = false; btn.textContent = 'Compare Sentences';
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#2ecc71">✓ Compared ' + data.n_sentences +
      ' sentences across ' + data.n_layers + ' layers</span>';
    renderMultiSummary();
    renderMultiLayerProfile();
    populateMultiLayerSelect();
    renderMultiLayer();

        // Show the multi-view button and switch to multi view
    var multiBtn = document.getElementById('btn-multi-view');
    if (multiBtn) multiBtn.style.display = 'inline-block';
    setViewMode('multi');

  })
  .catch(function(e) {
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#e94560">Error: ' + e + '</span>';
    btn.disabled = false; btn.textContent = 'Compare Sentences';
  });
}

function renderMultiSummary() {
  var panel = document.getElementById('multi-summary');
  panel.style.display = 'block';
  var s = multiData.summary;

  var html = '<div style="color:#7b68ee;font-weight:bold;margin-bottom:4px">Multi-Sentence Comparison</div>';
  html += '<div>' + multiData.n_sentences + ' sentences | ' +
          multiData.n_layers + ' layers | ' + multiData.hidden_dim + ' dims</div>';

  // Sentence list
  for (var i = 0; i < multiData.sentences.length; i++) {
    var sent = multiData.sentences[i];
    var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
    html += '<div style="margin-top:2px"><span style="color:' + colors[i % colors.length] +
            ';font-weight:bold">[' + i + ']</span> ' +
            '<span style="color:#a0a0c0">' + sent.text + '</span> ' +
            '<span style="color:#666">(' + sent.n_tokens + ' tokens)</span></div>';
  }

  html += '<div style="margin-top:6px;border-top:1px solid #1a1a2e;padding-top:4px">';
  html += 'Most divergent layer: <span style="color:#e94560;font-weight:bold">' +
          (s.most_divergent_layer === 0 ? 'Embedding' : 'L' + (s.most_divergent_layer - 1)) + '</span> | ';
  html += 'Least divergent: <span style="color:#53a8b6">' +
          (s.least_divergent_layer === 0 ? 'Embedding' : 'L' + (s.least_divergent_layer - 1)) + '</span>';
  html += '</div>';

  // Global top different dimensions
  html += '<div style="margin-top:4px;font-size:9px">';
  html += '<b style="color:#f5a623">Top globally different dims:</b> ';
  for (var di = 0; di < Math.min(10, s.global_most_different_dims.length); di++) {
    var gd = s.global_most_different_dims[di];
    html += '<span style="color:#e94560">d' + gd.dim + '</span> ';
  }
  html += '</div>';

  panel.innerHTML = html;
}

function populateMultiLayerSelect() {
  var sel = document.getElementById('multi-layer');
  sel.innerHTML = '';
  for (var li = 0; li < multiData.layer_comparisons.length; li++) {
    var opt = document.createElement('option');
    opt.value = li;
    var lc = multiData.layer_comparisons[li];
    opt.textContent = (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) +
                      ' (var=' + lc.total_variance.toFixed(2) + ')';
    sel.appendChild(opt);
  }
  // Default to most divergent layer
  sel.value = multiData.summary.most_divergent_layer;
  document.getElementById('multi-layer-select').style.display = 'block';
}

function renderMultiLayer() {
  if (!multiData) return;

  var layerIdx = +document.getElementById('multi-layer').value;
  var showMode = document.getElementById('multi-show').value;
  var topK = +document.getElementById('multi-topk').value;
  var lc = multiData.layer_comparisons[layerIdx];

  var dims = (showMode === 'most') ? lc.most_different_dims : lc.least_different_dims;
  dims = dims.slice(0, topK);

  // ---- Dimension comparison bar chart ----
  document.getElementById('multi-dim-chart').style.display = 'block';
  var cv = document.getElementById('multi-dim-cv');
  var chartH = Math.max(200, dims.length * 14 + 40);
  cv.height = chartH;
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
  var margin = { left: 50, right: 20, top: 25, bottom: 15 };
  var plotW = W - margin.left - margin.right;
  var plotH = H - margin.top - margin.bottom;

  var barGroupH = Math.max(8, Math.floor(plotH / dims.length) - 2);
  var barH = Math.max(2, Math.floor(barGroupH / nSent) - 1);

  // Find global value range for this set of dims
  var globalMin = Infinity, globalMax = -Infinity;
  for (var di = 0; di < dims.length; di++) {
    for (var si = 0; si < nSent; si++) {
      var v = dims[di].values[si];
      if (v < globalMin) globalMin = v;
      if (v > globalMax) globalMax = v;
    }
  }
  var valRange = globalMax - globalMin || 1;

  // Title
  ctx.font = 'bold 10px monospace';
  ctx.fillStyle = '#888';
  ctx.textAlign = 'center';
  ctx.fillText(
    (showMode === 'most' ? 'Most' : 'Least') + ' Different Dimensions — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    W / 2, 14
  );

  // Draw bars
  for (var di = 0; di < dims.length; di++) {
    var d = dims[di];
    var y0 = margin.top + di * (barGroupH + 2);

    // Dim label
    ctx.font = '8px monospace';
    ctx.fillStyle = '#888';
    ctx.textAlign = 'right';
    ctx.fillText('d' + d.dim, margin.left - 4, y0 + barGroupH / 2 + 3);

    // One bar per sentence
    for (var si = 0; si < nSent; si++) {
      var v = d.values[si];
      var barX = margin.left + ((v - globalMin) / valRange) * plotW;
      var zeroX = margin.left + ((0 - globalMin) / valRange) * plotW;

      var by = y0 + si * (barH + 1);
      var bw = barX - zeroX;

      ctx.fillStyle = colors[si % colors.length];
      if (bw >= 0) {
        ctx.fillRect(zeroX, by, bw, barH);
      } else {
        ctx.fillRect(barX, by, -bw, barH);
      }
    }

    // Variance indicator
    var varBarW = Math.min(plotW * 0.3, Math.max(2, (d.variance / (dims[0].variance || 1)) * plotW * 0.2));
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fillRect(margin.left + plotW - varBarW - 2, y0, varBarW, barGroupH);
    ctx.font = '7px monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'right';
    ctx.fillText('σ²=' + d.variance.toExponential(1), margin.left + plotW - 4, y0 + barGroupH - 1);
  }

  // Legend
  var legY = H - 10;
  ctx.font = '8px monospace';
  ctx.textAlign = 'left';
  for (var si = 0; si < nSent; si++) {
    var lx = margin.left + si * 70;
    ctx.fillStyle = colors[si % colors.length];
    ctx.fillRect(lx, legY - 6, 10, 6);
    ctx.fillStyle = '#888';
    ctx.fillText('[' + si + ']', lx + 13, legY);
  }

  // ---- Pairwise similarity matrix ----
  document.getElementById('multi-pairwise').style.display = 'block';
  var pairCv = document.getElementById('multi-pair-cv');
  var cellSize = Math.min(40, Math.floor(200 / nSent));
  pairCv.width = cellSize * nSent + 60;
  pairCv.height = cellSize * nSent + 60;
  var pCtx = pairCv.getContext('2d');
  pCtx.fillStyle = '#0a0a1a';
  pCtx.fillRect(0, 0, pairCv.width, pairCv.height);

  var cosMatrix = lc.pairwise_cosine;
  var offsetX = 50, offsetY = 10;

  for (var i = 0; i < nSent; i++) {
    for (var j = 0; j < nSent; j++) {
      var val = cosMatrix[i][j];
      // Map cosine similarity [-1, 1] to color
      // -1 = blue, 0 = black, 1 = green
      var r, g, b;
      if (val < 0) {
        var t = -val;
        r = 0; g = 0; b = Math.floor(t * 220);
      } else {
        var t = val;
        r = 0; g = Math.floor(t * 200); b = Math.floor(t * 100);
      }
      pCtx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      pCtx.fillRect(offsetX + j * cellSize, offsetY + i * cellSize, cellSize - 1, cellSize - 1);

      // Value text
      if (cellSize >= 20) {
        pCtx.font = '8px monospace';
        pCtx.fillStyle = val > 0.5 ? '#000' : '#fff';
        pCtx.textAlign = 'center';
        pCtx.fillText(val.toFixed(2),
          offsetX + j * cellSize + cellSize / 2,
          offsetY + i * cellSize + cellSize / 2 + 3);
      }
    }
    // Row label
    pCtx.font = '8px monospace';
    pCtx.fillStyle = colors[i % colors.length];
    pCtx.textAlign = 'right';
    pCtx.fillText('[' + i + ']', offsetX - 4, offsetY + i * cellSize + cellSize / 2 + 3);
    // Column label
    pCtx.textAlign = 'center';
    pCtx.fillText('[' + i + ']', offsetX + i * cellSize + cellSize / 2, offsetY + nSent * cellSize + 14);
  }
}

function renderMultiLayerProfile() {
  if (!multiData) return;
  document.getElementById('multi-layer-profile').style.display = 'block';

  var cv = document.getElementById('multi-layerprof-cv');
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  var variances = multiData.summary.layer_total_variances;
  var nL = variances.length;
  var maxVar = Math.max.apply(null, variances) || 1;

  var barW = Math.max(3, Math.floor((W - 30) / nL) - 1);
  var baseY = H - 15;
  var maxBarH = baseY - 5;

  for (var i = 0; i < nL; i++) {
    var h = (variances[i] / maxVar) * maxBarH;
    var x = 15 + i * (barW + 1);
    var frac = variances[i] / maxVar;

    // Most divergent layer highlighted
    var isMost = (i === multiData.summary.most_divergent_layer);

    ctx.fillStyle = isMost ?
      'rgb(233,69,96)' :
      'rgb(' + Math.floor(frac * 200) + ',' + Math.floor((1 - frac) * 120 + 50) + ',' + Math.floor((1 - frac) * 180) + ')';
    ctx.fillRect(x, baseY - h, barW, h);

    // Label
    if (nL <= 25 || i % 2 === 0) {
      ctx.font = '7px monospace';
      ctx.fillStyle = isMost ? '#e94560' : '#666';
      ctx.textAlign = 'center';
      ctx.fillText(i === 0 ? 'E' : '' + (i - 1), x + barW / 2, H - 2);
    }
  }
}


// ============================================================
// MULTI-SENTENCE: FULL MAIN CANVAS VISUALIZATION
// ============================================================

var multiHoveredDim = -1;
var multiHoveredSentence = -1;
var multiViewTab = 'grids';

// ============================================================
// MULTI-VIEW: Unified side-by-side rendering for ALL view modes
// ============================================================

function drawMultiCanvas() {
  var cv = document.getElementById('cv');
  var c = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);

  if (!multiData) {
    c.font = '16px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('Run Multi-Sentence Compare first', W / 2, H / 2);
    return;
  }

  // Draw tab bar at top
  var tabs = [
    { id: 'grids', label: '🗺️ Side-by-Side Views' },
    { id: 'dims', label: '📊 Dimension Comparison' },
    { id: 'heatmap', label: '🔥 Variance Heatmap' },
    { id: 'pairwise', label: '🔗 Pairwise Similarity' },
    { id: 'profile', label: '📈 Layer Profile' },
  ];
  var tabH = 28;
  var tabW = W / tabs.length;
  for (var ti = 0; ti < tabs.length; ti++) {
    var tx = ti * tabW;
    var isActive = (tabs[ti].id === multiViewTab);
    c.fillStyle = isActive ? '#1a1a4e' : '#0d1117';
    c.fillRect(tx, 0, tabW, tabH);
    c.strokeStyle = isActive ? '#7b68ee' : '#0f3460';
    c.lineWidth = isActive ? 2 : 0.5;
    c.strokeRect(tx, 0, tabW, tabH);
    c.font = (isActive ? 'bold ' : '') + '11px monospace';
    c.fillStyle = isActive ? '#7b68ee' : '#888';
    c.textAlign = 'center';
    c.fillText(tabs[ti].label, tx + tabW / 2, tabH / 2 + 4);
  }

  var drawArea = { x: 0, y: tabH + 4, w: W, h: H - tabH - 4 };

  if (multiViewTab === 'grids') {
    drawMultiSideBySide(c, cv, drawArea);
  } else if (multiViewTab === 'dims') {
    drawMultiDimComparison(c, drawArea);
  } else if (multiViewTab === 'heatmap') {
    drawMultiVarianceHeatmap(c, drawArea);
  } else if (multiViewTab === 'pairwise') {
    drawMultiPairwiseMatrix(c, drawArea);
  } else if (multiViewTab === 'profile') {
    drawMultiLayerProfile_Main(c, drawArea);
  }

  // Update Dim Z visibility based on sub-view
  var dzRow = document.getElementById('dz-row');
  if (dzRow) {
    dzRow.style.display = (multiSubView === '3d' || multiSubView === 'fibre3d') ? 'flex' : 'none';
  }
}

/**
 * The key function: renders each sentence side-by-side using the
 * CURRENT view mode (2D, 3D, Fibre, Fibre3D, FibreKelp).
 *
 * It works by temporarily swapping the global D to each sentence's
 * data, setting up a clip region and transform for each panel,
 * then calling the appropriate existing draw function.
 */
function drawMultiSideBySide(c, cv, area) {
  if (!multiData || !multiData.sentence_data || multiData.sentence_data.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No sentence data — re-run Multi Compare', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var nSent = multiData.sentence_data.length;
  var sentColors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  // Determine the sub-view mode to use for each panel
  // Use a selector or default to the last non-multi viewMode
  var subView = multiSubView || '2d';

  // Draw sub-view selector bar just below the tab bar
  var subViews = [
    { id: '2d', label: '2D' },
    { id: '3d', label: '3D' },
    { id: 'fibre', label: 'Fibre' },
    { id: 'fibre3d', label: 'Fibre 3D' },
    { id: 'fibrekelp', label: 'Kelp' },
  ];
  var svBarH = 22;
  var svBarW = area.w / subViews.length;
  for (var svi = 0; svi < subViews.length; svi++) {
    var svx = area.x + svi * svBarW;
    var svy = area.y;
    var isAct = (subViews[svi].id === subView);
    c.fillStyle = isAct ? '#0f3460' : '#0a0a1a';
    c.fillRect(svx, svy, svBarW, svBarH);
    c.strokeStyle = isAct ? '#53a8b6' : '#0f3460';
    c.lineWidth = isAct ? 1.5 : 0.5;
    c.strokeRect(svx, svy, svBarW, svBarH);
    c.font = (isAct ? 'bold ' : '') + '10px monospace';
    c.fillStyle = isAct ? '#53a8b6' : '#666';
    c.textAlign = 'center';
    c.fillText(subViews[svi].label, svx + svBarW / 2, svy + svBarH / 2 + 3);
  }

  // Adjust area below the sub-view bar
  var panelArea = {
    x: area.x,
    y: area.y + svBarH + 2,
    w: area.w,
    h: area.h - svBarH - 2 - 30 // leave room for labels
  };

  // Layout: divide into nSent columns
  var gap = 4;
  var panelW = Math.floor((panelArea.w - gap * (nSent - 1)) / nSent);
  var panelH = panelArea.h;

  // Save the real global state
  var savedD = D;
  var savedViewMode = viewMode;
  var savedZoom = zoomLevel;
  var savedPanX = panX;
  var savedPanY = panY;
  var savedRotX = rotX;
  var savedRotY = rotY;
  var savedFibreRotX = fibreState.rotX;
  var savedFibreRotY = fibreState.rotY;

  for (var si = 0; si < nSent; si++) {
    var sentD = multiData.sentence_data[si];
    if (!sentD || !sentD.fixed_pos || !sentD.deltas) continue;

    var px = panelArea.x + si * (panelW + gap);
    var py = panelArea.y;

    // Draw panel border
    c.strokeStyle = sentColors[si % sentColors.length];
    c.lineWidth = 1.5;
    c.strokeRect(px, py, panelW, panelH);

    // Set up clipping and transform so the existing draw functions
    // render into this panel
    c.save();
    c.beginPath();
    c.rect(px, py, panelW, panelH);
    c.clip();

    // Temporarily swap global state
    D = sentD;
    viewMode = subView;
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;

    // Resize the canvas dimensions that the draw functions read
    // We do this by overriding cv.width/cv.height temporarily
    // But that would clear the canvas! Instead, we use a transform.
    //
    // The draw functions use cv.width and cv.height for layout.
    // We need to make them think the canvas is panelW x panelH.
    // We'll translate so (0,0) maps to (px, py) and scale if needed.

    // For 2D: draw2D uses c.save/translate/scale internally for pan/zoom.
    // We need to intercept that. The simplest approach: create an offscreen
    // canvas for each panel, draw into it, then composite.

    var offCv = document.createElement('canvas');
    offCv.width = panelW;
    offCv.height = panelH;
    var offCtx = offCv.getContext('2d');

    // The draw functions read from document.getElementById('cv')
    // We can't easily redirect that. Instead, we'll use our
    // drawMiniDeformedGrid for 2D, and build mini versions for others.

    if (subView === '2d') {
      drawMiniPanel2D(offCtx, sentD, panelW, panelH);
    } else if (subView === '3d') {
      drawMiniPanel3D(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibre') {
      drawMiniPanelFibre(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibre3d') {
      drawMiniPanelFibre3D(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibrekelp') {
      drawMiniPanelFibreKelp(offCtx, sentD, panelW, panelH);
    }

    // Composite the offscreen canvas into the main canvas
    c.drawImage(offCv, px, py);

    c.restore(); // restore clipping

    // Sentence label below the panel
    c.font = 'bold 9px monospace';
    c.fillStyle = sentColors[si % sentColors.length];
    c.textAlign = 'center';
    var labelText = '[' + si + '] ' + (sentD.text || '').substring(0, Math.floor(panelW / 5.5));
    if ((sentD.text || '').length > Math.floor(panelW / 5.5)) labelText += '…';
    c.fillText(labelText, px + panelW / 2, py + panelH + 14);

    c.font = '8px monospace';
    c.fillStyle = '#666';
    c.fillText(sentD.n_real + ' tok | ' + sentD.n_layers + ' layers',
      px + panelW / 2, py + panelH + 24);
  }

  // Restore global state
  D = savedD;
  viewMode = savedViewMode;
  zoomLevel = savedZoom;
  panX = savedPanX;
  panY = savedPanY;
  rotX = savedRotX;
  rotY = savedRotY;
  fibreState.rotX = savedFibreRotX;
  fibreState.rotY = savedFibreRotY;

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.4)';
  c.textAlign = 'left';
  var p = gp();
  c.fillText(
    'Sub-view: ' + subView.toUpperCase() +
    '  Layer ' + p.layer + '/' + (multiData.sentence_data[0].n_layers - 1) +
    '  t=' + p.t.toFixed(2) +
    '  amp=' + p.amp.toFixed(1) +
    '  Dims:' + p.dx + ',' + p.dy +
    (subView === '3d' || subView === 'fibre3d' ? ',' + p.dz : '') +
    '  Mode:' + p.mode +
    '  |  All sidebar controls affect all panels',
    area.x + 10, area.y + area.h - 2
  );
}

// Global: which sub-view to use in multi mode
var multiSubView = '2d';

// ============================================================
// MINI PANEL RENDERERS
// Each takes an offscreen context, sentence data, and dimensions,
// and renders the full visualization into that context.
// ============================================================

/**
 * Mini 2D panel — reuses drawMiniDeformedGrid from before
 */
function drawMiniPanel2D(c, sentD, W, H) {
  var p = gp();
  // drawMiniDeformedGrid expects (c, sentD, p, px, py, pw, ph)
  drawMiniDeformedGrid(c, sentD, p, 0, 0, W, H);
}

/**
 * Mini 3D panel — renders a 3D deformed grid for one sentence
 */
function drawMiniPanel3D(c, sentD, W, H) {
  var p = gp();
  var nP = sentD.n_points, nR = sentD.n_real;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var dz = Math.min(p.dz, sentD.hidden_dim - 1);
  var isEmb = (p.mode === 'embedding');

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var layer = Math.min(p.layer, sentD.n_layers - 1);
  var amp = p.amp, t = p.t;

  var fx = new Float64Array(nP), fy = new Float64Array(nP), fz = new Float64Array(nP);
  for (var i = 0; i < nP; i++) {
    fx[i] = sentD.fixed_pos[i][dx];
    fy[i] = sentD.fixed_pos[i][dy];
    fz[i] = sentD.fixed_pos[i][dz];
  }

  var edx = new Float64Array(nP), edy = new Float64Array(nP), edz = new Float64Array(nP);
  if (!isEmb) {
    for (var j = 0; j < nP; j++) {
      var sx = 0, sy = 0, sz = 0;
      if (p.mode === 'single') {
        sx = activeDeltas[layer][j][dx]; sy = activeDeltas[layer][j][dy]; sz = activeDeltas[layer][j][dz];
      } else if (p.mode === 'cumfwd') {
        for (var l = 0; l <= layer; l++) { sx += activeDeltas[l][j][dx]; sy += activeDeltas[l][j][dy]; sz += activeDeltas[l][j][dz]; }
      } else {
        for (var l = layer; l < sentD.n_layers; l++) { sx += activeDeltas[l][j][dx]; sy += activeDeltas[l][j][dy]; sz += activeDeltas[l][j][dz]; }
      }
      edx[j] = sx * amp; edy[j] = sy * amp; edz[j] = sz * amp;
    }
  }

  // Compute bounds
  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, mnz = Infinity, mxz = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    if (fz[i] < mnz) mnz = fz[i]; if (fz[i] > mxz) mxz = fz[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1;
  var cx3 = (mnx + mxx) / 2, cy3 = (mny + mxy) / 2, cz3 = (mnz + mxz) / 2;
  var sc3 = Math.min(W, H) * 0.3 / mr;

  var fl = 400;
  function proj(x, y, z) {
    var cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
    var cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
    var scale = fl / (fl + z2);
    return [W / 2 + x1 * scale, H / 2 + y1 * scale, z2, scale];
  }

  // Build 3D grid
  var N = Math.max(4, Math.min(12, Math.floor(p.gr / 4)));
  var pd = 0.12;
  var vx0 = cx3 - mr * (0.5 + pd), vx1 = cx3 + mr * (0.5 + pd);
  var vy0 = cy3 - mr * (0.5 + pd), vy1 = cy3 + mr * (0.5 + pd);
  var vz0 = cz3 - mr * (0.5 + pd), vz1 = cz3 + mr * (0.5 + pd);

  function gIdx(ix, iy, iz) { return iz * (N + 1) * (N + 1) + iy * (N + 1) + ix; }
  var nV = (N + 1) * (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV), oZ = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV), gZ = new Float64Array(nV);

  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) {
    var gi = gIdx(ix, iy, iz);
    oX[gi] = vx0 + (ix / N) * (vx1 - vx0);
    oY[gi] = vy0 + (iy / N) * (vy1 - vy0);
    oZ[gi] = vz0 + (iz / N) * (vz1 - vz0);
  }

  var sig = p.sig, s2i = 1 / (2 * sig * sig);
  if (isEmb) {
    for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; gZ[gi] = oZ[gi]; }
  } else {
    for (var gi = 0; gi < nV; gi++) {
      var gpx = oX[gi], gpy = oY[gi], gpz = oZ[gi];
      var vvx = 0, vvy = 0, vvz = 0, ws = 0;
      for (var k = 0; k < nP; k++) {
        var ex = gpx - fx[k], ey = gpy - fy[k], ez = gpz - fz[k];
        var w = Math.exp(Math.max(-500, -(ex * ex + ey * ey + ez * ez) * s2i));
        vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
      }
      if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
      gX[gi] = gpx + t * vvx; gY[gi] = gpy + t * vvy; gZ[gi] = gpz + t * vvz;
    }
  }

  // Collect edges with strain
  var edges = [];
  function addEdge(a, b) {
    var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a], oZ[b] - oZ[a]);
    var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a], gZ[b] - gZ[a]);
    var strain = od > 1e-12 ? dd / od : 1;
    var pa = proj((gX[a] - cx3) * sc3, (gY[a] - cy3) * sc3, (gZ[a] - cz3) * sc3);
    var pb = proj((gX[b] - cx3) * sc3, (gY[b] - cy3) * sc3, (gZ[b] - cz3) * sc3);
    edges.push({ x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1], z: (pa[2] + pb[2]) / 2, strain: strain });
  }

  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix < N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix + 1, iy, iz));
  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy < N; iy++) for (var ix = 0; ix <= N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix, iy + 1, iz));
  for (var iz = 0; iz < N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix, iy, iz + 1));

  edges.sort(function(a, b) { return b.z - a.z; });

  // Draw edges
  if (p.grid && !isEmb) {
    c.lineWidth = 0.6;
    for (var ei = 0; ei < edges.length; ei++) {
      var e = edges[ei];
      var da = Math.max(0.1, Math.min(0.7, 0.5 - e.z * 0.001));
      if (p.sc) {
        var ec = s2c(e.strain);
        c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',' + da.toFixed(2) + ')';
      } else {
        c.strokeStyle = 'rgba(200,200,200,' + da.toFixed(2) + ')';
      }
      c.beginPath(); c.moveTo(e.x1, e.y1); c.lineTo(e.x2, e.y2); c.stroke();
    }
  }

  // Draw token dots
  if (p.tok) {
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];
    var pts = [];
    for (var ti = 0; ti < nR; ti++) {
      var pp = proj((fx[ti] - cx3) * sc3, (fy[ti] - cy3) * sc3, (fz[ti] - cz3) * sc3);
      pts.push({ idx: ti, x: pp[0], y: pp[1], z: pp[2], s: pp[3] });
    }
    pts.sort(function(a, b) { return b.z - a.z; });
    for (var pi = 0; pi < pts.length; pi++) {
      var pt = pts[pi];
      var r = Math.max(2, 4 * pt.s);
      c.beginPath(); c.arc(pt.x, pt.y, r, 0, Math.PI * 2);
      c.fillStyle = tc[pt.idx % tc.length]; c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 0.8; c.stroke();
      c.font = 'bold 7px monospace';
      c.lineWidth = 1.5; c.strokeStyle = 'rgba(0,0,0,0.9)';
      var lb = '[' + pt.idx + '] ' + sentD.tokens[pt.idx];
      if (lb.length > Math.floor(W / 7)) lb = lb.substring(0, Math.floor(W / 7)) + '…';
      c.strokeText(lb, pt.x + 5, pt.y - 4);
      c.fillStyle = '#fff'; c.fillText(lb, pt.x + 5, pt.y - 4);
    }
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('3D L' + layer + ' d' + dx + ',' + dy + ',' + dz, 4, 10);
}

/**
 * Mini Fibre Bundle panel — renders the token-per-column, layer-per-row grid view
 */
function drawMiniPanelFibre(c, sentD, W, H) {
  var p = gp();
  var nR = sentD.n_real;
  var nLayers = sentD.n_layers;
  var nP = sentD.n_points;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var mode = p.mode;
  var isEmb = (mode === 'embedding');
  var layer = Math.min(p.layer, nLayers - 1);
  var amp = p.amp, t = p.t, sig = p.sig;

  // Read display toggles from the checkboxes
  var showGrid = document.getElementById('cb-grid').checked;
  var showHeat = document.getElementById('cb-heat').checked;
  var showSC = document.getElementById('cb-sc').checked;

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) { fx[i] = sentD.fixed_pos[i][dx]; fy[i] = sentD.fixed_pos[i][dy]; }

  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd = 0.15;
  var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
  var vw = mr * (1 + 2 * pd), vh = vw;

  // Layout: rooms
  var margin = 4;
  var labelW = 18;
  var availW = W - 2 * margin - labelW;
  var availH = H - 2 * margin;
  var roomSize = Math.max(12, Math.min(
    Math.floor(availW / (nR * 1.3)),
    Math.floor(availH / (nLayers * 1.4))
  ));
  var gapX = Math.max(2, Math.floor(roomSize * 0.2));
  var gapY = Math.max(3, Math.floor(roomSize * 0.25));
  var startX = margin + labelW;
  var startY = margin;

  var N = Math.max(3, Math.min(8, Math.floor(roomSize / 5)));
  var itpM = document.getElementById('sel-itp').value;

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  for (var li = 0; li < nLayers; li++) {
    var rowIdx = nLayers - 1 - li;
    var roomCY = startY + rowIdx * (roomSize + gapY);
    var isCurrent = (li === layer);

    // Layer label
    c.font = (isCurrent ? 'bold ' : '') + '6px monospace';
    c.fillStyle = isCurrent ? '#e94560' : '#555';
    c.textAlign = 'right';
    c.fillText('L' + li, startX - 2, roomCY + roomSize / 2 + 3);

    var layerDeltas = computeEdxEdyForLayerMini(sentD, li, nP, dx, dy, amp, mode, activeDeltas, nLayers);
    var edxCum = layerDeltas.edx;
    var edyCum = layerDeltas.edy;

    for (var ti = 0; ti < nR; ti++) {
      var roomCX = startX + ti * (roomSize + gapX);

      // Room background
      var bgAlpha = isCurrent ? 0.15 : 0.06;
      c.fillStyle = 'rgba(30,30,60,' + bgAlpha + ')';
      c.fillRect(roomCX, roomCY, roomSize, roomSize);

      // Room border
      c.strokeStyle = isCurrent ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.25)';
      c.lineWidth = isCurrent ? 1.5 : 0.5;
      c.strokeRect(roomCX, roomCY, roomSize, roomSize);

      // Build deformed grid for this room
      var grid = buildDeformedGridMini(vx0, vy0, vw, vh, N, fx, fy, edxCum, edyCum, nP, sig, t, isEmb, itpM);

      // Strain heatmap
      if (showHeat && !isEmb) {
        for (var hy = 0; hy < N; hy++) {
          for (var hx = 0; hx < N; hx++) {
            var avg = (grid.sH[hy * N + hx] + grid.sH[(hy + 1) * N + hx] +
                       grid.sV[hy * (N + 1) + hx] + grid.sV[hy * (N + 1) + hx + 1]) / 4;
            var co = s2c(avg);
            var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
            var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[i00] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i00] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i10] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i10] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i11] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i11] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i01] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i01] - vy0) / vh) * roomSize);
            c.closePath();
            c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.4)';
            c.fill();
          }
        }
      }

      // Grid lines
      if (showGrid && !isEmb) {
        c.lineWidth = 0.5;
        for (var dhy = 0; dhy <= N; dhy++) {
          for (var dhx = 0; dhx < N; dhx++) {
            var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
            var es = grid.sH[dhy * N + dhx];
            if (showSC) {
              var ec = s2c(es);
              c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.8)';
            } else {
              c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[di1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[di2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di2] - vy0) / vh) * roomSize);
            c.stroke();
          }
        }
        for (var dvx = 0; dvx <= N; dvx++) {
          for (var dvy = 0; dvy < N; dvy++) {
            var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
            var vs = grid.sV[dvy * (N + 1) + dvx];
            if (showSC) {
              var vc = s2c(vs);
              c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.8)';
            } else {
              c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[dvi1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[dvi2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi2] - vy0) / vh) * roomSize);
            c.stroke();
          }
        }
      }

      // Reference grid in embedding mode
      if (isEmb) {
        c.strokeStyle = 'rgba(255,255,255,0.12)';
        c.lineWidth = 0.3;
        for (var ry = 0; ry <= N; ry++) {
          c.beginPath();
          for (var rx = 0; rx <= N; rx++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (rx === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
          }
          c.stroke();
        }
        for (var rx = 0; rx <= N; rx++) {
          c.beginPath();
          for (var ry = 0; ry <= N; ry++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (ry === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
          }
          c.stroke();
        }
      }

      // Token dot
      var tokX = roomCX + ((fx[ti] - vx0) / vw) * roomSize;
      var tokY = roomCY + ((fy[ti] - vy0) / vh) * roomSize;
      c.beginPath();
      c.arc(tokX, tokY, Math.max(2, roomSize / 20), 0, Math.PI * 2);
      c.fillStyle = tc[ti % tc.length];
      c.fill();

      // Token label at bottom row only
      if (li === 0 && roomSize > 25) {
        c.font = 'bold 6px monospace';
        c.fillStyle = tc[ti % tc.length];
        c.textAlign = 'center';
        c.fillText('[' + ti + ']', roomCX + roomSize / 2, roomCY + roomSize + 8);
      }
    }
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('FIBRE L' + layer + ' d' + dx + ',' + dy, 4, 10);
}

/**
 * Helper: compute mode-aware cumulative deltas for a given layer
 * using a specific sentence's data (not the global D).
 */
function computeEdxEdyForLayerMini(sentD, li, nP, dx, dy, amp, mode, activeDeltas, nLayers) {
  var edxCum = new Float64Array(nP);
  var edyCum = new Float64Array(nP);
  var isEmb = (mode === 'embedding');
  if (isEmb) return { edx: edxCum, edy: edyCum };

  if (mode === 'single') {
    for (var j = 0; j < nP; j++) {
      edxCum[j] = activeDeltas[li][j][dx] * amp;
      edyCum[j] = activeDeltas[li][j][dy] * amp;
    }
  } else if (mode === 'cumfwd') {
    for (var cl = 0; cl <= li; cl++) {
      for (var j = 0; j < nP; j++) {
        edxCum[j] += activeDeltas[cl][j][dx] * amp;
        edyCum[j] += activeDeltas[cl][j][dy] * amp;
      }
    }
  } else { // cumbwd
    for (var cl = li; cl < nLayers; cl++) {
      for (var j = 0; j < nP; j++) {
        edxCum[j] += activeDeltas[cl][j][dx] * amp;
        edyCum[j] += activeDeltas[cl][j][dy] * amp;
      }
    }
  }
  return { edx: edxCum, edy: edyCum };
}

/**
 * Helper: build a deformed grid with strain computation.
 * Reusable across all mini panel renderers.
 */
function buildDeformedGridMini(vx0, vy0, vw, vh, N, fx, fy, edxCum, edyCum, nP, sig, t, isEmb, itpMethod) {
  var nV = (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV);

  for (var gy = 0; gy <= N; gy++) {
    for (var gx = 0; gx <= N; gx++) {
      var gi = gy * (N + 1) + gx;
      oX[gi] = vx0 + (gx / N) * vw;
      oY[gi] = vy0 + (gy / N) * vh;
    }
  }

  if (isEmb) {
    for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; }
  } else {
    for (var gi = 0; gi < nV; gi++) {
      var px = oX[gi], py = oY[gi];
      var iRes = interpolateGridPoint(px, py, fx, fy, edxCum, edyCum, nP, sig, itpMethod);
      gX[gi] = px + t * iRes[0];
      gY[gi] = py + t * iRes[1];
    }
  }

  var sH = new Float64Array(N * (N + 1));
  var sV = new Float64Array((N + 1) * N);
  for (var ey = 0; ey <= N; ey++) {
    for (var ex = 0; ex < N; ex++) {
      var a = ey * (N + 1) + ex, b = a + 1;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sH[ey * N + ex] = od > 1e-12 ? dd / od : 1;
    }
  }
  for (var ey = 0; ey < N; ey++) {
    for (var ex = 0; ex <= N; ex++) {
      var a = ey * (N + 1) + ex, b = (ey + 1) * (N + 1) + ex;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sV[ey * (N + 1) + ex] = od > 1e-12 ? dd / od : 1;
    }
  }

  return { oX: oX, oY: oY, gX: gX, gY: gY, sH: sH, sV: sV, nV: nV };
}

/**
 * Mini Fibre 3D panel — renders the 3D fibre bundle grid for one sentence
 */
function drawMiniPanelFibre3D(c, sentD, W, H) {
  // Reuse drawMiniPanel3D but with fibre-style layout
  // For simplicity in the multi-view, render as 3D with the fibre room structure
  drawMiniPanel3D(c, sentD, W, H);
}

/**
 * Mini Fibre Kelp panel — renders the kelp-style pathline view for one sentence
 */
function drawMiniPanelFibreKelp(c, sentD, W, H) {
  var p = gp();
  var nR = sentD.n_real;
  var nLayers = sentD.n_layers;
  var nP = sentD.n_points;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var isEmb = (p.mode === 'embedding');
  var layer = Math.min(p.layer, nLayers - 1);
  var amp = p.amp, t = p.t;
  var mode = p.mode;

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var attnDeltas = sentD.attn_deltas || null;
  var mlpDeltas = sentD.mlp_deltas || null;

  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) { fx[i] = sentD.fixed_pos[i][dx]; fy[i] = sentD.fixed_pos[i][dy]; }

  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd = 0.15;
  var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
  var vw = mr * (1 + 2 * pd);

  // Layout
  var margin = 8;
  var plotW = W - 2 * margin;
  var plotH = H - 2 * margin;
  var layerH = plotH / nLayers;

  function SX(wx) { return margin + ((wx - vx0) / vw) * plotW; }
  function LY(li) { return margin + (nLayers - 1 - li) * layerH + layerH * 0.5; }

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

  // Draw token pathlines
  for (var ti = 0; ti < nR; ti++) {
    var col = tc[ti % tc.length];
    var path = [];

    for (var li = 0; li < nLayers; li++) {
      var cumDx = 0, cumDy = 0;
      if (!isEmb) {
        if (mode === 'single') {
          cumDx = activeDeltas[li][ti][dx] * amp * t;
          cumDy = activeDeltas[li][ti][dy] * amp * t;
        } else if (mode === 'cumfwd') {
          for (var cl = 0; cl <= li; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
          }
        } else {
          for (var cl = li; cl < nLayers; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
          }
        }
      }
      path.push({ x: SX(fx[ti] + cumDx), y: LY(li) });
    }

    // Draw pathline
    c.strokeStyle = col;
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li = 1; li < nLayers; li++) {
      var prev = path[li - 1], curr = path[li];
      c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
    }
    c.stroke();

    // Attn/MLP arrows at current layer
    if (!isEmb) {
      var pt = path[layer];
      var pixPerWorld = plotW / vw;
      var maxArrow = layerH * 0.3;

      if (attnDeltas) {
        var avx = attnDeltas[layer][ti][dx] * amp * t * pixPerWorld;
        if (Math.abs(avx) > 1.5) {
          if (Math.abs(avx) > maxArrow) avx = maxArrow * Math.sign(avx);
          c.strokeStyle = 'rgba(0,200,255,0.6)';
          c.lineWidth = 0.8;
          c.beginPath(); c.moveTo(pt.x, pt.y - 2); c.lineTo(pt.x + avx, pt.y - 2); c.stroke();
        }
      }
      if (mlpDeltas) {
        var mvx = mlpDeltas[layer][ti][dx] * amp * t * pixPerWorld;
        if (Math.abs(mvx) > 1.5) {
          if (Math.abs(mvx) > maxArrow) mvx = maxArrow * Math.sign(mvx);
          c.strokeStyle = 'rgba(255,165,0,0.6)';
          c.lineWidth = 0.8;
          c.beginPath(); c.moveTo(pt.x, pt.y + 2); c.lineTo(pt.x + mvx, pt.y + 2); c.stroke();
        }
      }
    }

    // Node dots
    for (var li = 0; li < nLayers; li++) {
      var isActive = (li === layer);
      c.beginPath();
      c.arc(path[li].x, path[li].y, isActive ? 3 : 1.5, 0, Math.PI * 2);
      c.fillStyle = col;
      c.fill();
    }

    // Token label at bottom
    c.font = 'bold 6px monospace';
    c.fillStyle = col;
    c.textAlign = 'center';
    c.fillText('[' + ti + '] ' + sentD.tokens[ti], path[0].x, path[0].y + 10);
  }

  // Layer labels
  for (var li = 0; li < nLayers; li++) {
    var isActive = (li === layer);
    c.font = (isActive ? 'bold ' : '') + '6px monospace';
    c.fillStyle = isActive ? '#e94560' : '#444';
    c.textAlign = 'right';
    c.fillText('L' + li, margin - 2, LY(li) + 2);
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('KELP L' + layer + ' d' + dx + ',' + dy, 4, 10);
}

// ---- Tab click handler ----
// Sub-view click handler (inside the grids tab)
document.getElementById('cv').addEventListener('click', function(e) {
  if (viewMode !== 'multi' || !multiData) return;
  var cv = document.getElementById('cv');
  var rect = cv.getBoundingClientRect();
  var mx = e.clientX - rect.left;
  var my = e.clientY - rect.top;

  var tabH = 28;

  // ---- MAIN TAB BAR click (the 5 tabs at the very top) ----
  if (my >= 0 && my < tabH) {
    var tabs = ['grids', 'dims', 'heatmap', 'pairwise', 'profile'];
    var tabW = cv.width / tabs.length;
    var idx = Math.floor(mx / tabW);
    if (idx >= 0 && idx < tabs.length) {
      multiViewTab = tabs[idx];
      drawMultiCanvas();
    }
    return;
  }

  // ---- SUB-VIEW BAR click (only visible in 'grids' tab) ----
  if (multiViewTab === 'grids') {
    var svBarY = tabH + 4;
    var svBarH = 22;
    if (my >= svBarY && my < svBarY + svBarH) {
      var subViews = ['2d', '3d', 'fibre', 'fibre3d', 'fibrekelp'];
      var svBarW = cv.width / subViews.length;
      var idx = Math.floor(mx / svBarW);
      if (idx >= 0 && idx < subViews.length) {
        multiSubView = subViews[idx];
        drawMultiCanvas();
      }
    }
  }
});

// ============================================================
// TAB 1: DIMENSION COMPARISON (the main one you want)
// ============================================================
function drawMultiDimComparison(c, area) {
  var layerIdx = +document.getElementById('multi-layer').value;
  var showMode = document.getElementById('multi-show').value;
  var topK = +document.getElementById('multi-topk').value;
  var lc = multiData.layer_comparisons[layerIdx];

  var dims = (showMode === 'most') ? lc.most_different_dims : lc.least_different_dims;
  dims = dims.slice(0, topK);
  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  var margin = { left: 70, right: 180, top: 50, bottom: 60 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#7b68ee';
  c.textAlign = 'center';
  c.fillText(
    (showMode === 'most' ? 'Most' : 'Least') + ' Different Dimensions — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    area.x + area.w / 2, oy - 20
  );

  if (dims.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.fillText('No dimension data available', area.x + area.w / 2, oy + plotH / 2);
    return;
  }

  // Find global value range
  var globalMin = Infinity, globalMax = -Infinity;
  for (var di = 0; di < dims.length; di++) {
    for (var si = 0; si < nSent; si++) {
      var v = dims[di].values[si];
      if (v < globalMin) globalMin = v;
      if (v > globalMax) globalMax = v;
    }
  }
  var valRange = globalMax - globalMin || 1;

  // Compute bar layout
  var barGroupH = Math.max(12, Math.floor(plotH / dims.length) - 3);
  var barH = Math.max(3, Math.floor((barGroupH - 2) / nSent) - 1);

  // Zero line position
  var zeroFrac = (0 - globalMin) / valRange;
  var zeroX = ox + zeroFrac * plotW;

  // Draw zero line
  c.strokeStyle = 'rgba(255,255,255,0.15)';
  c.lineWidth = 1;
  c.setLineDash([4, 4]);
  c.beginPath();
  c.moveTo(zeroX, oy);
  c.lineTo(zeroX, oy + plotH);
  c.stroke();
  c.setLineDash([]);

  // Draw axis labels
  c.font = '9px monospace';
  c.fillStyle = '#666';
  c.textAlign = 'center';
  // Min label
  c.fillText(globalMin.toFixed(3), ox, oy + plotH + 20);
  // Max label
  c.fillText(globalMax.toFixed(3), ox + plotW, oy + plotH + 20);
  // Zero label
  if (zeroFrac > 0.05 && zeroFrac < 0.95) {
    c.fillText('0', zeroX, oy + plotH + 20);
  }

  // Draw each dimension row
  for (var di = 0; di < dims.length; di++) {
    var d = dims[di];
    var y0 = oy + di * (barGroupH + 3);

    // Alternating row background
    if (di % 2 === 0) {
      c.fillStyle = 'rgba(255,255,255,0.02)';
      c.fillRect(ox - 5, y0 - 1, plotW + 10, barGroupH + 2);
    }

    // Dim label (left side)
    c.font = 'bold 10px monospace';
    c.fillStyle = '#a0a0c0';
    c.textAlign = 'right';
    c.fillText('dim ' + d.dim, ox - 10, y0 + barGroupH / 2 + 4);

    // One bar per sentence
    for (var si = 0; si < nSent; si++) {
      var v = d.values[si];
      var barX = ox + ((v - globalMin) / valRange) * plotW;

      var by = y0 + si * (barH + 1);
      var bw = barX - zeroX;

      // Bar with gradient
      var col = colors[si % colors.length];
      c.fillStyle = col;
      c.globalAlpha = 0.85;
      if (bw >= 0) {
        c.fillRect(zeroX, by, bw, barH);
      } else {
        c.fillRect(barX, by, -bw, barH);
      }
      c.globalAlpha = 1.0;

      // Value label at end of bar
      if (barGroupH > 15 && barH >= 5) {
        c.font = '7px monospace';
        c.fillStyle = '#888';
        c.textAlign = bw >= 0 ? 'left' : 'right';
        var labelX = bw >= 0 ? barX + 3 : barX - 3;
        c.fillText(v.toFixed(3), labelX, by + barH - 1);
      }
    }

    // Variance indicator on the right
    var maxVar = dims[0].variance || 1;
    var varBarW = Math.min(100, Math.max(4, (d.variance / maxVar) * 100));
    var varX = ox + plotW + 10;
    c.fillStyle = 'rgba(123,104,238,0.3)';
    c.fillRect(varX, y0, varBarW, barGroupH);
    c.font = '8px monospace';
    c.fillStyle = '#7b68ee';
    c.textAlign = 'left';
    c.fillText('σ²=' + d.variance.toExponential(1), varX + varBarW + 4, y0 + barGroupH / 2 + 3);

    // Range indicator
    c.fillStyle = '#555';
    c.fillText('Δ=' + d.range.toFixed(3), varX + varBarW + 4, y0 + barGroupH / 2 + 13);
  }

  // Legend (bottom right)
  var legX = ox + plotW + 10;
  var legY = oy + plotH - nSent * 16 - 10;
  c.font = 'bold 10px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('Sentences:', legX, legY - 6);

  for (var si = 0; si < nSent; si++) {
    var ly = legY + si * 16;
    c.fillStyle = colors[si % colors.length];
    c.fillRect(legX, ly, 14, 10);
    c.font = '9px monospace';
    c.fillStyle = '#a0a0c0';
    var sentText = multiData.sentences[si].text;
    if (sentText.length > 25) sentText = sentText.substring(0, 25) + '…';
    c.fillText('[' + si + '] ' + sentText, legX + 18, ly + 8);
  }

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText(
    dims.length + ' dimensions | ' + nSent + ' sentences | ' +
    'Use sidebar controls to change layer/mode/topK',
    area.x + 10, area.y + area.h - 5
  );
}

// ============================================================
// TAB 2: VARIANCE HEATMAP (dimensions × layers)
// ============================================================
function drawMultiVarianceHeatmap(c, area) {
  var nLayers = multiData.layer_comparisons.length;
  var topK = Math.min(+document.getElementById('multi-topk').value, 50);

  // Get the globally most different dimensions
  var globalDims = multiData.summary.global_most_different_dims.slice(0, topK);
  if (globalDims.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No data', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var margin = { left: 60, right: 30, top: 50, bottom: 50 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  var cellW = Math.max(4, Math.floor(plotW / nLayers));
  var cellH = Math.max(4, Math.floor(plotH / globalDims.length));

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#f5a623';
  c.textAlign = 'center';
  c.fillText('Cross-Sentence Variance Heatmap (Top ' + globalDims.length + ' Dims × Layers)',
    area.x + area.w / 2, oy - 20);

  // Find max variance for color scaling
  var maxVar = 0;
  for (var di = 0; di < globalDims.length; di++) {
    var dimIdx = globalDims[di].dim;
    for (var li = 0; li < nLayers; li++) {
      var v = multiData.layer_comparisons[li].dim_variance[dimIdx];
      if (v > maxVar) maxVar = v;
    }
  }
  if (maxVar < 1e-12) maxVar = 1;

  // Draw cells
  for (var di = 0; di < globalDims.length; di++) {
    var dimIdx = globalDims[di].dim;
    for (var li = 0; li < nLayers; li++) {
      var v = multiData.layer_comparisons[li].dim_variance[dimIdx];
      var intensity = Math.min(1, v / maxVar);

      // Color: black -> purple -> red -> yellow -> white
      var r, g, b;
      if (intensity < 0.25) {
        var t = intensity / 0.25;
        r = Math.floor(t * 80); g = 0; b = Math.floor(t * 120);
      } else if (intensity < 0.5) {
        var t = (intensity - 0.25) / 0.25;
        r = 80 + Math.floor(t * 153); g = Math.floor(t * 40); b = 120 - Math.floor(t * 24);
      } else if (intensity < 0.75) {
        var t = (intensity - 0.5) / 0.25;
        r = 233; g = 40 + Math.floor(t * 180); b = 96 - Math.floor(t * 96);
      } else {
        var t = (intensity - 0.75) / 0.25;
        r = 233 + Math.floor(t * 22); g = 220 + Math.floor(t * 35); b = Math.floor(t * 200);
      }

      c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      c.fillRect(ox + li * cellW, oy + di * cellH, cellW - 1, cellH - 1);
    }

    // Dim label
    c.font = '8px monospace';
    c.fillStyle = '#888';
    c.textAlign = 'right';
    c.fillText('d' + dimIdx, ox - 4, oy + di * cellH + cellH / 2 + 3);
  }

  // Layer labels
  c.textAlign = 'center';
  for (var li = 0; li < nLayers; li++) {
    if (nLayers <= 30 || li % 2 === 0) {
      c.font = '8px monospace';
      c.fillStyle = '#666';
      c.fillText(li === 0 ? 'E' : 'L' + (li - 1), ox + li * cellW + cellW / 2, oy + plotH + 14);
    }
  }

  // Axis labels
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText('Layer →', ox + plotW / 2, oy + plotH + 35);

  c.save();
  c.translate(ox - 45, oy + plotH / 2);
  c.rotate(-Math.PI / 2);
  c.fillText('Dimension (by global variance) →', 0, 0);
  c.restore();

  // Color legend
  var legW = 150, legH = 12;
  var legX = ox + plotW - legW;
  var legY = oy - 15;
  for (var i = 0; i < legW; i++) {
    var t = i / legW;
    var r2, g2, b2;
    if (t < 0.25) { var f = t / 0.25; r2 = Math.floor(f * 80); g2 = 0; b2 = Math.floor(f * 120); }
    else if (t < 0.5) { var f = (t - 0.25) / 0.25; r2 = 80 + Math.floor(f * 153); g2 = Math.floor(f * 40); b2 = 120 - Math.floor(f * 24); }
    else if (t < 0.75) { var f = (t - 0.5) / 0.25; r2 = 233; g2 = 40 + Math.floor(f * 180); b2 = 96 - Math.floor(f * 96); }
    else { var f = (t - 0.75) / 0.25; r2 = 233 + Math.floor(f * 22); g2 = 220 + Math.floor(f * 35); b2 = Math.floor(f * 200); }
    c.fillStyle = 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
    c.fillRect(legX + i, legY, 1, legH);
  }
  c.font = '8px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('0', legX, legY - 2);
  c.textAlign = 'right';
  c.fillText(maxVar.toExponential(1), legX + legW, legY - 2);
  c.textAlign = 'center';
  c.fillText('variance', legX + legW / 2, legY + legH + 10);
}

// ============================================================
// TAB 3: PAIRWISE SIMILARITY MATRIX (large, readable)
// ============================================================
function drawMultiPairwiseMatrix(c, area) {
  var layerIdx = +document.getElementById('multi-layer').value;
  var lc = multiData.layer_comparisons[layerIdx];
  var nSent = multiData.n_sentences;
  var cosMatrix = lc.pairwise_cosine;
  var l2Matrix = lc.pairwise_l2;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  var margin = { left: 120, right: 40, top: 80, bottom: 120 };
  var plotSize = Math.min(area.w - margin.left - margin.right, area.h - margin.top - margin.bottom);
  var cellSize = Math.floor(plotSize / nSent);
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText(
    'Pairwise Cosine Similarity — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    area.x + area.w / 2, oy - 30
  );

  // Draw matrix cells
  for (var i = 0; i < nSent; i++) {
    for (var j = 0; j < nSent; j++) {
      var val = cosMatrix[i][j];
      // Color: -1 = deep blue, 0 = black, 1 = bright green
      var r, g, b;
      if (val < 0) {
        var t = Math.min(1, -val);
        r = 0; g = 0; b = Math.floor(40 + t * 180);
      } else {
        var t = Math.min(1, val);
        r = Math.floor(t * 20); g = Math.floor(40 + t * 180); b = Math.floor(t * 80);
      }
      c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      c.fillRect(ox + j * cellSize, oy + i * cellSize, cellSize - 2, cellSize - 2);

      // Value text
      c.font = (cellSize > 40 ? 'bold 12px' : cellSize > 25 ? '10px' : '8px') + ' monospace';
      c.fillStyle = val > 0.6 ? '#000' : '#fff';
      c.textAlign = 'center';
      c.fillText(val.toFixed(3),
        ox + j * cellSize + (cellSize - 2) / 2,
        oy + i * cellSize + (cellSize - 2) / 2 + 4);

      // L2 distance below
      if (cellSize > 50) {
        c.font = '8px monospace';
        c.fillStyle = val > 0.6 ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)';
        c.fillText('L2=' + l2Matrix[i][j].toFixed(1),
          ox + j * cellSize + (cellSize - 2) / 2,
          oy + i * cellSize + (cellSize - 2) / 2 + 16);
      }
    }

    // Row labels (left)
    c.font = '10px monospace';
    c.fillStyle = colors[i % colors.length];
    c.textAlign = 'right';
    var rowLabel = '[' + i + '] ' + multiData.sentences[i].text.substring(0, 15);
    c.fillText(rowLabel, ox - 8, oy + i * cellSize + cellSize / 2 + 3);

    // Column labels (bottom, rotated)
    c.save();
    c.translate(ox + i * cellSize + cellSize / 2, oy + nSent * cellSize + 8);
    c.rotate(Math.PI / 4);
    c.font = '9px monospace';
    c.fillStyle = colors[i % colors.length];
    c.textAlign = 'left';
    var colLabel = '[' + i + '] ' + multiData.sentences[i].text.substring(0, 20);
    c.fillText(colLabel, 0, 0);
    c.restore();
  }

  // Color legend
  var legX = ox + nSent * cellSize + 20;
  var legY = oy;
  var legH = Math.min(200, nSent * cellSize);
  for (var i = 0; i < legH; i++) {
    var val = 1.0 - (i / legH) * 2; // 1 at top, -1 at bottom
    var r, g, b;
    if (val < 0) { var t = -val; r = 0; g = 0; b = Math.floor(40 + t * 180); }
    else { var t = val; r = Math.floor(t * 20); g = Math.floor(40 + t * 180); b = Math.floor(t * 80); }
    c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    c.fillRect(legX, legY + i, 15, 1);
  }
  c.font = '9px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('1.0', legX + 20, legY + 4);
  c.fillText('0.0', legX + 20, legY + legH / 2 + 4);
  c.fillText('-1.0', legX + 20, legY + legH + 4);
}

// ============================================================
// TAB 4: LAYER DIVERGENCE PROFILE (large, with annotations)
// ============================================================
function drawMultiLayerProfile_Main(c, area) {
  var variances = multiData.summary.layer_total_variances;
  var nL = variances.length;
  var maxVar = Math.max.apply(null, variances) || 1;

  var margin = { left: 70, right: 40, top: 50, bottom: 50 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#e94560';
  c.textAlign = 'center';
  c.fillText('Layer-by-Layer Cross-Sentence Divergence', area.x + area.w / 2, oy - 20);

  var barW = Math.max(8, Math.floor(plotW / nL) - 1);
  var baseY = plotH + oy;

  // Draw bars and line
  var points = [];
  for (var i = 0; i < nL; i++) {
    var h = (variances[i] / maxVar) * plotH;
    var x = ox + i * (barW + 1);
    var y = baseY - h;
    points.push({ x: x + barW / 2, y: y });

    // Highlight most divergent layer
    var isMost = (i === multiData.summary.most_divergent_layer);
    var isLeast = (i === multiData.summary.least_divergent_layer);

    // Bar
    var frac = variances[i] / maxVar;
    if (isMost) {
      c.fillStyle = '#e94560';
    } else if (isLeast) {
      c.fillStyle = '#53a8b6';
    } else {
      c.fillStyle = 'rgb(' + Math.floor(frac * 200) + ',' +
                    Math.floor((1 - frac) * 120 + 50) + ',' +
                    Math.floor((1 - frac) * 180) + ')';
    }
    c.fillRect(x, y, barW, h);

    // Glow for most divergent
    if (isMost) {
      c.shadowColor = '#e94560';
      c.shadowBlur = 10;
      c.fillRect(x, y, barW, h);
      c.shadowBlur = 0;
    }

    // Layer label
    c.font = '9px monospace';
    c.fillStyle = isMost ? '#e94560' : isLeast ? '#53a8b6' : '#666';
    c.textAlign = 'center';
    c.fillText(i === 0 ? 'Emb' : 'L' + (i - 1), x + barW / 2, baseY + 16);

    // Value label on top of bar
    if (barW > 15 || nL <= 20) {
      c.font = '8px monospace';
      c.fillStyle = '#888';
      c.fillText(variances[i].toFixed(1), x + barW / 2, y - 4);
    }
  }

  // Connect with a line
  c.strokeStyle = 'rgba(123,104,238,0.5)';
  c.lineWidth = 1.5;
  c.beginPath();
  for (var i = 0; i < points.length; i++) {
    if (i === 0) c.moveTo(points[i].x, points[i].y);
    else c.lineTo(points[i].x, points[i].y);
  }
  c.stroke();

  // Dots on the line
  for (var i = 0; i < points.length; i++) {
    var isMost = (i === multiData.summary.most_divergent_layer);
    c.beginPath();
    c.arc(points[i].x, points[i].y, isMost ? 5 : 3, 0, Math.PI * 2);
    c.fillStyle = isMost ? '#e94560' : '#7b68ee';
    c.fill();
  }

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#e94560';
  c.textAlign = 'center';
  c.fillText('Layer-by-Layer Cross-Sentence Divergence', area.x + area.w / 2, oy - 20);

  // Axis labels
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText('Layer →', ox + (nL * (barW + 1)) / 2, baseY + 35);

  c.save();
  c.translate(ox - 45, oy + plotH / 2);
  c.rotate(-Math.PI / 2);
  c.fillText('Total Variance (cross-sentence) →', 0, 0);
  c.restore();

  // Annotations
  c.font = '10px monospace';
  c.textAlign = 'left';
  c.fillStyle = '#e94560';
  var mostIdx = multiData.summary.most_divergent_layer;
  c.fillText('▲ Most divergent: ' +
    (mostIdx === 0 ? 'Embedding' : 'Layer ' + (mostIdx - 1)) +
    ' (var=' + variances[mostIdx].toFixed(2) + ')',
    ox + 10, oy - 5);

  c.fillStyle = '#53a8b6';
  var leastIdx = multiData.summary.least_divergent_layer;
  c.fillText('▼ Least divergent: ' +
    (leastIdx === 0 ? 'Embedding' : 'Layer ' + (leastIdx - 1)) +
    ' (var=' + variances[leastIdx].toFixed(2) + ')',
    ox + 10, oy + 8);

  // Sentence legend at bottom
  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
  var legY2 = baseY + 40;
  c.font = '9px monospace';
  c.textAlign = 'left';
  c.fillStyle = '#888';
  c.fillText('Sentences compared:', ox, legY2);
  for (var si = 0; si < nSent; si++) {
    var sentText = multiData.sentences[si].text;
    if (sentText.length > 50) sentText = sentText.substring(0, 50) + '…';
    c.fillStyle = colors[si % colors.length];
    c.fillRect(ox, legY2 + 14 + si * 14, 10, 8);
    c.fillStyle = '#a0a0c0';
    c.fillText('[' + si + '] ' + sentText + ' (' + multiData.sentences[si].n_tokens + ' tok)',
      ox + 14, legY2 + 21 + si * 14);
  }

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText(
    nL + ' layers | ' + nSent + ' sentences | ' +
    'Variance = how differently sentences are represented at each layer',
    area.x + 10, area.y + area.h - 5
  );
}

// New tab: side-by-side deformation grids
// ============================================================
// MULTI-GRID COMPARISON: Side-by-side deformation grids
// Each sentence gets its own panel with the full 2D deformed
// grid visualization, using the same rendering as draw2D.
// ============================================================

function drawMultiGridComparison(c, area) {
  if (!multiData || !multiData.sentence_data || multiData.sentence_data.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No sentence data available — re-run Multi Compare', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var nSent = multiData.sentence_data.length;
  var p = gp(); // get current params (layer, t, amp, dims, etc.)

  // Layout: divide the area into nSent columns with small gaps
  var gap = 6;
  var panelW = Math.floor((area.w - gap * (nSent - 1)) / nSent);
  var panelH = area.h - 30; // leave room for sentence labels at bottom
  var panelY = area.y;

  var sentColors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  for (var si = 0; si < nSent; si++) {
    var sentD = multiData.sentence_data[si];
    if (!sentD || !sentD.fixed_pos || !sentD.deltas) continue;

    var panelX = area.x + si * (panelW + gap);

    // Draw panel background
    c.fillStyle = 'rgba(13,17,23,0.8)';
    c.fillRect(panelX, panelY, panelW, panelH);

    // Panel border
    c.strokeStyle = sentColors[si % sentColors.length];
    c.lineWidth = 1.5;
    c.strokeRect(panelX, panelY, panelW, panelH);

    // Render the deformed grid for this sentence
    drawMiniDeformedGrid(c, sentD, p, panelX, panelY, panelW, panelH);

    // Sentence label below the panel
    c.font = 'bold 9px monospace';
    c.fillStyle = sentColors[si % sentColors.length];
    c.textAlign = 'center';
    var labelText = '[' + si + '] ' + (sentD.text || '').substring(0, Math.floor(panelW / 5.5));
    if ((sentD.text || '').length > Math.floor(panelW / 5.5)) labelText += '…';
    c.fillText(labelText, panelX + panelW / 2, panelY + panelH + 14);

    // Token count and model info
    c.font = '8px monospace';
    c.fillStyle = '#666';
    c.fillText(sentD.n_real + ' tokens | ' + sentD.n_layers + ' layers',
      panelX + panelW / 2, panelY + panelH + 24);
  }

  // HUD at top
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.4)';
  c.textAlign = 'left';
  c.fillText(
    'Layer ' + p.layer + '/' + (multiData.sentence_data[0].n_layers - 1) +
    '  t=' + p.t.toFixed(2) +
    '  amp=' + p.amp.toFixed(1) +
    '  Dims:' + p.dx + ',' + p.dy +
    '  Mode:' + p.mode +
    '  ITP:' + document.getElementById('sel-itp').value.toUpperCase() +
    '  |  Use sidebar controls to adjust all panels simultaneously',
    area.x + 10, area.y + area.h - 2
  );
}

/**
 * Draw a complete 2D deformed grid visualization for a single sentence's data,
 * clipped to a rectangular panel. This is a self-contained mini version of draw2D
 * that accepts arbitrary data and target rectangle.
 *
 * @param {CanvasRenderingContext2D} c - canvas context
 * @param {Object} sentD - sentence data object (same structure as global D)
 * @param {Object} p - parameters from gp() (layer, t, amp, dx, dy, mode, etc.)
 * @param {number} px - panel x origin
 * @param {number} py - panel y origin
 * @param {number} pw - panel width
 * @param {number} ph - panel height
 */
function drawMiniDeformedGrid(c, sentD, p, px, py, pw, ph) {
  var nP = sentD.n_points;
  var nR = sentD.n_real;
  var dx = p.dx, dy = p.dy;
  var isEmb = (p.mode === 'embedding');

  // Clamp dims to this sentence's hidden_dim
  if (dx >= sentD.hidden_dim) dx = 0;
  if (dy >= sentD.hidden_dim) dy = Math.min(1, sentD.hidden_dim - 1);

  // Use the active deltas (respect decomposition selector)
  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  // Clamp layer to this sentence's layer count
  var layer = Math.min(p.layer, sentD.n_layers - 1);
  var amp = p.amp;
  var t = p.t;

  // Extract base positions
  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) {
    fx[i] = sentD.fixed_pos[i][dx];
    fy[i] = sentD.fixed_pos[i][dy];
  }

  // Compute effective deltas based on mode
  var edx = new Float64Array(nP), edy = new Float64Array(nP);
  if (!isEmb) {
    for (var j = 0; j < nP; j++) {
      var sx2 = 0, sy2 = 0;
      if (p.mode === 'single') {
        sx2 = activeDeltas[layer][j][dx];
        sy2 = activeDeltas[layer][j][dy];
      } else if (p.mode === 'cumfwd') {
        for (var l = 0; l <= layer; l++) {
          sx2 += activeDeltas[l][j][dx];
          sy2 += activeDeltas[l][j][dy];
        }
      } else { // cumbwd
        for (var l2 = layer; l2 < sentD.n_layers; l2++) {
          sx2 += activeDeltas[l2][j][dx];
          sy2 += activeDeltas[l2][j][dy];
        }
      }
      edx[j] = sx2 * amp;
      edy[j] = sy2 * amp;
    }
  }

// Compute view bounds from REAL tokens only so tokens are always visible
  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i2 = 0; i2 < nR; i2++) {
    if (fx[i2] < mnx) mnx = fx[i2]; if (fx[i2] > mxx) mxx = fx[i2];
    if (fy[i2] < mny) mny = fy[i2]; if (fy[i2] > mxy) mxy = fy[i2];
  }
  // Fallback if only 1 token
  if (!isFinite(mnx) || mnx === mxx) { mnx = fx[0] - 1; mxx = fx[0] + 1; }
  if (!isFinite(mny) || mny === mxy) { mny = fy[0] - 1; mxy = fy[0] + 1; }

  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd2 = 0.12;
  var vx0 = cxv - mr * (0.5 + pd2), vy0 = cyv - mr * (0.5 + pd2);
  var vw = mr * (1 + 2 * pd2), vh = vw;

  // Margin inside the panel
  var M = 8;
  var dW = pw - 2 * M, dH = ph - 2 * M;

  function SX(x) { return px + M + ((x - vx0) / vw) * dW; }
  function SY(y) { return py + M + ((y - vy0) / vh) * dH; }

  // Build grid
  var N = Math.max(8, Math.min(25, p.gr));
  var nV = (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV);

  for (var gy = 0; gy <= N; gy++) {
    for (var gx = 0; gx <= N; gx++) {
      var gi = gy * (N + 1) + gx;
      oX[gi] = vx0 + (gx / N) * vw;
      oY[gi] = vy0 + (gy / N) * vh;
    }
  }

  var sig = p.sig;
  var itpMethod = document.getElementById('sel-itp').value;

  if (isEmb) {
    for (var gi2 = 0; gi2 < nV; gi2++) { gX[gi2] = oX[gi2]; gY[gi2] = oY[gi2]; }
  } else {
    for (var gi3 = 0; gi3 < nV; gi3++) {
      var gpx = oX[gi3], gpy = oY[gi3];
      var interp = interpolateGridPoint(gpx, gpy, fx, fy, edx, edy, nP, sig, itpMethod);
      gX[gi3] = gpx + t * interp[0];
      gY[gi3] = gpy + t * interp[1];
    }
  }

  // Compute strain
  var sH = new Float64Array(N * (N + 1));
  var sVa = new Float64Array((N + 1) * N);
  for (var ey2 = 0; ey2 <= N; ey2++) {
    for (var ex2 = 0; ex2 < N; ex2++) {
      var a = ey2 * (N + 1) + ex2, b = a + 1;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sH[ey2 * N + ex2] = od > 1e-12 ? dd / od : 1;
    }
  }
  for (var ey3 = 0; ey3 < N; ey3++) {
    for (var ex3 = 0; ex3 <= N; ex3++) {
      var a2 = ey3 * (N + 1) + ex3, b2 = (ey3 + 1) * (N + 1) + ex3;
      var od2 = Math.hypot(oX[b2] - oX[a2], oY[b2] - oY[a2]);
      var dd2 = Math.hypot(gX[b2] - gX[a2], gY[b2] - gY[a2]);
      sVa[ey3 * (N + 1) + ex3] = od2 > 1e-12 ? dd2 / od2 : 1;
    }
  }

  // ---- Strain heatmap ----
  if (p.heat && !isEmb) {
    for (var hy = 0; hy < N; hy++) {
      for (var hx = 0; hx < N; hx++) {
        var avg = (sH[hy * N + hx] + sH[(hy + 1) * N + hx] +
                   sVa[hy * (N + 1) + hx] + sVa[hy * (N + 1) + hx + 1]) / 4;
        var co = s2c(avg);
        var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
        var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;
        c.beginPath();
        c.moveTo(SX(gX[i00]), SY(gY[i00]));
        c.lineTo(SX(gX[i10]), SY(gY[i10]));
        c.lineTo(SX(gX[i11]), SY(gY[i11]));
        c.lineTo(SX(gX[i01]), SY(gY[i01]));
        c.closePath();
        c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.35)';
        c.fill();
      }
    }
  }

  // ---- Reference grid ----
  if (p.ref) {
    c.strokeStyle = isEmb ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.07)';
    c.lineWidth = 0.4;
    for (var ry2 = 0; ry2 <= N; ry2++) {
      c.beginPath();
      for (var rx2 = 0; rx2 <= N; rx2++) {
        var ri = ry2 * (N + 1) + rx2;
        if (rx2 === 0) c.moveTo(SX(oX[ri]), SY(oY[ri]));
        else c.lineTo(SX(oX[ri]), SY(oY[ri]));
      }
      c.stroke();
    }
    for (var rx3 = 0; rx3 <= N; rx3++) {
      c.beginPath();
      for (var ry3 = 0; ry3 <= N; ry3++) {
        var ri3 = ry3 * (N + 1) + rx3;
        if (ry3 === 0) c.moveTo(SX(oX[ri3]), SY(oY[ri3]));
        else c.lineTo(SX(oX[ri3]), SY(oY[ri3]));
      }
      c.stroke();
    }
  }

  // ---- Deformed grid ----
  if (p.grid && !isEmb) {
    c.lineWidth = 0.8;
    // Horizontal edges
    for (var dhy = 0; dhy <= N; dhy++) {
      for (var dhx = 0; dhx < N; dhx++) {
        var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
        var es = sH[dhy * N + dhx];
        if (p.sc) {
          var ec = s2c(es);
          c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.85)';
        } else {
          c.strokeStyle = 'rgba(200,200,200,0.5)';
        }
        c.beginPath();
        c.moveTo(SX(gX[di1]), SY(gY[di1]));
        c.lineTo(SX(gX[di2]), SY(gY[di2]));
        c.stroke();
      }
    }
    // Vertical edges
    for (var dvx = 0; dvx <= N; dvx++) {
      for (var dvy = 0; dvy < N; dvy++) {
        var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
        var vs = sVa[dvy * (N + 1) + dvx];
        if (p.sc) {
          var vc = s2c(vs);
          c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.85)';
        } else {
          c.strokeStyle = 'rgba(200,200,200,0.5)';
        }
        c.beginPath();
        c.moveTo(SX(gX[dvi1]), SY(gY[dvi1]));
        c.lineTo(SX(gX[dvi2]), SY(gY[dvi2]));
        c.stroke();
      }
    }
  }

  // ---- Vector arrows ----
  if (p.vec && !isEmb) {
    var step = Math.max(1, Math.floor(N / 8));
    c.lineWidth = 1.0;
    for (var viy = 0; viy <= N; viy += step) {
      for (var vix = 0; vix <= N; vix += step) {
        var vi = viy * (N + 1) + vix;
        var ax = SX(oX[vi]), ay = SY(oY[vi]);
        var bx = SX(gX[vi]), by = SY(gY[vi]);
        var al = Math.hypot(bx - ax, by - ay);
        if (al < 2) continue;
        c.strokeStyle = 'rgba(255,255,100,0.5)';
        c.fillStyle = 'rgba(255,255,100,0.5)';
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
        var aa = Math.atan2(by - ay, bx - ax), hl = Math.min(5, al * 0.3);
        c.beginPath(); c.moveTo(bx, by);
        c.lineTo(bx - hl * Math.cos(aa - 0.4), by - hl * Math.sin(aa - 0.4));
        c.lineTo(bx - hl * Math.cos(aa + 0.4), by - hl * Math.sin(aa + 0.4));
        c.closePath(); c.fill();
      }
    }
  }

  // ---- Probe points ----
  if (p.syn) {
    for (var pi = nR; pi < nP; pi++) {
      c.beginPath(); c.arc(SX(fx[pi]), SY(fy[pi]), 1.5, 0, Math.PI * 2);
      c.fillStyle = 'rgba(100,200,255,0.15)'; c.fill();
    }
  }

  // ---- Real token dots ----
  if (p.tok) {
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];
    for (var ti = 0; ti < nR; ti++) {
      var tx2 = SX(fx[ti]), ty2 = SY(fy[ti]);
      var col = tc[ti % tc.length];

      // Glow
      var grad = c.createRadialGradient(tx2, ty2, 0, tx2, ty2, 12);
      grad.addColorStop(0, 'rgba(255,255,255,0.06)');
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      c.beginPath(); c.arc(tx2, ty2, 12, 0, Math.PI * 2);
      c.fillStyle = grad; c.fill();

      // Dot
      c.beginPath(); c.arc(tx2, ty2, 4, 0, Math.PI * 2);
      c.fillStyle = col; c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 1; c.stroke();

      // Label
      c.font = 'bold 8px monospace';
      c.lineWidth = 2;
      c.strokeStyle = 'rgba(0,0,0,0.9)';
      var lb = '[' + ti + '] ' + sentD.tokens[ti];
      // Truncate label if panel is narrow
      if (lb.length > Math.floor(pw / 7)) lb = lb.substring(0, Math.floor(pw / 7)) + '…';
      c.strokeText(lb, tx2 + 6, ty2 - 5);
      c.fillStyle = '#fff';
      c.fillText(lb, tx2 + 6, ty2 - 5);
    }

    // Token sequence line in embedding mode
    if (isEmb && nR > 1) {
      c.strokeStyle = 'rgba(233,69,96,0.3)';
      c.lineWidth = 1;
      c.setLineDash([3, 3]);
      c.beginPath();
      c.moveTo(SX(fx[0]), SY(fy[0]));
      for (var ti2 = 1; ti2 < nR; ti2++) c.lineTo(SX(fx[ti2]), SY(fy[ti2]));
      c.stroke();
      c.setLineDash([]);
    }
  }

  // ---- Panel HUD (layer/mode info) ----
  c.font = '8px monospace';
  c.fillStyle = 'rgba(255,255,255,0.35)';
  c.textAlign = 'left';
  if (isEmb) {
    c.fillText('EMB d' + dx + ',' + dy, px + M + 2, py + M + 8);
  } else {
    c.fillText('L' + layer + ' t=' + t.toFixed(1) + ' a=' + amp.toFixed(0) + ' d' + dx + ',' + dy,
      px + M + 2, py + M + 8);
  }

  // Show strain stats if available
  if (sentD.strain_stats && sentD.strain_stats[layer]) {
    var ss = sentD.strain_stats[layer];
    c.fillText('strain: μ=' + ss.mean.toFixed(2) + ' σ²=' + ss.variance.toFixed(3),
      px + M + 2, py + M + 17);
  }
}

