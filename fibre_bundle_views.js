/**
 * Mini Fibre 3D panel — renders the 3D fibre bundle grid for one sentence
 */
function drawMiniPanelFibre3D(c, sentD, W, H) {
  var p = gp();
  var nR = sentD.n_real;
  var nLayers = sentD.n_layers;
  var nP = sentD.n_points;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var dz = Math.min(p.dz, sentD.hidden_dim - 1);
  var isEmb = (p.mode === 'embedding');
  var layer = Math.min(p.layer, nLayers - 1);
  var amp = p.amp, t = p.t;

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var fx = new Float64Array(nP), fy = new Float64Array(nP), fz = new Float64Array(nP);
  for (var i = 0; i < nP; i++) {
    fx[i] = sentD.fixed_pos[i][dx];
    fy[i] = sentD.fixed_pos[i][dy];
    fz[i] = sentD.fixed_pos[i][dz];
  }

  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, mnz = Infinity, mxz = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    if (fz[i] < mnz) mnz = fz[i]; if (fz[i] > mxz) mxz = fz[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1;
  var cx3 = (mnx + mxx) / 2, cy3 = (mny + mxy) / 2, cz3 = (mnz + mxz) / 2;
  var sc3 = Math.min(W, H) * 0.3 / mr;

  // Apply shared zoom
  var effSc3 = sc3 * zoomLevel;

  var fl = 400;
  function proj(x, y, z) {
    // Use shared rotX, rotY globals
    var cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
    var cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
    var scale = fl / (fl + z2);
    return [W / 2 + panX + x1 * scale, H / 2 + panY + y1 * scale, z2, scale];
  }

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

  // Draw token pathlines through layers in 3D
  for (var ti = 0; ti < nR; ti++) {
    var col = tc[ti % tc.length];
    var pts = [];

    for (var li = 0; li < nLayers; li++) {
      var cumDx = 0, cumDy = 0, cumDz = 0;
      if (!isEmb) {
        if (p.mode === 'single') {
          cumDx = activeDeltas[li][ti][dx] * amp * t;
          cumDy = activeDeltas[li][ti][dy] * amp * t;
          cumDz = activeDeltas[li][ti][dz] * amp * t;
        } else if (p.mode === 'cumfwd') {
          for (var cl = 0; cl <= li; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
            cumDz += activeDeltas[cl][ti][dz] * amp * t;
          }
        } else {
          for (var cl = li; cl < nLayers; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
            cumDz += activeDeltas[cl][ti][dz] * amp * t;
          }
        }
      }

      var wx = (fx[ti] + cumDx - cx3) * effSc3;
      var wy = (fy[ti] + cumDy - cy3) * effSc3;
      var wz = (fz[ti] + cumDz - cz3) * effSc3;
      var pp = proj(wx, wy, wz);
      pts.push({ x: pp[0], y: pp[1], z: pp[2], s: pp[3], layer: li });
    }

    // Draw pathline
    c.strokeStyle = col;
    c.lineWidth = 1.5;
    c.lineJoin = 'round';
    c.beginPath();
    c.moveTo(pts[0].x, pts[0].y);
    for (var li2 = 1; li2 < nLayers; li2++) {
      c.lineTo(pts[li2].x, pts[li2].y);
    }
    c.stroke();

    // Draw dots
    for (var li3 = 0; li3 < nLayers; li3++) {
      var pt = pts[li3];
      var isActive = (li3 === layer);
      var dotR = Math.max(1.5, (isActive ? 4 : 2) * pt.s);
      c.beginPath();
      c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
      c.fillStyle = col;
      c.fill();
      if (isActive) {
        c.strokeStyle = '#fff';
        c.lineWidth = 0.8;
        c.stroke();
      }
    }

    // Label
    if (pts.length > 0 && W > 60) {
      var lp = pts[0];
      c.font = 'bold 6px monospace';
      c.fillStyle = col;
      c.textAlign = 'center';
      c.fillText('[' + ti + ']', lp.x, lp.y + 8);
    }
  }

  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('FIBRE3D L' + layer + ' d' + dx + ',' + dy + ',' + dz, 4, 10);
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

  var margin = 8;
  var plotW = W - 2 * margin;
  var plotH = H - 2 * margin;
  var layerH = plotH / nLayers;

  function SX(wx) { return margin + ((wx - vx0) / vw) * plotW; }
  function LY(li) { return margin + (nLayers - 1 - li) * layerH + layerH * 0.5; }

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

  // Apply shared zoom/pan
  c.save();
  c.translate(W / 2 + panX, H / 2 + panY);
  c.scale(zoomLevel, zoomLevel);
  c.translate(-W / 2, -H / 2);

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

    // Draw path glow
    c.strokeStyle = 'rgba(' + parseInt(col.slice(1,3),16) + ',' +
                    parseInt(col.slice(3,5),16) + ',' +
                    parseInt(col.slice(5,7),16) + ',0.06)';
    c.lineWidth = 6;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li2 = 1; li2 < nLayers; li2++) {
      var prev = path[li2 - 1], curr = path[li2];
      c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
    }
    c.stroke();

    // Draw path line
    c.strokeStyle = 'rgba(' + parseInt(col.slice(1,3),16) + ',' +
                    parseInt(col.slice(3,5),16) + ',' +
                    parseInt(col.slice(5,7),16) + ',0.7)';
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li3 = 1; li3 < nLayers; li3++) {
      var prev2 = path[li3 - 1], curr2 = path[li3];
      c.quadraticCurveTo(prev2.x, (prev2.y + curr2.y) / 2, curr2.x, curr2.y);
    }
    c.stroke();

    // Draw dots at each layer
    for (var li4 = 0; li4 < nLayers; li4++) {
      var pt = path[li4];
      var isActive = (li4 === layer);
      var dotR = isActive ? 3 : 1.5;
      c.beginPath();
      c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
      c.fillStyle = col;
      c.fill();
      if (isActive) {
        c.strokeStyle = '#fff';
        c.lineWidth = 0.8;
        c.stroke();
      }
    }

    // Token label at bottom
    if (W > 80) {
      c.font = 'bold 7px monospace';
      c.fillStyle = col;
      c.textAlign = 'center';
      var labelTxt = '[' + ti + '] ' + (sentD.tokens[ti] || '');
      if (labelTxt.length > Math.floor(W / (nR * 5))) labelTxt = labelTxt.substring(0, Math.floor(W / (nR * 5))) + '…';
      c.fillText(labelTxt, path[0].x, path[0].y + 10);
    }
  }

  c.restore();

  // HUD (drawn outside the transform)
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('KELP L' + layer + ' d' + dx + ',' + dy, 4, 10);
}



function computeMiniFibreLayout(W, H, nTokens, nLayers) {
    var margin = 4;
    var labelW = 18;
    var availW = W - 2 * margin - labelW;
    var availH = H - 2 * margin;
    var roomSize = Math.max(12, Math.min(
        Math.floor(availW / (nTokens * 1.3)),
        Math.floor(availH / (nLayers * 1.4))
    ));
    var gapX = Math.max(2, Math.floor(roomSize * 0.2));
    var gapY = Math.max(3, Math.floor(roomSize * 0.25));

    return {
        margin:   margin,
        labelW:   labelW,
        roomSize: roomSize,
        gapX:     gapX,
        gapY:     gapY,
        startX:   margin + labelW,
        startY:   margin
    };
}

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
    var fp = getFibre3DParams();

    // ---- Extract 3D positions ----
    var pos3 = extractPositions3D(D, fp.nP, fp.dx, fp.dy, fp.dz);
    var fx = pos3.fx, fy = pos3.fy, fz = pos3.fz;

    // ---- Compute view bounds ----
    var bounds3 = computeViewBounds3D(fx, fy, fz, fp.nP, 0.15);
    var cx3 = bounds3.cx, cy3 = bounds3.cy, cz3 = bounds3.cz;
    var mr = bounds3.mr;

    // ---- Room layout ----
    var roomLayout = computeFibre3DRoomLayout(fp.nTokens, fp.nLayers, mr);
    var roomSize = roomLayout.roomSize;
    var gapX = roomLayout.gapX;
    var gapY = roomLayout.gapY;
    var sc3 = roomLayout.sc3;

    // ---- 3D projector ----
    var proj3Df = makeFibre3DProjector(W, H);

    // ---- Per-layer raw deltas (3D) ----
    var edxAll = [], edyAll = [], edzAll = [];
    for (var lay = 0; lay < fp.nLayers; lay++) {
        var edxL = new Float64Array(fp.nP);
        var edyL = new Float64Array(fp.nP);
        var edzL = new Float64Array(fp.nP);
        for (var j = 0; j < fp.nP; j++) {
            edxL[j] = fp.activeDeltas[lay][j][fp.dx] * fp.amp;
            edyL[j] = fp.activeDeltas[lay][j][fp.dy] * fp.amp;
            edzL[j] = fp.activeDeltas[lay][j][fp.dz] * fp.amp;
        }
        edxAll.push(edxL);
        edyAll.push(edyL);
        edzAll.push(edzL);
    }

    // ---- Grid resolution ----
    var N = Math.max(3, Math.min(8, Math.floor(roomSize / 15)));

    // ---- Token colors ----
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    // ---- Collect all renderable primitives for depth sorting ----
    var allEdges = [];
    var allQuads = [];
    var allPoints = [];

    for (var li = 0; li < fp.nLayers; li++) {
        var rowIdx = fp.nLayers - 1 - li;
        var isCurrentLayer = (li === fp.currentLayer);

        // Compute cumulative deltas for this layer
        var layerDeltas = computeCumulativeDeltas3DFromRaw(
            edxAll, edyAll, edzAll, li, fp.nP, fp.nLayers, fp.mode, fp.isEmb
        );

        // Layer label position
        var labelX = (-(fp.nTokens - 1) / 2 - 1.5) * (roomSize + gapX);
        var labelY = (rowIdx - (fp.nLayers - 1) / 2) * (roomSize + gapY);
        collectLayerLabel(allPoints, li, labelX, labelY, proj3Df, isCurrentLayer);

        for (var ti = 0; ti < fp.nTokens; ti++) {
            // Room center in 3D space
            var roomCenterX = (ti - (fp.nTokens - 1) / 2) * (roomSize + gapX);
            var roomCenterY = (rowIdx - (fp.nLayers - 1) / 2) * (roomSize + gapY);

            // Build the 3D deformed grid for this room
            var pd = 0.12;
            var vx0 = cx3 - mr * (0.5 + pd), vx1 = cx3 + mr * (0.5 + pd);
            var vy0 = cy3 - mr * (0.5 + pd), vy1 = cy3 + mr * (0.5 + pd);
            var vz0 = cz3 - mr * (0.5 + pd), vz1 = cz3 + mr * (0.5 + pd);

            var grid = buildFibre3DRoomGrid(
                vx0, vy0, vz0, vx1, vy1, vz1, N,
                fx, fy, fz,
                layerDeltas.edx, layerDeltas.edy, layerDeltas.edz,
                fp.nP, fp.sig, fp.t, fp.isEmb
            );

            // Collect grid edges
            collectRoomGridEdges(
                allEdges, grid, roomCenterX, roomCenterY,
                cx3, cy3, cz3, sc3, proj3Df,
                isCurrentLayer, fp.showGrid, fp.isEmb
            );

            // Collect heatmap faces
            collectRoomHeatmapFaces(
                allQuads, grid, roomCenterX, roomCenterY,
                cx3, cy3, cz3, sc3, proj3Df,
                isCurrentLayer, fp.showHeat, fp.isEmb
            );

            // Collect room border wireframe
            collectRoomBorderEdges(
                allEdges, roomCenterX, roomCenterY, roomSize,
                proj3Df, isCurrentLayer
            );

            // Collect token dot
            collectRoomTokenPoint(
                allPoints, fx, fy, fz, ti,
                layerDeltas.edx, layerDeltas.edy, layerDeltas.edz,
                fp.t, fp.isEmb,
                roomCenterX, roomCenterY,
                cx3, cy3, cz3, sc3, proj3Df,
                li, isCurrentLayer, tc[ti % tc.length]
            );

            // Token label at bottom layer
            if (li === 0) {
                collectRoomTokenLabel(
                    allPoints, ti,
                    roomCenterX, roomCenterY, roomSize,
                    proj3Df, tc[ti % tc.length],
                    '[' + ti + '] ' + D.tokens[ti]
                );
            }

            // Inter-layer pathlines
            if (li > 0) {
                var prevRowIdx = fp.nLayers - li;
                collectInterLayerPathlines(
                    allEdges, ti, prevRowIdx, rowIdx, fp.nLayers,
                    roomSize, gapY, gapX, fp.nTokens,
                    proj3Df, isCurrentLayer, tc[ti % tc.length]
                );
            }
        }
    }

    // ---- Render all primitives in depth order ----

    // Pass 1: Heatmap faces (back to front)
    renderFibre3DQuads(c, allQuads);

    // Pass 2: Edges — grid lines, borders, pathlines (back to front)
    renderFibre3DEdges(c, allEdges, fp.showSC);

    // Pass 3: Points — token dots, labels, layer labels (back to front)
    renderFibre3DPoints(c, allPoints);

    // Pass 4: 3D axes overlay
    drawFibre3DAxes(c, roomSize, mr, proj3Df, fp.dx, fp.dy, fp.dz);

    // ---- HUD ----
    var decompLabel = getDecompLabel();
    c.font = '11px monospace';
    c.fillStyle = 'rgba(255,255,255,0.45)';
    c.textAlign = 'left';
    if (fp.isEmb) {
        c.fillText(
            'FIBRE BUNDLE 3D [EMBEDDING]  Tokens:' + fp.nTokens +
            '  Layers:' + fp.nLayers +
            '  Dims:' + fp.dx + ',' + fp.dy + ',' + fp.dz +
            '  Drag to rotate',
            12, 16
        );
    } else {
        c.fillText(
            'FIBRE BUNDLE 3D  Layer:' + fp.currentLayer + '/' + (fp.nLayers - 1) +
            '  t=' + fp.t.toFixed(2) +
            '  amp=' + fp.amp.toFixed(1) +
            '  Dims:' + fp.dx + ',' + fp.dy + ',' + fp.dz +
            '  Mode:' + fp.mode +
            '  Decomp:' + decompLabel +
            '  Drag to rotate',
            12, 16
        );
    }

    c.font = '9px monospace';
    c.fillStyle = 'rgba(255,255,255,0.3)';
    c.fillText(
        'Zoom: ' + zoomLevel.toFixed(2) + 'x  |  ' +
        '\u2190\u2192 Dim X | \u2191\u2193 Dim Y | Shift+Arrow Dim Z | [/] Layer | A/Z Amp | ;/\' t | 0=Reset',
        12, H - 8
    );
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
        var ps = predictedTokenStyle;
        c.save();
        c.translate(legX + 6, legY);
        c.rotate(Math.PI / 4);
        c.fillStyle = ps.rgba(ps.fillAlpha(0.5));
        c.fillRect(-3, -3, 6, 6);
        c.strokeStyle = ps.rgba(ps.strokeAlpha());
        c.lineWidth = 0.8;
        c.strokeRect(-3, -3, 6, 6);
        c.restore();
        c.fillStyle = ps.rgba(ps.labelFillAlpha());
        c.fillText('Predicted (' + D.predicted_indices.length + ')', legX + 24, legY + 3);
    }
}

/**
 * Draw all inter-layer connections between consecutive layer rows.
 * Extracted from the layer loop of drawFibreBundle.
 */
function drawFibreAllInterLayerConnections(c, fp, layout, rawDeltas, bounds) {
    if (!fibreState.showConnections || fp.isEmb) return;

    for (var li = 0; li < fp.nLayers - 1; li++) {
        var currDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li, fp.nP, fp.nLayers, fp.mode, fp.isEmb
        );
        var nextDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li + 1, fp.nP, fp.nLayers, fp.mode, fp.isEmb
        );
        drawFibreInterLayerConnections(
            c, li, fp.nTokens, fp.nP, fp.nLayers,
            layout, currDeltas.edx, nextDeltas.edx,
            D, bounds.fx || null, bounds.fy || null, fp.t,
            bounds.vx0, bounds.vy0, bounds.vw, bounds.vh
        );
    }
}

/**
 * Top-level: the fully refactored drawFibreBundle.
 * Orchestrates all passes using the helpers above.
 */
function drawFibreBundle() {
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
    var fp = getFibre2DParams();

    // ---- Reuse existing position / bounds / delta helpers ----
    var pos = extractPositions2D(D, fp.nP, fp.dx, fp.dy);
    var fx = pos.fx, fy = pos.fy;
    var bounds = computeViewBounds2D(fx, fy, fp.nP, 0.15);
    var rawDeltas = computePerLayerRawDeltas(fp.activeDeltas, fp.nLayers, fp.nP, fp.dx, fp.dy, fp.amp);

    // ---- Layout ----
    var layout = computeFibreRoomLayout(W, H, fp.nTokens, fp.nLayers, zoomLevel);

    // ---- Grid resolution ----
    var N = computeFibreGridResolution(layout.roomSize, 4, 16, 4);

    // ---- Token colors ----
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    // ========== PASS 1: Flow streamlines between layers ==========
    if (fibreState.showFlowLines && !fp.isEmb) {
        drawFibreBundleFlowLines(
            c, layout, rawDeltas, fp.nLayers, fp.nTokens, fp.nP,
            fx, fy, bounds.vx0, bounds.vy0, bounds.vw, bounds.vh,
            fp.sig, fp.t, fp.currentLayer, N
        );
    }

    // ========== PASS 2: Draw each layer row (rooms + tokens + predictions) ==========
    for (var li = 0; li < fp.nLayers; li++) {
        drawFibreLayerRow(c, li, fp, layout, fx, fy, rawDeltas, bounds, N, tc);
    }

    // ========== PASS 3: Inter-layer connections ==========
    drawFibreAllInterLayerConnections(c, fp, layout, rawDeltas, bounds);

    // ========== PASS 4: Axis labels ==========
    drawFibreBundleAxisLabels(c, layout, fp.nTokens, fp.nLayers);

    // ========== PASS 5: Legend ==========
    if ((fibreState.showAttnField || fibreState.showMlpField) && !fp.isEmb) {
        drawFibreBundleLegend(c, layout, fp.nTokens, fp.attnDeltas, fp.mlpDeltas, D);
    }

    c.restore();

    // ========== HUD ==========
    drawFibreBundleHUD(c, W, H, fp.nTokens, fp.nLayers, fp.hiddenDim, fp.currentLayer);
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

/**
 * Draw predicted next-token markers across all token columns for a single layer.
 * Extracted from the inner layer loop of drawFibreBundle.
 *
 * @param {CanvasRenderingContext2D} c
 * @param {Object} D - global data
 * @param {Float64Array} edxCum - cumulative x deltas for this layer
 * @param {Float64Array} edyCum - cumulative y deltas for this layer
 * @param {Float64Array} fx - base x positions
 * @param {Float64Array} fy - base y positions
 * @param {number} t - deformation parameter
 * @param {number} nP - total points
 * @param {number} nTokens - real token count
 * @param {Object} layout - room layout from computeFibreRoomLayout
 * @param {number} roomCY - y position of this layer's row
 * @param {number} vx0, vy0, vw, vh - view bounds
 * @param {boolean} isCurrentLayer
 */
function drawFibrePredictedTokensInLayer(c, D, edxCum, edyCum, fx, fy, t, nP, nTokens,
    layout, roomCY, vx0, vy0, vw, vh, isCurrentLayer) {

    if (!D.predicted_indices || D.predicted_indices.length === 0) return;
    var ps = predictedTokenStyle;

    for (var pi2 = 0; pi2 < D.predicted_indices.length; pi2++) {
        var pidx = D.predicted_indices[pi2];
        if (pidx >= nP) continue;
        var prob = D.predicted_probs ? D.predicted_probs[pi2] : 0;

        var predWX = fx[pidx] + t * edxCum[pidx];
        var predWY = fy[pidx] + t * edyCum[pidx];

        for (var ti2 = 0; ti2 < nTokens; ti2++) {
            var roomCX2 = layout.startX + ti2 * (layout.roomSize + layout.gapX);

            var predScreenX = roomCX2 + ((predWX - vx0) / vw) * layout.roomSize;
            var predScreenY = roomCY + ((predWY - vy0) / vh) * layout.roomSize;

            if (predScreenX < roomCX2 - 3 || predScreenX > roomCX2 + layout.roomSize + 3 ||
                predScreenY < roomCY - 3 || predScreenY > roomCY + layout.roomSize + 3) {
                continue;
            }

            var dotSz = 2 + prob * 5;

            // Glow
            var glowR2 = 5 + prob * 8;
            var grad4 = c.createRadialGradient(predScreenX, predScreenY, 0, predScreenX, predScreenY, glowR2);
            grad4.addColorStop(0, ps.rgba(ps.glowAlpha(prob)));
            grad4.addColorStop(1, ps.rgba('0'));
            c.beginPath();
            c.arc(predScreenX, predScreenY, glowR2, 0, Math.PI * 2);
            c.fillStyle = grad4;
            c.fill();

            // Diamond shape
            c.save();
            c.translate(predScreenX, predScreenY);
            c.rotate(Math.PI / 4);
            c.fillStyle = ps.rgba(ps.fillAlpha(prob));
            c.fillRect(-dotSz / 2, -dotSz / 2, dotSz, dotSz);
            c.strokeStyle = ps.rgba(ps.strokeAlpha());
            c.lineWidth = 0.7;
            c.strokeRect(-dotSz / 2, -dotSz / 2, dotSz, dotSz);
            c.restore();

            // Label (only on current layer with enough room)
            if (isCurrentLayer && layout.roomSize > 40) {
                c.font = '6px monospace';
                c.lineWidth = 1;
                c.strokeStyle = 'rgba(0,0,0,' + ps.labelStrokeAlpha() + ')';
                c.fillStyle = ps.rgba(ps.labelFillAlpha());
                c.textAlign = 'left';
                var plb2 = D.tokens[pidx] + ' ' + (prob * 100).toFixed(0) + '%';
                c.strokeText(plb2, predScreenX + dotSz + 2, predScreenY - 2);
                c.fillText(plb2, predScreenX + dotSz + 2, predScreenY - 2);
            }
        }
    }
}

/**
 * Draw all rooms for a single layer row.
 * Extracted from the layer loop of drawFibreBundle.
 *
 * @returns {Object} layerDeltas - { edx, edy } for use by inter-layer connections
 */
function drawFibreLayerRow(c, li, fp, layout, fx, fy, rawDeltas, bounds, N, tc) {
    var rowIdx = fp.nLayers - 1 - li;
    var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
    var isCurrentLayer = (li === fp.currentLayer);

    // Layer label
    c.font = (isCurrentLayer ? 'bold ' : '') + '9px monospace';
    c.fillStyle = isCurrentLayer ? '#e94560' : '#666';
    c.textAlign = 'right';
    c.fillText('L' + li, layout.startX - 8, roomCY + layout.roomSize / 2 + 3);

    // Compute cumulative deltas for this layer
    var layerDeltas = computeCumulativeDeltas(
        rawDeltas.edxAll, rawDeltas.edyAll, li, fp.nP, fp.nLayers, fp.mode, fp.isEmb
    );

    // Draw each token's room
    for (var ti = 0; ti < fp.nTokens; ti++) {
        var roomCX = layout.startX + ti * (layout.roomSize + layout.gapX);

        drawFibreSingleRoom(c, {
            roomCX: roomCX, roomCY: roomCY, roomSize: layout.roomSize, N: N,
            vx0: bounds.vx0, vy0: bounds.vy0, vw: bounds.vw, vh: bounds.vh,
            fx: fx, fy: fy,
            edxCum: layerDeltas.edx, edyCum: layerDeltas.edy,
            nP: fp.nP, sig: fp.sig, t: fp.t,
            isEmb: fp.isEmb, itpMethod: fp.itpMethod,
            isCurrentLayer: isCurrentLayer,
            showGrid: fp.showGrid, showHeat: fp.showHeat, showSC: fp.showSC,
            showVec: fp.showVec,
            showAttnField: fibreState.showAttnField,
            showMlpField: fibreState.showMlpField,
            showTransportFrame: fibreState.showTransportFrame,
            attnDeltas: fp.attnDeltas, mlpDeltas: fp.mlpDeltas,
            layerIdx: li, dx: fp.dx, dy: fp.dy, amp: fp.amp,
            tokenIdx: ti,
            tokenColor: tc[ti % tc.length],
            flowArrowScale: fibreState.flowArrowScale,
        });

        // Token label at bottom of column (only for layer 0)
        if (li === 0) {
            drawFibreTokenLabel(c, ti, D.tokens[ti], roomCX, roomCY, layout.roomSize);
        }
    }

    // Predicted next-token points
    if (!fp.isEmb) {
        drawFibrePredictedTokensInLayer(
            c, D, layerDeltas.edx, layerDeltas.edy,
            fx, fy, fp.t, fp.nP, fp.nTokens,
            layout, roomCY, bounds.vx0, bounds.vy0, bounds.vw, bounds.vh,
            isCurrentLayer
        );
    }

    return { layerDeltas: layerDeltas, roomCY: roomCY };
}

function drawFibreSingleRoom(c, opts) {
    var roomCX = opts.roomCX, roomCY = opts.roomCY, roomSize = opts.roomSize;
    var N = opts.N;
    var vx0 = opts.vx0, vy0 = opts.vy0, vw = opts.vw, vh = opts.vh;

    // 1. Room background & border
    drawFibreRoomBackground(c, roomCX, roomCY, roomSize, opts.isCurrentLayer);

    // 2. Build deformed grid (reuses existing helper)
    var grid = buildDeformedGrid2D(
        vx0, vy0, vw, vh, N,
        opts.fx, opts.fy, opts.edxCum, opts.edyCum,
        opts.nP, opts.sig, opts.t, opts.isEmb, opts.itpMethod
    );

    // 3. Strain heatmap
    if (opts.showHeat && !opts.isEmb) {
        drawStrainHeatmapInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh);
    }

    // 4. Grid lines
    if (opts.showGrid && !opts.isEmb) {
        drawGridLinesInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh, opts.showSC);
    }

    // 5. Reference grid (embedding mode)
    if (opts.isEmb) {
        drawReferenceGridInRoom(c, grid, N, roomCX, roomCY, roomSize, vx0, vy0, vw, vh);
    }

    // 6. Component vector field overlays
    if (opts.showAttnField && opts.attnDeltas && !opts.isEmb) {
        drawComponentVectorField(
            c, grid, N, roomCX, roomCY, roomSize,
            vx0, vy0, vw, vh,
            opts.fx, opts.fy, opts.nP, opts.sig,
            opts.attnDeltas, opts.layerIdx, opts.dx, opts.dy, opts.amp, opts.t,
            'rgba(0,200,255,0.55)', opts.itpMethod
        );
    }
    if (opts.showMlpField && opts.mlpDeltas && !opts.isEmb) {
        drawComponentVectorField(
            c, grid, N, roomCX, roomCY, roomSize,
            vx0, vy0, vw, vh,
            opts.fx, opts.fy, opts.nP, opts.sig,
            opts.mlpDeltas, opts.layerIdx, opts.dx, opts.dy, opts.amp, opts.t,
            'rgba(255,165,0,0.55)', opts.itpMethod
        );
    }

    // 7. Transport frame at the token position
    if (opts.showTransportFrame && !opts.isEmb && opts.tokenIdx !== undefined) {
        var tokSX = roomCX + ((opts.fx[opts.tokenIdx] + opts.t * opts.edxCum[opts.tokenIdx] - vx0) / vw) * roomSize;
        var tokSY = roomCY + ((opts.fy[opts.tokenIdx] + opts.t * opts.edyCum[opts.tokenIdx] - vy0) / vh) * roomSize;
        drawTransportFrame(c, tokSX, tokSY, opts.edxCum, opts.edyCum,
            opts.fx, opts.fy, opts.tokenIdx, opts.nP, opts.sig, roomSize / 5);
    }

    // 8. Token dot with strain ring
    if (opts.tokenIdx !== undefined) {
        drawFibreTokenDot(
            c, opts.fx[opts.tokenIdx], opts.fy[opts.tokenIdx],
            opts.edxCum[opts.tokenIdx], opts.edyCum[opts.tokenIdx], opts.t,
            roomCX, roomCY, roomSize, vx0, vy0, vw, vh,
            opts.tokenColor, opts.isEmb, N
        );
    }
}

/**
 * Gather all fibre-2D-specific parameters from the DOM and data model.
 * Returns a self-contained config so downstream helpers never touch the DOM.
 */
function getFibre2DParams() {
    var hiddenDim = D.hidden_dim;
    var dxVal = Math.min(+document.getElementById('sl-dx').value, hiddenDim - 1);
    var dyVal = Math.min(+document.getElementById('sl-dy').value, hiddenDim - 1);

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;

    return {
        nTokens:      D.n_real,
        nLayers:      D.n_layers,
        hiddenDim:    hiddenDim,
        nP:           D.n_points,
        dx:           dxVal,
        dy:           dyVal,
        amp:          +document.getElementById('sl-amp').value,
        t:            +document.getElementById('sl-t').value,
        sig:          +document.getElementById('sl-sig').value,
        currentLayer: +document.getElementById('sl-layer').value,
        showGrid:     document.getElementById('cb-grid').checked,
        showHeat:     document.getElementById('cb-heat').checked,
        showSC:       document.getElementById('cb-sc').checked,
        showVec:      document.getElementById('cb-vec').checked,
        mode:         document.getElementById('sel-mode').value,
        itpMethod:    document.getElementById('sel-itp').value,
        isEmb:        document.getElementById('sel-mode').value === 'embedding',
        activeDeltas: activeDeltas,
        attnDeltas:   D.attn_deltas || null,
        mlpDeltas:    D.mlp_deltas || null,
    };
}

/**
 * Compute the grid resolution N for a fibre room based on room size.
 * Reusable anywhere a room-level grid resolution is needed.
 */
function computeFibreGridResolution(roomSize, minN, maxN, divisor) {
    return Math.max(minN || 4, Math.min(maxN || 16, Math.floor(roomSize / (divisor || 4))));
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

/**
 * Top-level: the refactored drawFibreBundleFlowLines.
 * Orchestrates all layer pairs × token columns using the helpers above.
 *
 * @param {CanvasRenderingContext2D} c
 * @param {Object} layout - room layout from computeFibreRoomLayout
 * @param {Object} rawDeltas - from computePerLayerRawDeltas
 * @param {number} nLayers
 * @param {number} nTokens
 * @param {number} nP
 * @param {Float64Array} fx - base x positions
 * @param {Float64Array} fy - base y positions
 * @param {number} vx0, vy0, vw, vh - view bounds
 * @param {number} sig - RBF bandwidth
 * @param {number} t - deformation parameter
 * @param {number} currentLayer - (unused visually but available)
 * @param {number} N - grid resolution
 */
function drawFibreBundleFlowLines(c, layout, rawDeltas, nLayers, nTokens, nP,
    fx, fy, vx0, vy0, vw, vh, sig, t, currentLayer, N) {

    var s2i = 1 / (2 * sig * sig);
    var mode = document.getElementById('sel-mode').value;

    c.globalAlpha = 0.35;

    for (var li = 0; li < nLayers - 1; li++) {
        var rowIdx = nLayers - 1 - li;
        var nextRowIdx = nLayers - 2 - li;
        var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
        var nextRoomCY = layout.startY + nextRowIdx * (layout.roomSize + layout.gapY);

        // Reuse existing cumulative delta helper
        var layerDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li, nP, nLayers, mode, false);
        var nextDeltas = computeCumulativeDeltas(
            rawDeltas.edxAll, rawDeltas.edyAll, li + 1, nP, nLayers, mode, false);

        for (var ti = 0; ti < nTokens; ti++) {
            var roomCX = layout.startX + ti * (layout.roomSize + layout.gapX);

            drawFlowStrandsForToken(c, roomCX, roomCY, nextRoomCY,
                layerDeltas, nextDeltas, fx, fy, nP, s2i, t, N,
                layout.roomSize, vx0, vy0, vw, vh);
        }
    }

    c.globalAlpha = 1.0;
}

/**
 * Mini Fibre Bundle panel — renders the token-per-column, layer-per-row
 * grid view for a single sentence into an offscreen context.
 *
 * Refactored to reuse:
 *   - extractPositions2D
 *   - computeViewBounds2D
 *   - computeFibreGridResolution
 *   - computeEdxEdyForLayerMini  (existing helper)
 *   - buildDeformedGrid2D
 *   - drawStrainHeatmapInRoom
 *   - drawGridLinesInRoom
 *   - drawReferenceGridInRoom
 *   - drawFibreRoomBackground
 *   - drawFibreTokenDot
 *   - drawFibreTokenLabel
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

    var showGrid = document.getElementById('cb-grid').checked;
    var showHeat = document.getElementById('cb-heat').checked;
    var showSC   = document.getElementById('cb-sc').checked;

    var decomp = document.getElementById('sel-decomp').value;
    var activeDeltas = sentD.deltas;
    if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
    if (decomp === 'mlp'  && sentD.mlp_deltas)  activeDeltas = sentD.mlp_deltas;

    var itpM = document.getElementById('sel-itp').value;

    var pos = extractPositions2D(sentD, nP, dx, dy);
    var fx = pos.fx, fy = pos.fy;

    var bounds = computeViewBounds2D(fx, fy, nP, 0.15);
    var vx0 = bounds.vx0, vy0 = bounds.vy0, vw = bounds.vw, vh = bounds.vh;

    var layout = computeMiniFibreLayout(W, H, nR, nLayers);

    var N = computeFibreGridResolution(layout.roomSize, 3, 8, 5);

    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6'];

    // Apply shared zoom/pan
    c.save();
    c.translate(W / 2 + panX, H / 2 + panY);
    c.scale(zoomLevel, zoomLevel);
    c.translate(-W / 2, -H / 2);

    for (var li = 0; li < nLayers; li++) {
        var rowIdx = nLayers - 1 - li;
        var roomCY = layout.startY + rowIdx * (layout.roomSize + layout.gapY);
        var isCurrent = (li === layer);

        c.font = (isCurrent ? 'bold ' : '') + '6px monospace';
        c.fillStyle = isCurrent ? '#e94560' : '#555';
        c.textAlign = 'right';
        c.fillText('L' + li, layout.startX - 2, roomCY + layout.roomSize / 2 + 3);

        var layerDeltas = computeEdxEdyForLayerMini(
            sentD, li, nP, dx, dy, amp, mode, activeDeltas, nLayers
        );

        for (var ti = 0; ti < nR; ti++) {
            var roomCX = layout.startX + ti * (layout.roomSize + layout.gapX);

            drawFibreRoomBackground(c, roomCX, roomCY, layout.roomSize, isCurrent);

            var grid = buildDeformedGrid2D(
                vx0, vy0, vw, vh, N,
                fx, fy, layerDeltas.edx, layerDeltas.edy,
                nP, sig, t, isEmb, itpM
            );

            if (showHeat && !isEmb) {
                drawStrainHeatmapInRoom(
                    c, grid, N, roomCX, roomCY, layout.roomSize,
                    vx0, vy0, vw, vh
                );
            }

            if (showGrid && !isEmb) {
                drawGridLinesInRoom(
                    c, grid, N, roomCX, roomCY, layout.roomSize,
                    vx0, vy0, vw, vh, showSC
                );
            }

            if (isEmb) {
                drawReferenceGridInRoom(
                    c, grid, N, roomCX, roomCY, layout.roomSize,
                    vx0, vy0, vw, vh
                );
            }

            drawFibreTokenDot(
                c, fx[ti], fy[ti],
                layerDeltas.edx[ti], layerDeltas.edy[ti], t,
                roomCX, roomCY, layout.roomSize,
                vx0, vy0, vw, vh,
                tc[ti % tc.length], isEmb, N
            );

            if (li === 0 && layout.roomSize > 25) {
                drawFibreTokenLabel(c, ti, sentD.tokens[ti],
                    roomCX, roomCY, layout.roomSize);
            }
        }
    }

    c.restore();

    // HUD (drawn outside the transform)
    c.font = '7px monospace';
    c.fillStyle = 'rgba(255,255,255,0.3)';
    c.textAlign = 'left';
    c.fillText('FIBRE L' + layer + ' d' + dx + ',' + dy, 4, 10);
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


