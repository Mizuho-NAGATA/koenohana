// sketch.js — durationでサイズを変え、ランのような曲線的開き方をするバージョン

// ----- 調整定数 -----
const MIN_DUR = 0.3;      // 秒: これ以下は最小サイズとして扱う
const MAX_DUR = 12.0;     // 秒: これ以上は最大サイズにクランプ
const SIZE_SCALE_RANGE = [0.6, 2.2]; // durationがMIN..MAXでマップされる倍率
const ORGANICNESS = 0.9;  // 0..1 大きいほど強く曲線的に開く（ランっぽさ）
const NORMALIZE_CENTROID = 8000;
const SHARP_WEIGHTS = { centroid: 0.6, flatness: 0.25, zcr: 0.15 };

let audioCtx;
let masterGain;
let sampleBuffers = [];
let embeddings = [];
let palettes = {
  animals: ['#2B2D42','#8D99AE','#EF233C','#D90429','#FFD166'],
  pastel: ['#ffadad','#ffd6a5','#fdffb6','#caffbf','#9bf6ff'],
  deep: ['#0b2d4a','#104e8b','#1f8a70','#6bd3d5','#ffcf5c'],
  monochrome: ['#222222','#7f7f7f','#dcdcdc','#f6d365','#ffb199']
};

let ui = {};
let umapInstance;
let canvasW, canvasH;
let points = [];
let baseOpenDuration = 700;

function setup(){
  const container = document.getElementById('canvas-container');
  canvasW = container.clientWidth;
  canvasH = container.clientHeight;
  const cnv = createCanvas(canvasW, canvasH);
  cnv.parent('canvas-container');
  angleMode(DEGREES);
  noLoop();

  ui.fileInput = document.getElementById('file-input');
  ui.regen = document.getElementById('regen-btn');
  ui.paletteSelect = document.getElementById('palette-select');
  ui.customColors = document.getElementById('custom-colors');
  ui.col1 = document.getElementById('col1');
  ui.col2 = document.getElementById('col2');
  ui.col3 = document.getElementById('col3');
  ui.sizeScale = document.getElementById('size-scale');
  ui.jitter = document.getElementById('jitter');
  ui.export = document.getElementById('export-png');
  ui.status = document.getElementById('status');

  if (ui.fileInput) ui.fileInput.addEventListener('change', onFilesSelected);
  if (ui.regen) ui.regen.addEventListener('click', onGenerateClick);
  if (ui.paletteSelect) ui.paletteSelect.addEventListener('change', onPaletteChange);
  if (ui.export) ui.export.addEventListener('click', exportPNG);

  setStatus('ファイルを選択して「生成」を押してください');

  window.addEventListener('resize', () => {
    canvasW = container.clientWidth;
    canvasH = container.clientHeight;
    resizeCanvas(canvasW, canvasH);
    redraw();
  });
}

function initAudio(){
  try {
    if (!audioCtx){
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = 0.9;
      masterGain.connect(audioCtx.destination);
      setStatus('AudioContext を初期化しました');
    } else if (audioCtx.state === 'suspended'){
      audioCtx.resume().then(()=> setStatus('AudioContext resumed'));
    }
  } catch(e){
    console.error('Audio init error', e);
    setStatus('Audio 初期化に失敗（console を確認）');
  }
}

async function onFilesSelected(ev){
  const files = ev.target.files;
  if (!files || files.length === 0) return;
  setStatus('ファイル読み込み中...');
  sampleBuffers = [];
  try {
    const buffers = await loadFilesAsAudio(files);
    sampleBuffers = buffers.map((b,i)=>({ name: files[i].name || `file-${i+1}`, buffer: b, features:null, embedding:null }));
    setStatus(`${sampleBuffers.length} 件のファイルを読み込みました。次に「生成」を押してください`);
  } catch(e){
    console.error(e);
    setStatus('ファイル読み込みに失敗しました（console を確認）');
  }
}

async function loadFilesAsAudio(files){
  if (!audioCtx) initAudio();
  const out = [];
  for (let f of files){
    const ab = await f.arrayBuffer();
    const buf = await audioCtx.decodeAudioData(ab.slice(0));
    if (buf.numberOfChannels > 1){
      const mono = audioCtx.createBuffer(1, buf.length, buf.sampleRate);
      const dst = mono.getChannelData(0);
      const ch0 = buf.getChannelData(0);
      const ch1 = buf.getChannelData(1);
      for (let i=0;i<buf.length;i++) dst[i] = 0.5*( (ch0[i]||0) + (ch1[i]||0) );
      out.push(mono);
    } else out.push(buf);
  }
  return out;
}

async function onGenerateClick(){
  if (!sampleBuffers.length){
    setStatus('ファイルが選択されていません。まず wav を選んでください');
    return;
  }
  setStatus('特徴抽出と UMAP を実行中...');
  try {
    await analyzeAndEmbed();
    buildVisualPoints();
    redraw();
    setStatus('生成完了 — つぼみをクリックして再生＋開花');
  } catch(e){
    console.error(e);
    setStatus('生成中にエラーが発生しました（console を確認）');
  }
}

async function analyzeAndEmbed(){
  let featureMatrix = [];
  for (let s of sampleBuffers){
    const feats = await extractFeaturesFromBuffer(s.buffer);
    s.features = feats;
    featureMatrix.push(feats.vector);
  }

  if (window.UMAP){
    umapInstance = new window.UMAP({nComponents:2, nNeighbors: 8, minDist: 0.2});
    try {
      const embedding = umapInstance.fit(featureMatrix);
      embeddings = embedding;
      for (let i=0;i<sampleBuffers.length;i++) sampleBuffers[i].embedding = embeddings[i];
    } catch (e){
      console.warn('umap-js error', e);
      embeddings = sampleBuffers.map(()=>[Math.random(), Math.random()]);
      for (let i=0;i<sampleBuffers.length;i++) sampleBuffers[i].embedding = embeddings[i];
    }
  } else {
    console.warn('UMAP not available, using fallback layout');
    embeddings = sampleBuffers.map(()=>[Math.random(), Math.random()]);
    for (let i=0;i<sampleBuffers.length;i++) sampleBuffers[i].embedding = embeddings[i];
  }
}

async function extractFeaturesFromBuffer(buffer){
  const sr = buffer.sampleRate;
  const channelData = buffer.getChannelData(0);
  const frameSize = 1024;
  const hop = 512;
  const frames = [];
  for (let i=0;i+frameSize < channelData.length; i+=hop){
    frames.push(channelData.slice(i, i+frameSize));
  }

  const featsPerFrame = [];
  for (let f of frames){
    featsPerFrame.push({
      rms: computeRMS(f),
      spectralCentroid: computeSpectralCentroid(f, sr),
      spectralFlatness: computeSpectralFlatness(f),
      zcr: computeZCR(f)
    });
  }

  const keys = ['rms','spectralCentroid','spectralFlatness','zcr'];
  const stats = {};
  for (let k of keys){
    const vals = featsPerFrame.map(x=>x[k]||0);
    stats[k] = { mean: mean(vals), std: stddev(vals) };
  }

  const vector = [
    Math.log1p(stats.rms.mean || 0),
    Math.log1p(stats.rms.std || 0),
    Math.log1p(stats.spectralCentroid.mean || 0),
    Math.log1p(stats.spectralCentroid.std || 0),
    stats.spectralFlatness.mean || 0,
    stats.spectralFlatness.std || 0,
    stats.zcr.mean || 0,
    stats.zcr.std || 0
  ];
  return { vector, stats };
}

function computeRMS(frame){ let s=0; for(let v of frame) s+=v*v; return Math.sqrt(s/frame.length); }
function computeZCR(frame){ let c=0; for (let i=1;i<frame.length;i++) if ((frame[i]>0)!=(frame[i-1]>0)) c++; return c/frame.length; }
function computeSpectralCentroid(frame, sr){
  const N = frame.length;
  const nyquist = sr/2;
  const binCount = 32;
  const binSize = Math.max(1, Math.floor(N/binCount));
  let mag = new Array(binCount).fill(0);
  for (let b=0;b<binCount;b++){
    let s=0;
    for (let i=b*binSize;i<(b+1)*binSize && i<N;i++) s += Math.abs(frame[i]);
    mag[b] = s;
  }
  let numer=0, denom=0;
  for (let b=0;b<binCount;b++){ numer += b * mag[b]; denom += mag[b]; }
  const centroidBin = denom>0 ? numer/denom : 0;
  return (centroidBin/binCount) * nyquist;
}
function computeSpectralFlatness(frame){
  let vals = frame.map(x=>Math.abs(x)+1e-12);
  let logsum = 0;
  for (let v of vals) logsum += Math.log(v);
  const geom = Math.exp(logsum/vals.length);
  const arith = vals.reduce((a,b)=>a+b,0)/vals.length;
  return geom/arith;
}
function mean(a){ return a.length ? a.reduce((s,x)=>s+x,0)/a.length : 0; }
function stddev(a){ if (!a.length) return 0; const m=mean(a); return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/a.length); }

// buildVisualPoints: duration -> sizeMultiplier, and per-petal organic params
function buildVisualPoints(){
  points = [];
  if (!sampleBuffers.length) return;
  const xs = sampleBuffers.map(s=>s.embedding[0]);
  const ys = sampleBuffers.map(s=>s.embedding[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  for (let s of sampleBuffers){
    const nx = (s.embedding[0] - minX) / (maxX - minX || 1);
    const ny = (s.embedding[1] - minY) / (maxY - minY || 1);
    const x = lerp(80, width-80, nx);
    const y = lerp(80, height-80, ny);

    const stats = s.features.stats;
    const energy = stats.rms.mean;
    const centroid = stats.spectralCentroid.mean;
    const flatness = stats.spectralFlatness.mean;
    const zcr = stats.zcr.mean;

    const petals = Math.max(3, Math.round(3 + (centroid/1000)*9));
    const baseSize = 18 + energy*200 * (ui.sizeScale ? parseFloat(ui.sizeScale.value) : 1.0);
    const petalLength = 0.6 + (centroid/3000) * 1.8;
    const fuzz = Math.min(1, flatness*6);

    // duration-based size multiplier (log mapping)
    const dur = Math.max(MIN_DUR, Math.min(MAX_DUR, (s.buffer && s.buffer.duration) ? s.buffer.duration : MIN_DUR));
    const logNorm = (Math.log(dur) - Math.log(MIN_DUR)) / (Math.log(MAX_DUR) - Math.log(MIN_DUR));
    const sizeMultiplier = lerp(SIZE_SCALE_RANGE[0], SIZE_SCALE_RANGE[1], clamp(logNorm, 0, 1));

    // sharpness calc
    const normCentroid = clamp(centroid / NORMALIZE_CENTROID, 0, 1);
    const sharpness = clamp(
      SHARP_WEIGHTS.centroid * normCentroid +
      SHARP_WEIGHTS.flatness * flatness +
      SHARP_WEIGHTS.zcr * zcr,
      0, 1
    );

    // create per-petal organic params
    const petalParams = [];
    for (let k=0;k<petals;k++){
      petalParams.push({
        offset: Math.random() * 0.18 * (0.6 + Math.random()*1.2), // slight per-petal delay (fraction of duration)
        bend: 0.4 + Math.random()*0.9,   // bend factor
        twist: (Math.random()-0.5) * 18, // rotation offset deg
        controlJitter: Math.random()*0.25 // jitter for control point movement
      });
    }

    points.push({
      x,y,
      petals,
      size: baseSize * sizeMultiplier, // final target size already scaled by duration
      baseSize,
      sizeMultiplier,
      petalLength,
      fuzz,
      zcr,
      name: s.name,
      buffer: s.buffer,
      features: s.features,
      openProgress: 0,
      isOpening: false,
      openStart: 0,
      openDuration: baseOpenDuration * (0.85 + Math.random()*0.35),
      sharpness,
      duration: dur,
      petalParams
    });
  }
}

function draw(){
  background(18,20,29);
  noStroke();
  fill(0,0,0,80);
  rect(0,0,width,height);

  if (!points.length){
    push();
    fill(150,170,190,30);
    textAlign(CENTER, CENTER);
    textSize(18);
    text('右側の「wav をアップロード」で\nファイルを選び、「生成」を押してください', width/2, height/2);
    pop();
    noLoop();
    return;
  }

  const now = millis();
  let anyAnim = false;
  for (let p of points){
    if (p.isOpening){
      anyAnim = true;
      const prog = clamp((now - p.openStart) / p.openDuration, 0, 1);
      p.openProgress = easeOutCubic(prog);
      if (prog >= 1){
        p.openProgress = 1;
        p.isOpening = false;
      }
    }
  }
  if (anyAnim) loop(); else noLoop();

  const sorted = points.slice().sort((a,b)=>a.size - b.size);
  const pal = getActivePalette();
  for (let p of sorted){
    push();
    translate(p.x + random(- (ui.jitter ? ui.jitter.value : 6), (ui.jitter ? ui.jitter.value : 6)),
              p.y + random(- (ui.jitter ? ui.jitter.value : 6), (ui.jitter ? ui.jitter.value : 6)));
    drawOrganicFlower(p, pal);
    pop();
  }
}

// 描画: ランのような曲線的な開き方を実装
function drawOrganicFlower(p, pal){
  const col = colorFromPalette(pal, p);
  noStroke();
  const op = p.openProgress || 0;

  // dynamic size: interpolate from bud->final; final size already scaled by duration in p.size
  const budBase = p.baseSize * 0.18;
  const budRadius = budBase * 10;
  const curSize = lerp(budBase, p.size, op);
  const curPetalLen = lerp(p.petalLength * 0.12, p.petalLength, op);
  const alphaBase = lerp(0.0, 1.0, op);

  // gentle shadow
  push();
  noStroke();
  const shadowW = lerp(budRadius*0.6, curSize*1.1, op);
  fill(10, 10, 10, 40);
  ellipse(6, curSize*0.35, shadowW, shadowW*0.45);
  pop();

  if (op < 0.02){
    // large bud button (unchanged)
    stroke(red(col), green(col), blue(col), 110);
    strokeWeight(2.5);
    noFill();
    ellipse(0,0, budRadius, budRadius);
    noStroke();
    for (let i=0;i<5;i++){
      const r = budRadius * (0.5 + i*0.12);
      fill(red(col), green(col), blue(col), Math.floor(12 * (6-i)));
      ellipse(0,0,r,r);
    }
    fill(lerpColorString('#ffffff', col, 0.12));
    ellipse(-3, -3, budRadius*0.48, budRadius*0.48);
    fill(30,30,30,220);
    textAlign(CENTER, CENTER);
    textSize(Math.max(10, budRadius*0.06));
    text('▶', budRadius*0.02, 0);
    return;
  }

  // get sharpness
  const sharp = clamp(p.sharpness || 0.5, 0, 1);

  // petal parameters
  const n = p.petals;
  const timeBase = millis() * 0.0008;
  for (let layer=0; layer<4; layer++){
    const layerT = layer / 4;
    const layerAlpha = lerp(0.24, 0.92, 1 - layerT) * alphaBase;
    for (let k=0;k<n;k++){
      const params = p.petalParams[k % p.petalParams.length];
      // per-petal open progression: add offset and slight per-petal easing
      const petalDelay = params.offset * p.openDuration;
      const localT = clamp((p.openProgress * p.openDuration - petalDelay) / p.openDuration, 0, 1);
      const petOp = easeOutCubic(localT);

      // compute petal geometry influenced by sharpness and organicness
      const tipStretch = 1 + sharp * 0.6;
      const petLen = curSize * curPetalLen * tipStretch * (1 - layerT*0.06);
      const petWidthBase = curSize * (0.36 * (1 - 0.45 * sharp));
      const petWidth = petWidthBase * (1 - layerT*0.06);

      // bending: combine fixed bend param and organicalness * noise/time
      const bend = params.bend * (0.6 + 0.8 * ORGANICNESS) * (0.6 + 0.4*(1-petOp));
      const twist = params.twist * (1 - 0.5*petOp);
      const wobble = Math.sin(timeBase * (1 + params.controlJitter*6) + k * 0.7) * (2.0 * params.controlJitter);

      push();
      rotate((360/n) * k + twist + wobble);
      // control points move nonlinearly with petOp -> gives curved opening
      const cp1x = petLen * (0.18 + 0.12 * (1-petOp));
      const cp1y = -petWidth * (0.7 * bend) * (1 - 0.5*petOp); // curvature reduces as opens
      const cp2x = petLen * (0.55 + 0.25 * petOp);
      const cp2y = -petWidth * (0.06 + 0.18 * (1-petOp));

      // tip influenced by petOp to move along a curved path (y offset)
      const tipX = petLen * (0.9 + 0.15 * petOp);
      const tipY = - petLen * 0.08 * ORGANICNESS * (1 - petOp); // during early open tip might curve upward

      // color and shading
      fill(lerpColorString(col, '#ffffff', 0.06 + layerT*0.06), Math.floor(255 * layerAlpha));
      noStroke();
      beginShape();
      vertex(0,0);
      // bezier to tip (right side)
      bezierVertex(cp1x, cp1y, cp2x, cp2y, tipX, tipY);
      // back via mirrored left side (creates smooth petal)
      bezierVertex(cp2x, -cp2y, cp1x, -cp1y, 0,0);
      endShape(CLOSE);

      // edge stroke
      const edgeAlpha = 80 + sharp * 60;
      stroke(lerpColorString('#ffffff', col, 0.18));
      strokeWeight(0.6 + sharp*0.8);
      stroke(0,0,0,Math.floor(edgeAlpha * (0.6 - layerT*0.25)));
      noFill();
      beginShape();
      vertex(0,0);
      bezierVertex(cp1x, cp1y, cp2x, cp2y, tipX, tipY);
      bezierVertex(cp2x, -cp2y, cp1x, -cp1y, 0,0);
      endShape(CLOSE);
      pop();
    }
  }

  // center pattern
  push();
  noStroke();
  for (let i=0;i<6;i++){
    const r = curSize*0.12 * (1 + i*0.22);
    fill(lerpColorString('#ffffff', col, 0.12 + i*0.02), Math.floor((180 - i*20) * alphaBase));
    ellipse(0,0,r,r);
  }
  fill(255,255,255,170 * alphaBase);
  ellipse(-curSize*0.04, -curSize*0.06, curSize*0.06, curSize*0.04);
  pop();
}

function mouseMoved(){
  for (let p of points){
    const op = p.openProgress || 0;
    let hitR;
    if (op < 0.02){
      const budBase = p.baseSize * 0.18;
      const budRadius = budBase * 10;
      hitR = budRadius * 0.6;
    } else {
      hitR = p.size * (0.4 + 0.6 * op);
    }
    const d = dist(mouseX, mouseY, p.x, p.y);
    if (d < hitR){ cursor('pointer'); return; }
  }
  cursor('default');
}

function mouseClicked(){
  if (!points.length) return;
  initAudio();
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();

  for (let p of points){
    const op = p.openProgress || 0;
    let hitR;
    if (op < 0.02){
      const budBase = p.baseSize * 0.18;
      const budRadius = budBase * 10;
      hitR = budRadius * 0.6;
    } else {
      hitR = p.size * (0.4 + 0.6 * op);
    }
    const d = dist(mouseX, mouseY, p.x, p.y);
    if (d < hitR){
      if (p.buffer) playBuffer(p.buffer);
      triggerOpen(p);
      return;
    }
  }
}

function triggerOpen(p){
  if (!p) return;
  if (!p.isOpening && (p.openProgress || 0) < 1.0){
    // opening duration may scale with duration (optionally)
    p.openDuration = baseOpenDuration * (0.6 + 0.8 * (Math.log(p.duration + 1) / Math.log(MAX_DUR + 1)));
    p.isOpening = true;
    p.openStart = millis();
    loop();
  }
}

function playBuffer(buf){
  if (!audioCtx) initAudio();
  try {
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    const g = audioCtx.createGain();
    g.gain.value = 0.95;
    src.connect(g);
    g.connect(masterGain || audioCtx.destination);
    src.start();
  } catch(e){ console.error('playBuffer error', e); setStatus('再生エラー'); }
}

function regenerate(){ onGenerateClick(); }
function exportPNG(){ saveCanvas(canvas, 'umap-flowers', 'png'); }

function onPaletteChange(e){
  const v = e.target.value;
  document.getElementById('custom-colors').style.display = (v==='custom' ? 'block' : 'none');
  redraw();
}
function getActivePalette(){
  const v = ui.paletteSelect && ui.paletteSelect.value ? ui.paletteSelect.value : 'animals';
  if (v === 'animals') return palettes.animals;
  if (v === 'pastel') return palettes.pastel;
  if (v === 'deep') return palettes.deep;
  if (v === 'monochrome') return palettes.monochrome;
  if (ui.col1 && ui.col2 && ui.col3) return [ui.col1.value, ui.col2.value, ui.col3.value];
  return palettes.animals;
}
function colorFromPalette(pal, p){
  const i = Math.floor( (hashStringTo01(p.name) * pal.length) ) % pal.length;
  return tinyColorShift(pal[i], (hashStringTo01(p.name+'h')-0.5)*30);
}
function tinyColorShift(hex, deg){
  const c = color(hex);
  colorMode(HSB, 360, 100, 100, 255);
  const h = hue(c), s = saturation(c), b = brightness(c);
  const nh = (h + deg + 360) % 360;
  const nc = color(nh, s, b);
  colorMode(RGB,255);
  return nc;
}
function lerpColorString(a,b,t){
  const ca = color(a);
  const cb = color(b);
  return lerpColor(ca,cb,t);
}
function hashStringTo01(s){
  let h=2166136261;
  for(let i=0;i<s.length;i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return ((h >>> 0) % 1000)/1000;
}

function setStatus(s){ if (ui.status) ui.status.innerText = s; console.log('[status]', s); }
function lerp(a,b,t){ return a + (b-a)*t; }
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function easeOutCubic(t){ t = clamp(t,0,1); return 1 - Math.pow(1-t,3); }