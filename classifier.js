// BluePort AI in-browser classifier: CLIP ViT-B/32 (ONNX, Transformers.js) + logistic-regression head.
// Nothing leaves the browser: the model is downloaded once (about 90 MB, cached) and photos are processed locally.
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const CLASSES = [
  { key: 'eletronico', en: 'E-waste', pt: 'Eletrônico', hint: 'Never in general waste. Deliver to a WEEE collection point; ports usually contract a licensed e-waste handler.' },
  { key: 'metal', en: 'Metal', pt: 'Metal', hint: 'Recyclable. Rinse and place in the metal stream; ship-generated scrap is segregated under MARPOL Annex V.' },
  { key: 'organico', en: 'Organic', pt: 'Orgânico', hint: 'Compost or organic stream. Ship-generated food waste has strict discharge rules under MARPOL Annex V.' },
  { key: 'papel', en: 'Paper / cardboard', pt: 'Papel / papelão', hint: 'Recyclable when clean and dry. Flatten cardboard; oily or wet paper goes to general waste.' },
  { key: 'plastico', en: 'Plastic', pt: 'Plástico', hint: 'Recyclable in most streams. Rinse containers; plastic must never be discharged at sea (MARPOL Annex V).' },
  { key: 'vidro', en: 'Glass', pt: 'Vidro', hint: 'Recyclable. Keep whole where possible; broken glass goes in a rigid container for safety.' },
];

const $ = (s) => document.querySelector(s);
const statusEl = $('#bp-status'), barEl = $('#bp-bar'), resultEl = $('#bp-result'), previewEl = $('#bp-preview'), dropEl = $('#bp-drop');
let processor, model, probe, loading;

function setStatus(text, pct) {
  statusEl.textContent = text;
  if (pct != null) { barEl.style.width = pct + '%'; barEl.parentElement.style.opacity = pct >= 100 ? 0 : 1; }
}

async function load() {
  if (model) return;
  if (loading) return loading;
  loading = (async () => {
    setStatus('Downloading the model (about 90 MB, once; it stays cached in your browser)…', 2);
    probe = await (await fetch('probe.json')).json();
    const progress = (p) => { if (p.status === 'progress' && p.file && p.file.includes('onnx')) setStatus(`Downloading the model… ${Math.round(p.progress)}%`, Math.max(2, p.progress)); };
    processor = await AutoProcessor.from_pretrained(MODEL_ID);
    model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8', progress_callback: progress });
    setStatus('Model ready. Drop a photo or pick an example.', 100);
  })();
  return loading;
}

async function classify(src) {
  previewEl.src = src; previewEl.style.display = 'block'; dropEl.classList.add('has-image');
  resultEl.innerHTML = '';
  await load();
  setStatus('Classifying…');
  const t0 = performance.now();
  const img = await RawImage.read(src);
  const inputs = await processor(img);
  const { image_embeds } = await model(inputs);
  let e = Array.from(image_embeds.data); const n = Math.hypot(...e); e = e.map((v) => v / n);
  const logits = probe.W.map((w, i) => w.reduce((s, wi, j) => s + wi * e[j], 0) + probe.b[i]);
  const m = Math.max(...logits); const ex = logits.map((l) => Math.exp(l - m)); const Z = ex.reduce((a, b) => a + b);
  const p = ex.map((v) => v / Z);
  const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
  const top = order[0], conf = p[top], c = CLASSES[top];
  const ms = Math.round(performance.now() - t0);
  const rows = order.slice(0, 3).map((i) => `
    <div class="bp-row"><span class="bp-lbl">${CLASSES[i].en} <small>· ${CLASSES[i].pt}</small></span>
      <span class="bp-track"><span class="bp-fill" style="width:${(p[i] * 100).toFixed(1)}%"></span></span>
      <span class="bp-pct">${(p[i] * 100).toFixed(1)}%</span></div>`).join('');
  const verdict = conf < 0.5
    ? `<p class="bp-note">Low confidence (${(conf * 100).toFixed(0)}%). The item may be mixed, unusual or outside the six trained categories. Best guess: <b>${c.en}</b>.</p>`
    : `<p class="bp-verdict"><b>${c.en}</b> <span>(${c.pt})</span></p><p class="bp-note">${c.hint}</p>`;
  resultEl.innerHTML = verdict + rows + `<p class="bp-meta">${ms} ms in your browser · nothing was uploaded</p>`;
  setStatus('Done. Try another photo.');
}

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const r = new FileReader(); r.onload = () => classify(r.result); r.readAsDataURL(file);
}

$('#bp-file').addEventListener('change', (ev) => handleFile(ev.target.files[0]));
dropEl.addEventListener('dragover', (ev) => { ev.preventDefault(); dropEl.classList.add('over'); });
dropEl.addEventListener('dragleave', () => dropEl.classList.remove('over'));
dropEl.addEventListener('drop', (ev) => { ev.preventDefault(); dropEl.classList.remove('over'); handleFile(ev.dataTransfer.files[0]); });
dropEl.addEventListener('click', () => $('#bp-file').click());
document.querySelectorAll('.bp-example').forEach((el) => el.addEventListener('click', () => classify(el.src)));
window.addEventListener('paste', (ev) => { const f = [...(ev.clipboardData?.files || [])][0]; if (f) handleFile(f); });
$('#bp-load')?.addEventListener('click', load);
