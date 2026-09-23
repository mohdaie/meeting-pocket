const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
let hf = null;
let transcriber = null;
let generator = null;
let transcriberKey = '';
let generatorKey = '';

async function lib() {
  if (!hf) hf = await import(TRANSFORMERS_CDN);
  return hf;
}

function aiDevice() {
  return 'gpu' in navigator ? 'webgpu' : 'wasm';
}

async function blobToMono16k(blob) {
  const ctx = new AudioContext();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  } finally {
    await ctx.close().catch(() => {});
  }
}

export async function transcribeBlob(blob, {
  model = 'onnx-community/whisper-base',
  onProgress = () => {},
} = {}) {
  const { pipeline } = await lib();
  const device = aiDevice();
  const key = `${model}:${device}`;
  if (!transcriber || transcriberKey !== key) {
    onProgress(`Loading ${model.split('/').pop()}…`);
    transcriber = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype: device === 'webgpu' ? {
        encoder_model: 'fp16',
        decoder_model_merged: 'q4',
      } : 'q8',
      progress_callback: (p) => {
        if (p?.status === 'progress' && Number.isFinite(p.progress)) {
          onProgress(`Downloading speech AI ${Math.round(p.progress)}%`);
        }
      },
    });
    transcriberKey = key;
  }

  onProgress('Preparing audio…');
  const audio = await blobToMono16k(blob);
  onProgress('Transcribing locally…');
  const result = await transcriber(audio, {
    chunk_length_s: 25,
    stride_length_s: 5,
    return_timestamps: true,
    task: 'transcribe',
  });
  return {
    text: (result?.text || '').trim(),
    chunks: result?.chunks || [],
  };
}

async function getGenerator(model, onProgress) {
  const { pipeline } = await lib();
  const device = aiDevice();
  const key = `${model}:${device}`;
  if (!generator || generatorKey !== key) {
    onProgress(`Loading ${model.split('/').pop()}…`);
    generator = await pipeline('text-generation', model, {
      device,
      dtype: device === 'webgpu' ? 'q4' : 'q8',
      progress_callback: (p) => {
        if (p?.status === 'progress' && Number.isFinite(p.progress)) {
          onProgress(`Downloading summary AI ${Math.round(p.progress)}%`);
        }
      },
    });
    generatorKey = key;
  }
  return generator;
}

function splitTranscript(text, maxChars = 6500) {
  const parts = [];
  let remaining = text.trim();
  while (remaining.length > maxChars) {
    let cut = remaining.lastIndexOf('\n', maxChars);
    if (cut < maxChars * 0.6) cut = remaining.lastIndexOf('. ', maxChars);
    if (cut < maxChars * 0.6) cut = maxChars;
    parts.push(remaining.slice(0, cut + 1));
    remaining = remaining.slice(cut + 1).trim();
  }
  if (remaining) parts.push(remaining);
  return parts;
}

async function generateText(messages, model, onProgress, maxNewTokens = 380) {
  const gen = await getGenerator(model, onProgress);
  const out = await gen(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    repetition_penalty: 1.08,
  });
  const generated = out?.[0]?.generated_text;
  if (Array.isArray(generated)) return generated.at(-1)?.content || '';
  if (typeof generated === 'string') return generated;
  return '';
}

export async function summarizeTranscript(text, {
  model = 'onnx-community/Qwen2.5-0.5B-Instruct',
  onProgress = () => {},
} = {}) {
  const parts = splitTranscript(text);
  const mini = [];
  for (let i = 0; i < parts.length; i++) {
    onProgress(`Analysing part ${i + 1}/${parts.length}…`);
    const response = await generateText([
      { role: 'system', content: 'You extract facts from meeting transcripts. Do not invent facts. Keep names, dates, numbers, decisions, actions and unresolved questions.' },
      { role: 'user', content: `Summarize this transcript chunk in concise bullet points. Preserve important details.\n\n${parts[i]}` },
    ], model, onProgress, 280);
    mini.push(response);
  }

  onProgress('Building meeting brief…');
  return await generateText([
    { role: 'system', content: 'You are a meeting-notes assistant. Use only the supplied notes. If something is uncertain, say it is unclear. Never invent owners or deadlines.' },
    { role: 'user', content: `Create a concise meeting brief from these chunk notes. Use exactly these headings:\nOVERVIEW\nDECISIONS\nACTION ITEMS\nDATES & NUMBERS\nRISKS / CONCERNS\nOPEN QUESTIONS\n\nChunk notes:\n${mini.join('\n\n---\n\n')}` },
  ], model, onProgress, 500);
}

export async function askMeeting(question, transcript, {
  model = 'onnx-community/Qwen2.5-0.5B-Instruct',
  onProgress = () => {},
} = {}) {
  const terms = question.toLowerCase().split(/\W+/).filter(x => x.length > 3);
  const paras = transcript.split(/\n+/).filter(Boolean);
  const scored = paras.map((p, idx) => ({
    p, idx,
    score: terms.reduce((n,t) => n + (p.toLowerCase().includes(t) ? 2 : 0), 0),
  })).sort((a,b) => b.score - a.score || a.idx - b.idx);
  const selected = (scored.some(x => x.score > 0) ? scored.slice(0,12) : scored.slice(0,8))
    .sort((a,b) => a.idx - b.idx).map(x => x.p).join('\n');

  onProgress('Reading relevant transcript…');
  return await generateText([
    { role: 'system', content: 'Answer questions about a meeting using only the provided transcript excerpts. Be concise. If the answer is not supported, say it was not clearly mentioned.' },
    { role: 'user', content: `Question: ${question}\n\nTranscript excerpts:\n${selected}` },
  ], model, onProgress, 260);
}
