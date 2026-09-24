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

function isMobileDevice() {
  return navigator.userAgentData?.mobile === true
    || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent || '');
}

function mobileLocalModelError(feature = 'Local AI') {
  return new Error(
    feature + ' is disabled on phones/tablets because loading Qwen in the browser can exceed renderer memory and crash the page. Use Gemini 3.8 Flash instead.'
  );
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

function prepareMeetingAudio(input) {
  if (!input?.length) return { audio: input, silent: true };

  let mean = 0;
  for (let i = 0; i < input.length; i++) mean += input[i];
  mean /= input.length;

  const audio = new Float32Array(input.length);
  let peak = 0;
  let sumSq = 0;
  for (let i = 0; i < input.length; i++) {
    const v = input[i] - mean;
    audio[i] = v;
    const a = Math.abs(v);
    if (a > peak) peak = a;
    sumSq += v * v;
  }

  const rms = Math.sqrt(sumSq / audio.length);
  if (peak < 0.003 || rms < 0.00035) {
    return { audio, silent: true, peak, rms };
  }

  // Quiet room recordings benefit from a conservative digital gain before
  // Whisper. Cap it so background noise is not amplified excessively.
  const gain = Math.min(5, Math.max(1, 0.82 / Math.max(peak, 0.001)));
  if (gain > 1.05) {
    for (let i = 0; i < audio.length; i++) {
      audio[i] = Math.max(-1, Math.min(1, audio[i] * gain));
    }
  }

  return { audio, silent: false, peak, rms, gain };
}

function normalizeWords(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}'-]+/gu, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function repetitionScore(text) {
  const words = normalizeWords(text);
  if (words.length < 10) return 0;

  const counts = new Map();
  for (const w of words) counts.set(w, (counts.get(w) || 0) + 1);
  const dominantWord = Math.max(...counts.values()) / words.length;

  let maxPhraseRun = 1;
  for (let n = 1; n <= Math.min(5, Math.floor(words.length / 4)); n++) {
    for (let i = 0; i + n * 4 <= words.length; i++) {
      const phrase = words.slice(i, i + n).join(' ');
      let run = 1;
      let pos = i + n;
      while (pos + n <= words.length && words.slice(pos, pos + n).join(' ') === phrase) {
        run++;
        pos += n;
      }
      if (run > maxPhraseRun) maxPhraseRun = run;
    }
  }

  if (maxPhraseRun >= 5) return 1;
  if (dominantWord >= 0.5) return 0.95;
  if (maxPhraseRun >= 4) return 0.9;
  return dominantWord;
}

function cleanChunkText(text) {
  return (text || '')
    .replace(/\s+/g, ' ')
    .replace(/([.!?])\1+/g, '$1')
    .trim();
}

function filterTranscript(result) {
  const sourceChunks = Array.isArray(result?.chunks) ? result.chunks : [];
  let filteredSegments = 0;

  if (sourceChunks.length) {
    const chunks = [];
    for (const chunk of sourceChunks) {
      const text = cleanChunkText(chunk?.text || '');
      if (!text) continue;
      if (repetitionScore(text) >= 0.9) {
        filteredSegments++;
        chunks.push({ ...chunk, text: '[unclear audio]' });
      } else {
        chunks.push({ ...chunk, text });
      }
    }

    // Avoid long runs of identical unclear markers.
    const compact = chunks.filter((chunk, i) =>
      chunk.text !== '[unclear audio]' || i === 0 || chunks[i - 1]?.text !== '[unclear audio]'
    );

    return {
      text: compact.map(x => x.text).join(' ').replace(/\s+/g, ' ').trim(),
      chunks: compact,
      filteredSegments,
    };
  }

  const text = cleanChunkText(result?.text || '');
  if (repetitionScore(text) >= 0.9) {
    return { text: '[unclear audio]', chunks: [], filteredSegments: 1 };
  }
  return { text, chunks: [], filteredSegments: 0 };
}

const GEMINI_TRANSCRIBE_MODEL = 'gemini-3.5-transcribe';
const GROQ_TRANSCRIBE_MODEL = 'whisper-large-v3-turbo';
const GEMINI_INLINE_LIMIT = 18 * 1024 * 1024;
const GROQ_FREE_FILE_LIMIT = 25 * 1024 * 1024;

async function responseError(response, label) {
  let detail = '';
  try {
    const payload = await response.clone().json();
    detail = payload?.error?.message || payload?.message || '';
  } catch {
    try { detail = (await response.text()).slice(0, 240); } catch {}
  }
  const suffix = detail ? ': ' + detail : '';
  return new Error(label + ' failed (' + response.status + ')' + suffix);
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const stride = 0x8000;
  for (let i = 0; i < bytes.length; i += stride) {
    binary += String.fromCharCode(...bytes.subarray(i, i + stride));
  }
  return btoa(binary);
}

function extractGeminiText(payload) {
  if (typeof payload?.output_text === 'string' && payload.output_text.trim()) {
    return payload.output_text.trim();
  }

  const outputs = Array.isArray(payload?.outputs) ? payload.outputs : [];
  const outputText = outputs
    .filter(x => x?.type === 'text' && typeof x.text === 'string')
    .map(x => x.text)
    .join('\n')
    .trim();
  if (outputText) return outputText;

  const steps = Array.isArray(payload?.steps) ? payload.steps : [];
  for (let i = steps.length - 1; i >= 0; i--) {
    const parts = Array.isArray(steps[i]?.content) ? steps[i].content : [];
    const text = parts
      .filter(x => x?.type === 'text' && typeof x.text === 'string')
      .map(x => x.text)
      .join('\n')
      .trim();
    if (text) return text;
  }

  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  return candidates
    .flatMap(x => x?.content?.parts || [])
    .map(x => x?.text || '')
    .join('\n')
    .trim();
}

async function geminiInlineInput(blob) {
  return {
    type: 'audio',
    data: arrayBufferToBase64(await blob.arrayBuffer()),
    mime_type: blob.type || 'audio/webm',
  };
}

async function uploadGeminiFile(blob, apiKey, onProgress) {
  onProgress('Uploading meeting audio to Gemini…');
  const mimeType = blob.type || 'audio/webm';

  const startResponse = await fetch('https://generativelanguage.googleapis.com/upload/v1beta/files', {
    method: 'POST',
    headers: {
      'x-goog-api-key': apiKey,
      'X-Goog-Upload-Protocol': 'resumable',
      'X-Goog-Upload-Command': 'start',
      'X-Goog-Upload-Header-Content-Length': String(blob.size),
      'X-Goog-Upload-Header-Content-Type': mimeType,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file: { display_name: 'meeting-pocket-' + Date.now() },
    }),
  });
  if (!startResponse.ok) throw await responseError(startResponse, 'Gemini upload start');

  const uploadUrl = startResponse.headers.get('x-goog-upload-url');
  if (!uploadUrl) {
    throw new Error('Gemini upload URL was not exposed to this browser. Try a shorter recording or Local mode.');
  }

  const uploadResponse = await fetch(uploadUrl, {
    method: 'POST',
    headers: {
      'X-Goog-Upload-Offset': '0',
      'X-Goog-Upload-Command': 'upload, finalize',
    },
    body: blob,
  });
  if (!uploadResponse.ok) throw await responseError(uploadResponse, 'Gemini audio upload');

  const uploaded = await uploadResponse.json();
  let file = uploaded?.file || uploaded;
  const name = file?.name || '';

  for (let i = 0; name && file?.state === 'PROCESSING' && i < 40; i++) {
    onProgress('Gemini is preparing the recording…');
    await new Promise(resolve => setTimeout(resolve, 750));
    const check = await fetch('https://generativelanguage.googleapis.com/v1beta/' + name, {
      headers: { 'x-goog-api-key': apiKey },
    });
    if (!check.ok) throw await responseError(check, 'Gemini file check');
    file = await check.json();
  }

  if (file?.state === 'FAILED') throw new Error('Gemini could not process this audio file.');
  return {
    input: {
      type: 'audio',
      uri: file?.uri,
      mime_type: file?.mimeType || file?.mime_type || mimeType,
    },
    name,
  };
}

async function deleteGeminiFile(name, apiKey) {
  if (!name) return;
  try {
    await fetch('https://generativelanguage.googleapis.com/v1beta/' + name, {
      method: 'DELETE',
      headers: { 'x-goog-api-key': apiKey },
    });
  } catch {}
}

async function transcribeWithGemini(blob, apiKey, onProgress) {
  if (!apiKey) throw new Error('Gemini API key is not configured.');

  let remoteName = '';
  try {
    let audioInput;
    if (blob.size <= GEMINI_INLINE_LIMIT) {
      onProgress('Preparing audio for Gemini multilingual transcription…');
      audioInput = await geminiInlineInput(blob);
    } else {
      const uploaded = await uploadGeminiFile(blob, apiKey, onProgress);
      audioInput = uploaded.input;
      remoteName = uploaded.name;
    }

    onProgress('Transcribing with Gemini 3.5 · auto multilingual…');
    const response = await fetch('https://generativelanguage.googleapis.com/v1beta/interactions', {
      method: 'POST',
      headers: {
        'x-goog-api-key': apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: GEMINI_TRANSCRIBE_MODEL,
        input: [audioInput],
        generation_config: {
          transcription_config: {
            language_codes: ['en-US', 'ms-MY', 'zh-CN', 'ta-IN'],
            mode: { type: 'verbatim' },
          },
        },
      }),
    });
    if (!response.ok) throw await responseError(response, 'Gemini transcription');

    const payload = await response.json();
    const text = cleanChunkText(extractGeminiText(payload));
    if (!text) throw new Error('Gemini returned an empty transcript.');

    return {
      text,
      chunks: [],
      filteredSegments: 0,
      provider: 'gemini',
      model: GEMINI_TRANSCRIBE_MODEL,
    };
  } finally {
    await deleteGeminiFile(remoteName, apiKey);
  }
}

function extensionForMime(mimeType) {
  if ((mimeType || '').includes('mp4')) return 'm4a';
  if ((mimeType || '').includes('ogg')) return 'ogg';
  if ((mimeType || '').includes('wav')) return 'wav';
  return 'webm';
}

async function transcribeWithGroq(blob, apiKey, onProgress) {
  if (!apiKey) throw new Error('Groq API key is not configured.');
  if (blob.size > GROQ_FREE_FILE_LIMIT) {
    throw new Error('Recording is over Groq free-tier 25 MB upload limit.');
  }

  onProgress('Transcribing with Groq Whisper Large V3…');
  const form = new FormData();
  form.append('file', blob, 'meeting-pocket.' + extensionForMime(blob.type));
  form.append('model', GROQ_TRANSCRIBE_MODEL);
  form.append('response_format', 'verbose_json');
  form.append('temperature', '0');
  form.append('timestamp_granularities[]', 'segment');

  const response = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + apiKey },
    body: form,
  });
  if (!response.ok) throw await responseError(response, 'Groq transcription');

  const payload = await response.json();
  const result = {
    text: payload?.text || '',
    chunks: Array.isArray(payload?.segments)
      ? payload.segments.map(s => ({
          text: s.text || '',
          timestamp: [s.start ?? null, s.end ?? null],
        }))
      : [],
  };
  const filtered = filterTranscript(result);
  return {
    ...filtered,
    provider: 'groq',
    model: GROQ_TRANSCRIBE_MODEL,
  };
}

async function transcribeLocally(blob, {
  model = 'onnx-community/whisper-small',
  onProgress = () => {},
} = {}) {
  const { pipeline } = await lib();
  const device = aiDevice();
  const key = model + ':' + device;
  if (!transcriber || transcriberKey !== key) {
    onProgress('Loading ' + model.split('/').pop() + '…');
    transcriber = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype: device === 'webgpu' ? {
        encoder_model: 'fp16',
        decoder_model_merged: 'q4',
      } : 'q8',
      progress_callback: (p) => {
        if (p?.status === 'progress' && Number.isFinite(p.progress)) {
          onProgress('Downloading speech AI ' + Math.round(p.progress) + '%');
        }
      },
    });
    transcriberKey = key;
  }

  onProgress('Preparing meeting audio…');
  const decoded = await blobToMono16k(blob);
  const prepared = prepareMeetingAudio(decoded);
  if (prepared.silent) {
    return { text: '', chunks: [], filteredSegments: 0, provider: 'local', model };
  }

  onProgress('Transcribing locally with multilingual Whisper…');
  const result = await transcriber(prepared.audio, {
    chunk_length_s: 28,
    stride_length_s: 4,
    return_timestamps: true,
    task: 'transcribe',
  });
  onProgress('Cleaning transcript…');
  return {
    ...filterTranscript(result),
    provider: 'local',
    model,
  };
}

export async function transcribeBlob(blob, {
  model = 'onnx-community/whisper-small',
  engine = 'auto',
  geminiApiKey = '',
  groqApiKey = '',
  onProgress = () => {},
} = {}) {
  const attempts = [];

  const run = async (label, fn) => {
    try {
      return await fn();
    } catch (err) {
      attempts.push(label + ': ' + (err?.message || err));
      return null;
    }
  };

  if (engine === 'gemini') {
    const result = await run('Gemini', () => transcribeWithGemini(blob, geminiApiKey, onProgress));
    if (result) return result;
    throw new Error(attempts.join(' | '));
  }

  if (engine === 'groq') {
    const result = await run('Groq', () => transcribeWithGroq(blob, groqApiKey, onProgress));
    if (result) return result;
    throw new Error(attempts.join(' | '));
  }

  if (engine === 'local') {
    return transcribeLocally(blob, { model, onProgress });
  }

  // Prefer Groq in Auto mode. It is fast, multilingual, and avoids loading
  // heavyweight browser AI on mobile. Gemini remains a cloud fallback.
  if (groqApiKey) {
    const result = await run('Groq', () => transcribeWithGroq(blob, groqApiKey, onProgress));
    if (result) return result;
    onProgress('Groq unavailable · trying Gemini fallback…');
  }

  if (geminiApiKey) {
    const result = await run('Gemini', () => transcribeWithGemini(blob, geminiApiKey, onProgress));
    if (result) return result;
    onProgress('Gemini unavailable · using local fallback…');
  }

  try {
    return await transcribeLocally(blob, { model, onProgress });
  } catch (err) {
    attempts.push('Local Whisper: ' + (err?.message || err));
    throw new Error(attempts.join(' | '));
  }
}

const GEMINI_SUMMARY_MODEL = 'gemini-3.8-flash';

async function generateWithGemini(prompt, apiKey, onProgress, label = 'Gemini') {
  if (!apiKey) throw new Error('Gemini API key is not configured.');
  onProgress(label + '…');

  const response = await fetch(
    'https://generativelanguage.googleapis.com/v1beta/models/' + GEMINI_SUMMARY_MODEL + ':generateContent',
    {
      method: 'POST',
      headers: {
        'x-goog-api-key': apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          thinkingConfig: { thinkingLevel: 'low' },
        },
      }),
    }
  );

  if (!response.ok) throw await responseError(response, label);
  const payload = await response.json();
  const text = (payload?.candidates || [])
    .flatMap(candidate => candidate?.content?.parts || [])
    .map(part => part?.text || '')
    .join('\n')
    .trim();

  if (!text) throw new Error(label + ' returned an empty response.');
  return text;
}

const GROQ_SUMMARY_MODEL = 'qwen/qwen3.8-27b';

async function generateWithGroq(prompt, apiKey, onProgress, label = 'Groq Qwen 3.8 27B') {
  if (!apiKey) throw new Error('Groq API key is not configured.');
  onProgress(label + '…');

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: GROQ_SUMMARY_MODEL,
      messages: [
        {
          role: 'system',
          content: 'You are Meeting Pocket, a factual multilingual meeting-notes assistant. Never invent facts, owners, dates, or decisions.'
        },
        { role: 'user', content: prompt },
      ],
      temperature: 0.2,
      reasoning_effort: 'none',
      reasoning_format: 'hidden',
      max_completion_tokens: 1800,
    }),
  });

  if (!response.ok) throw await responseError(response, label);
  const payload = await response.json();
  const text = payload?.choices?.[0]?.message?.content?.trim() || '';
  if (!text) throw new Error(label + ' returned an empty response.');
  return text;
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

async function summarizeLocally(text, model, onProgress) {
  const parts = splitTranscript(text);
  const mini = [];
  for (let i = 0; i < parts.length; i++) {
    onProgress(`Analysing part ${i + 1}/${parts.length} locally…`);
    const response = await generateText([
      { role: 'system', content: 'You extract facts from meeting transcripts. Treat [unclear audio] as missing information. Do not invent facts. Keep names, dates, numbers, decisions, actions and unresolved questions.' },
      { role: 'user', content: `Summarize this transcript chunk in concise bullet points. Preserve important details.\n\n${parts[i]}` },
    ], model, onProgress, 280);
    mini.push(response);
  }

  onProgress('Building meeting brief locally…');
  return await generateText([
    { role: 'system', content: 'You are a meeting-notes assistant. Use only the supplied notes. Treat [unclear audio] as missing information. If something is uncertain, say it is unclear. Never invent owners or deadlines.' },
    { role: 'user', content: `Create a concise meeting brief from these chunk notes. Use exactly these headings:\nOVERVIEW\nDECISIONS\nACTION ITEMS\nDATES & NUMBERS\nRISKS / CONCERNS\nOPEN QUESTIONS\n\nChunk notes:\n${mini.join('\n\n---\n\n')}` },
  ], model, onProgress, 500);
}

export async function summarizeTranscript(text, {
  engine = 'auto',
  model = 'onnx-community/Qwen2.5-0.5B-Instruct',
  geminiApiKey = '',
  groqApiKey = '',
  onProgress = () => {},
} = {}) {
  const prompt = `You are Meeting Pocket, a multilingual meeting-notes assistant.
Use only the transcript below. The transcript may mix English, Malay, Mandarin, Tamil or other languages.
Preserve names, dates, numbers, decisions, action owners, deadlines and unresolved questions.
Do not invent missing information. Treat [unclear audio] as missing information.
Return plain text using exactly these headings:

OVERVIEW
DECISIONS
ACTION ITEMS
DATES & NUMBERS
RISKS / CONCERNS
OPEN QUESTIONS

Transcript:
${text}`;

  if (engine === 'groq') {
    const result = await generateWithGroq(prompt, groqApiKey, onProgress, 'Generating summary with Groq Qwen 3.8 27B');
    return { text: result, provider: 'groq', model: GROQ_SUMMARY_MODEL };
  }

  if (engine === 'gemini') {
    const result = await generateWithGemini(prompt, geminiApiKey, onProgress, 'Generating summary with Gemini 3.8 Flash');
    return { text: result, provider: 'gemini', model: GEMINI_SUMMARY_MODEL };
  }

  if (engine === 'local') {
    if (isMobileDevice()) throw mobileLocalModelError('Local Qwen summary');
    const result = await summarizeLocally(text, model, onProgress);
    return { text: result, provider: 'local', model };
  }

  const attempts = [];

  // Groq is now the primary cloud engine for Meeting Pocket.
  if (groqApiKey) {
    try {
      const result = await generateWithGroq(prompt, groqApiKey, onProgress, 'Generating summary with Groq Qwen 3.8 27B');
      return { text: result, provider: 'groq', model: GROQ_SUMMARY_MODEL };
    } catch (err) {
      attempts.push('Groq: ' + (err?.message || err));
      onProgress(geminiApiKey ? 'Groq unavailable · trying Gemini…' : 'Groq unavailable…');
    }
  }

  if (geminiApiKey) {
    try {
      const result = await generateWithGemini(prompt, geminiApiKey, onProgress, 'Generating summary with Gemini 3.8 Flash');
      return { text: result, provider: 'gemini', model: GEMINI_SUMMARY_MODEL };
    } catch (err) {
      attempts.push('Gemini: ' + (err?.message || err));
      onProgress('Cloud summary unavailable…');
    }
  }

  if (isMobileDevice()) {
    const detail = attempts.length ? ' ' + attempts.join(' | ') : '';
    throw new Error(
      'Summary needs a working Groq or Gemini API key on mobile. Local Qwen is disabled to prevent browser crashes.' + detail
    );
  }

  onProgress('Cloud AI unavailable · using local Qwen fallback…');
  const result = await summarizeLocally(text, model, onProgress);
  return { text: result, provider: 'local', model };
}

function selectRelevantTranscript(question, transcript) {
  const terms = question.toLowerCase().split(/\W+/).filter(x => x.length > 3);
  const paras = transcript.split(/\n+/).filter(Boolean);
  const scored = paras.map((p, idx) => ({
    p, idx,
    score: terms.reduce((n,t) => n + (p.toLowerCase().includes(t) ? 2 : 0), 0),
  })).sort((a,b) => b.score - a.score || a.idx - b.idx);

  return (scored.some(x => x.score > 0) ? scored.slice(0,12) : scored.slice(0,8))
    .sort((a,b) => a.idx - b.idx)
    .map(x => x.p)
    .join('\n');
}

async function askLocally(question, transcript, model, onProgress) {
  const selected = selectRelevantTranscript(question, transcript);
  onProgress('Reading relevant transcript locally…');
  return await generateText([
    { role: 'system', content: 'Answer questions about a meeting using only the provided transcript excerpts. Treat [unclear audio] as missing information. Be concise. If the answer is not supported, say it was not clearly mentioned.' },
    { role: 'user', content: `Question: ${question}\n\nTranscript excerpts:\n${selected}` },
  ], model, onProgress, 260);
}

export async function askMeeting(question, transcript, {
  engine = 'auto',
  model = 'onnx-community/Qwen2.5-0.5B-Instruct',
  geminiApiKey = '',
  groqApiKey = '',
  onProgress = () => {},
} = {}) {
  const prompt = `Answer the question using only the meeting transcript below.
The meeting may contain multiple languages. Preserve the meaning of multilingual/code-switched statements.
If the answer is not supported by the transcript, say it was not clearly mentioned.
Be concise and factual.

Question:
${question}

Transcript:
${transcript}`;

  if (engine === 'groq') {
    const result = await generateWithGroq(prompt, groqApiKey, onProgress, 'Asking Groq Qwen 3.8 27B');
    return { text: result, provider: 'groq', model: GROQ_SUMMARY_MODEL };
  }

  if (engine === 'gemini') {
    const result = await generateWithGemini(prompt, geminiApiKey, onProgress, 'Asking Gemini 3.8 Flash');
    return { text: result, provider: 'gemini', model: GEMINI_SUMMARY_MODEL };
  }

  if (engine === 'local') {
    if (isMobileDevice()) throw mobileLocalModelError('Local Qwen Ask');
    const result = await askLocally(question, transcript, model, onProgress);
    return { text: result, provider: 'local', model };
  }

  const attempts = [];

  if (groqApiKey) {
    try {
      const result = await generateWithGroq(prompt, groqApiKey, onProgress, 'Asking Groq Qwen 3.8 27B');
      return { text: result, provider: 'groq', model: GROQ_SUMMARY_MODEL };
    } catch (err) {
      attempts.push('Groq: ' + (err?.message || err));
      onProgress(geminiApiKey ? 'Groq unavailable · trying Gemini…' : 'Groq unavailable…');
    }
  }

  if (geminiApiKey) {
    try {
      const result = await generateWithGemini(prompt, geminiApiKey, onProgress, 'Asking Gemini 3.8 Flash');
      return { text: result, provider: 'gemini', model: GEMINI_SUMMARY_MODEL };
    } catch (err) {
      attempts.push('Gemini: ' + (err?.message || err));
      onProgress('Cloud Ask unavailable…');
    }
  }

  if (isMobileDevice()) {
    const detail = attempts.length ? ' ' + attempts.join(' | ') : '';
    throw new Error(
      'Ask needs a working Groq or Gemini API key on mobile. Local Qwen is disabled to prevent browser crashes.' + detail
    );
  }

  onProgress('Cloud AI unavailable · using local Qwen fallback…');
  const result = await askLocally(question, transcript, model, onProgress);
  return { text: result, provider: 'local', model };
}
