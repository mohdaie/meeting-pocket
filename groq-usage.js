const STORAGE_KEY = 'meeting-pocket-groq-usage-v1';
const MAX_EVENTS = 250;

export const GROQ_FREE_REFERENCE = {
  'qwen/qwen3.8-27b': { label: 'Qwen 3.8 27B', rpd: 1000, tpm: 8000, tpd: 200000 },
  'whisper-large-v3-turbo': { label: 'Whisper Large V3 Turbo', rpd: 2000, ash: 7200, asd: 28800 },
};

function localDay(ts = Date.now()) {
  const d = new Date(ts);
  return [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, '0'),
    String(d.getDate()).padStart(2, '0'),
  ].join('-');
}

function readEvents() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeEvents(events) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(events.slice(-MAX_EVENTS)));
  } catch {}
}

function num(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function groqRateHeaders(response) {
  const h = response?.headers;
  if (!h) return {};
  return {
    limitRequests: num(h.get('x-ratelimit-limit-requests')),
    remainingRequests: num(h.get('x-ratelimit-remaining-requests')),
    limitTokens: num(h.get('x-ratelimit-limit-tokens')),
    remainingTokens: num(h.get('x-ratelimit-remaining-tokens')),
    resetRequests: h.get('x-ratelimit-reset-requests') || '',
    resetTokens: h.get('x-ratelimit-reset-tokens') || '',
    retryAfter: h.get('retry-after') || '',
  };
}

export function recordGroqUsage(event) {
  const row = {
    ts: Date.now(),
    kind: event.kind || 'request',
    model: event.model || 'unknown',
    ok: event.ok !== false,
    status: Number(event.status || 0),
    inputTokens: Number(event.inputTokens || 0),
    outputTokens: Number(event.outputTokens || 0),
    totalTokens: Number(event.totalTokens || 0),
    audioSeconds: Number(event.audioSeconds || 0),
    latencyMs: Number(event.latencyMs || 0),
    serverTotalSeconds: Number(event.serverTotalSeconds || 0),
    rate: event.rate || {},
    error: event.error ? String(event.error).slice(0, 220) : '',
  };
  const events = readEvents();
  events.push(row);
  writeEvents(events);
  try {
    window.dispatchEvent(new CustomEvent('meeting-pocket-groq-usage'));
  } catch {}
  return row;
}

export function clearGroqUsage() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
  try {
    window.dispatchEvent(new CustomEvent('meeting-pocket-groq-usage'));
  } catch {}
}

export function getGroqUsageSnapshot() {
  const all = readEvents();
  const day = localDay();
  const today = all.filter(x => localDay(x.ts) === day);
  const success = today.filter(x => x.ok);
  const failed = today.filter(x => !x.ok);
  const qwen = today.filter(x => x.model === 'qwen/qwen3.8-27b');
  const whisper = today.filter(x => x.model === 'whisper-large-v3-turbo');

  const latestFor = (model) => [...today].reverse().find(x => x.model === model) || null;
  const latest = today.at(-1) || null;
  const sum = (rows, field) => rows.reduce((n, x) => n + (Number(x[field]) || 0), 0);

  return {
    day,
    events: today,
    totalRequests: today.length,
    successRequests: success.length,
    failedRequests: failed.length,
    inputTokens: sum(today, 'inputTokens'),
    outputTokens: sum(today, 'outputTokens'),
    totalTokens: sum(today, 'totalTokens'),
    audioSeconds: sum(today, 'audioSeconds'),
    qwen: {
      requests: qwen.length,
      inputTokens: sum(qwen, 'inputTokens'),
      outputTokens: sum(qwen, 'outputTokens'),
      totalTokens: sum(qwen, 'totalTokens'),
      latest: latestFor('qwen/qwen3.8-27b'),
    },
    whisper: {
      requests: whisper.length,
      audioSeconds: sum(whisper, 'audioSeconds'),
      latest: latestFor('whisper-large-v3-turbo'),
    },
    latest,
  };
}
