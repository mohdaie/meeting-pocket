import { putMeeting, getMeeting, listMeetings, putChunk, getChunks } from './db.js';
import { transcribeBlob, summarizeTranscript, askMeeting } from './ai.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const homeView = $('#homeView');
const recordView = $('#recordView');
const meetingView = $('#meetingView');
const timerEl = $('#timer');
const chunkStatus = $('#chunkStatus');
const meetingList = $('#meetingList');
const capabilityCard = $('#capabilityCard');
const settingsDialog = $('#settingsDialog');
const speechModelSelect = $('#speechModelSelect');
const summaryModelSelect = $('#summaryModelSelect');
const keepAwakeToggle = $('#keepAwakeToggle');

let mediaRecorder = null;
let mediaStream = null;
let timerHandle = null;
let startedAt = 0;
let currentMeetingId = null;
let currentMeeting = null;
let chunkIndex = 0;
let wakeLock = null;
let stopping = false;

function showView(view) {
  [homeView, recordView, meetingView].forEach(v => v.classList.remove('active'));
  view.classList.add('active');
  window.scrollTo(0,0);
}

function fmtDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h ? `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function fmtDate(ts) {
  return new Intl.DateTimeFormat(undefined, { dateStyle:'medium', timeStyle:'short' }).format(new Date(ts));
}

function detectCapabilities() {
  const mic = !!navigator.mediaDevices?.getUserMedia;
  const recorder = 'MediaRecorder' in window;
  const webgpu = 'gpu' in navigator;
  const idb = 'indexedDB' in window;
  capabilityCard.innerHTML = `
    <div><span class="${mic && recorder && idb ? 'status-good' : 'status-warn'}">${mic && recorder && idb ? '● Ready to record locally' : '● Browser capability issue'}</span></div>
    <div style="margin-top:6px">AI acceleration: <strong>${webgpu ? 'WebGPU available' : 'CPU/WASM fallback'}</strong></div>`;
}

async function refreshHistory() {
  const rows = await listMeetings();
  meetingList.innerHTML = '';
  if (!rows.length) {
    meetingList.innerHTML = '<div class="empty">No meetings yet.</div>';
    return;
  }
  for (const m of rows) {
    const btn = document.createElement('button');
    btn.className = 'meeting-item';
    btn.innerHTML = `<strong>${escapeHtml(m.title || 'Meeting')}</strong><small>${fmtDate(m.startedAt)} · ${fmtDuration(m.durationSec || 0)}${m.transcript ? ' · Transcribed' : ''}</small>`;
    btn.onclick = () => openMeeting(m.id);
    meetingList.appendChild(btn);
  }
}

function escapeHtml(v='') {
  return v.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

async function requestWakeLock() {
  if (!keepAwakeToggle.checked || !('wakeLock' in navigator)) return;
  try { wakeLock = await navigator.wakeLock.request('screen'); } catch {}
}

async function releaseWakeLock() {
  try { await wakeLock?.release(); } catch {}
  wakeLock = null;
}

function pickMimeType() {
  const types = ['audio/webm;codecs=opus','audio/webm','audio/mp4'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    alert('This browser cannot record audio. Use a recent Chrome/Edge/Safari over HTTPS.');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch (err) {
    alert(`Microphone access failed: ${err.message || err}`);
    return;
  }

  currentMeetingId = crypto.randomUUID();
  startedAt = Date.now();
  chunkIndex = 0;
  stopping = false;
  currentMeeting = {
    id: currentMeetingId,
    title: `Meeting · ${new Intl.DateTimeFormat(undefined,{hour:'2-digit',minute:'2-digit'}).format(new Date())}`,
    startedAt,
    endedAt: null,
    durationSec: 0,
    mimeType: '',
    transcript: '',
    transcriptChunks: [],
    summary: '',
    status: 'recording',
  };

  const mimeType = pickMimeType();
  currentMeeting.mimeType = mimeType || 'audio/webm';
  await putMeeting(currentMeeting);

  mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  mediaRecorder.addEventListener('dataavailable', async (event) => {
    if (!event.data || !event.data.size) return;
    const idx = chunkIndex++;
    chunkStatus.textContent = `Saved chunk ${idx + 1} locally`;
    try { await putChunk(currentMeetingId, idx, event.data); }
    catch { chunkStatus.textContent = 'Warning: local save failed'; }
  });
  mediaRecorder.addEventListener('stop', finishRecording);

  mediaRecorder.start(30000); // 30-second recoverable chunks
  await requestWakeLock();
  showView(recordView);
  document.body.classList.add('ultra-dim');

  const tick = () => {
    timerEl.textContent = fmtDuration(Math.floor((Date.now() - startedAt) / 1000));
  };
  tick();
  timerHandle = setInterval(tick, 1000);
}

async function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive' || stopping) return;
  stopping = true;
  $('#stopBtn').disabled = true;
  chunkStatus.textContent = 'Finishing recording…';
  mediaRecorder.requestData();
  setTimeout(() => mediaRecorder?.state !== 'inactive' && mediaRecorder.stop(), 120);
}

async function finishRecording() {
  clearInterval(timerHandle);
  timerHandle = null;
  await releaseWakeLock();
  mediaStream?.getTracks().forEach(t => t.stop());
  mediaStream = null;
  document.body.classList.remove('ultra-dim');
  $('#stopBtn').disabled = false;

  const endedAt = Date.now();
  currentMeeting.endedAt = endedAt;
  currentMeeting.durationSec = Math.max(1, Math.floor((endedAt - startedAt) / 1000));
  currentMeeting.status = 'recorded';
  await putMeeting(currentMeeting);
  await openMeeting(currentMeeting.id);
}

async function openMeeting(id) {
  const meeting = await getMeeting(id);
  if (!meeting) return;
  currentMeetingId = id;
  currentMeeting = meeting;
  $('#meetingTitle').textContent = meeting.title || 'Meeting';
  $('#transcriptContent').textContent = meeting.transcript || '';
  $('#transcriptStatus').textContent = meeting.transcript ? 'Saved locally.' : 'No transcript yet.';
  $('#summaryStatus').textContent = meeting.summary ? 'Saved locally.' : 'No summary yet.';
  renderSummary(meeting.summary || '');
  $('#answerBox').textContent = '';
  showView(meetingView);
}

async function doTranscribe() {
  if (!currentMeeting) return;
  const btn = $('#transcribeBtn');
  btn.disabled = true;
  try {
    const rows = await getChunks(currentMeeting.id);
    if (!rows.length) throw new Error('No recorded audio chunks found.');

    const texts = [];
    const timedChunks = [];
    let offsetSec = 0;

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const result = await transcribeBlob(row.blob, {
        model: speechModelSelect.value,
        onProgress: (s) => $('#transcriptStatus').textContent = `Chunk ${i + 1}/${rows.length} · ${s}`,
      });

      if (result.text) texts.push(result.text);
      for (const chunk of result.chunks || []) {
        const ts = Array.isArray(chunk.timestamp) ? chunk.timestamp : null;
        timedChunks.push({
          ...chunk,
          timestamp: ts ? [
            Number.isFinite(ts[0]) ? ts[0] + offsetSec : ts[0],
            Number.isFinite(ts[1]) ? ts[1] + offsetSec : ts[1],
          ] : ts,
        });
      }

      // Save progress after every chunk so a long transcription can resume manually
      // without risking the already captured meeting audio.
      currentMeeting.transcript = texts.join('\n');
      currentMeeting.transcriptChunks = timedChunks;
      currentMeeting.status = i === rows.length - 1 ? 'transcribed' : 'transcribing';
      await putMeeting(currentMeeting);
      $('#transcriptContent').textContent = currentMeeting.transcript;

      offsetSec += 30;
    }

    $('#transcriptStatus').textContent = `Done · ${currentMeeting.transcript.length.toLocaleString()} characters`;
  } catch (err) {
    $('#transcriptStatus').textContent = `Transcription failed: ${err.message || err}`;
  } finally { btn.disabled = false; }
}

async function doSummarize() {
  if (!currentMeeting?.transcript) {
    $('#summaryStatus').textContent = 'Transcribe the meeting first.';
    return;
  }
  const btn = $('#summarizeBtn');
  btn.disabled = true;
  try {
    const summary = await summarizeTranscript(currentMeeting.transcript, {
      model: summaryModelSelect.value,
      onProgress: (s) => $('#summaryStatus').textContent = s,
    });
    currentMeeting.summary = summary;
    currentMeeting.status = 'summarized';
    await putMeeting(currentMeeting);
    $('#summaryStatus').textContent = 'Done · generated locally';
    renderSummary(summary);
  } catch (err) {
    $('#summaryStatus').textContent = `Summary failed: ${err.message || err}`;
  } finally { btn.disabled = false; }
}

function renderSummary(text) {
  const root = $('#summaryContent');
  root.innerHTML = '';
  if (!text) return;
  const headings = ['OVERVIEW','DECISIONS','ACTION ITEMS','DATES & NUMBERS','RISKS / CONCERNS','OPEN QUESTIONS'];
  const escaped = escapeHtml(text);
  const pattern = new RegExp(`(?:^|\\n)(${headings.map(h => h.replace(/[&/]/g,'\\$&')).join('|')})\\s*\\n`, 'g');
  const matches = [...escaped.matchAll(pattern)];
  if (!matches.length) {
    const div = document.createElement('div');
    div.className = 'summary-block';
    div.innerHTML = `<p>${escaped.replace(/\n/g,'<br>')}</p>`;
    root.appendChild(div);
    return;
  }
  for (let i=0;i<matches.length;i++) {
    const h = matches[i][1];
    const start = matches[i].index + matches[i][0].length;
    const end = i+1 < matches.length ? matches[i+1].index : escaped.length;
    const body = escaped.slice(start,end).trim();
    const div = document.createElement('div');
    div.className = 'summary-block';
    div.innerHTML = `<h3>${h}</h3><p>${body.replace(/\n/g,'<br>')}</p>`;
    root.appendChild(div);
  }
}

async function doAsk() {
  const q = $('#questionInput').value.trim();
  if (!q) return;
  if (!currentMeeting?.transcript) {
    $('#answerBox').textContent = 'Transcribe the meeting first.';
    return;
  }
  const btn = $('#askBtn');
  btn.disabled = true;
  try {
    $('#answerBox').textContent = 'Reading meeting…';
    const ans = await askMeeting(q, currentMeeting.transcript, {
      model: summaryModelSelect.value,
      onProgress: s => $('#answerBox').textContent = s,
    });
    $('#answerBox').textContent = ans;
  } catch (err) {
    $('#answerBox').textContent = `Could not answer: ${err.message || err}`;
  } finally { btn.disabled = false; }
}

$('#startBtn').onclick = startRecording;
$('#stopBtn').onclick = stopRecording;
$('#dimBtn').onclick = () => document.body.classList.toggle('ultra-dim');
$('#backBtn').onclick = async () => { showView(homeView); await refreshHistory(); };
$('#refreshHistoryBtn').onclick = refreshHistory;
$('#settingsBtn').onclick = () => settingsDialog.showModal();
$('#transcribeBtn').onclick = doTranscribe;
$('#summarizeBtn').onclick = doSummarize;
$('#askBtn').onclick = doAsk;

$$('.tab').forEach(tab => tab.onclick = () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  $$('.panel').forEach(x => x.classList.remove('active'));
  tab.classList.add('active');
  $(`#${tab.dataset.tab}Panel`).classList.add('active');
});

document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && mediaRecorder?.state === 'recording') await requestWakeLock();
});

if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
detectCapabilities();
refreshHistory();
