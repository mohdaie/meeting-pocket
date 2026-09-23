import { putMeeting, getMeeting, listMeetings, putChunk, getChunks, deleteMeetingFully } from './db.js';
import { transcribeBlob, summarizeTranscript, askMeeting } from './ai.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const homeView = $('#homeView');
const recordView = $('#recordView');
const meetingView = $('#meetingView');
const timerEl = $('#timer');
const chunkStatus = $('#chunkStatus');
const meetingList = $('#meetingList');
const capabilityPill = $('#capabilityPill');
const selectedMicPill = $('#selectedMicPill');
const recordingMic = $('#recordingMic');
const settingsDialog = $('#settingsDialog');
const historyDialog = $('#historyDialog');
const idleTimerEl = $('#idleTimer');
const audioInputSelect = $('#audioInputSelect');
const refreshMicsBtn = $('#refreshMicsBtn');
const micHelp = $('#micHelp');
const speechModelSelect = $('#speechModelSelect');
const transcriptLanguageSelect = $('#transcriptLanguageSelect');
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
let pendingChunkSaves = [];

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
  const ready = mic && recorder && idb;
  capabilityPill.textContent = ready
    ? `Ready to record locally · ${webgpu ? 'WebGPU available' : 'browser AI fallback mode'}`
    : 'This browser may not support local recording properly.';
}

const MIC_STORAGE_KEY = 'meeting-pocket-audio-input';

function selectedMicId() {
  return localStorage.getItem(MIC_STORAGE_KEY) || '';
}

function selectedMicLabel() {
  if (!audioInputSelect) return 'Automatic';
  const option = audioInputSelect.options[audioInputSelect.selectedIndex];
  return option?.textContent || 'Automatic';
}

function updateSelectedMicUi() {
  if (!selectedMicPill) return;
  const label = selectedMicLabel().replace(/^Automatic.*$/i, 'Automatic');
  selectedMicPill.textContent = `🎤 Microphone: ${label}`;
}

async function refreshAudioInputs({ requestPermission = false } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    if (micHelp) micHelp.textContent = 'This browser cannot list microphone inputs.';
    return;
  }

  let temporaryStream = null;
  if (requestPermission) {
    try {
      temporaryStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (err) {
      if (micHelp) micHelp.textContent = `Microphone access failed: ${err.message || err}`;
      return;
    }
  }

  try {
    const saved = selectedMicId();
    const devices = (await navigator.mediaDevices.enumerateDevices())
      .filter(d => d.kind === 'audioinput');

    audioInputSelect.innerHTML = '';
    const automatic = document.createElement('option');
    automatic.value = '';
    automatic.textContent = 'Automatic · Android chooses';
    audioInputSelect.appendChild(automatic);

    devices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${index + 1}`;
      audioInputSelect.appendChild(option);
    });

    if (saved && devices.some(d => d.deviceId === saved)) {
      audioInputSelect.value = saved;
      if (micHelp) micHelp.textContent = 'Selected microphone will be requested when recording starts.';
    } else {
      audioInputSelect.value = '';
      // Without microphone permission browsers may hide non-default devices.
      // Only discard a stale saved device after an explicit permissioned refresh.
      if (saved && requestPermission) localStorage.removeItem(MIC_STORAGE_KEY);
      if (micHelp) micHelp.textContent = devices.some(d => d.label)
        ? 'Choose earbuds, headset, USB mic, or leave Automatic.'
        : 'Tap Refresh microphones to allow access and reveal device names.';
    }
    updateSelectedMicUi();
  } finally {
    temporaryStream?.getTracks().forEach(track => track.stop());
  }
}

function meetingAudioConstraints(deviceId = '') {
  const audio = {
    echoCancellation: false,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
    sampleRate: 48000,
  };
  if (deviceId) audio.deviceId = { exact: deviceId };
  return audio;
}

async function refreshHistory() {
  const rows = await listMeetings();
  meetingList.innerHTML = '';
  if (!rows.length) {
    meetingList.innerHTML = '<div class="empty">No meetings yet.</div>';
    return;
  }
  for (const m of rows) {
    const row = document.createElement('div');
    row.className = 'meeting-row';

    const btn = document.createElement('button');
    btn.className = 'meeting-item';
    btn.innerHTML = `<strong>${escapeHtml(m.title || 'Meeting')}</strong><small>${fmtDate(m.startedAt)} · ${fmtDuration(m.durationSec || 0)}${m.transcript ? ' · Transcribed' : ''}</small>`;
    btn.onclick = async () => {
      historyDialog.close();
      await openMeeting(m.id);
    };

    const del = document.createElement('button');
    del.className = 'meeting-delete';
    del.type = 'button';
    del.setAttribute('aria-label', `Permanently delete ${m.title || 'meeting'}`);
    del.textContent = 'Delete';
    del.onclick = async (event) => {
      event.stopPropagation();
      const ok = confirm('Permanently delete this meeting?\n\nThe audio recording, transcript and summary will be removed from this device. This cannot be undone.');
      if (!ok) return;

      del.disabled = true;
      del.textContent = 'Deleting…';
      try {
        await deleteMeetingFully(m.id);
        if (currentMeetingId === m.id) {
          currentMeetingId = null;
          currentMeeting = null;
        }
        await refreshHistory();
      } catch (err) {
        del.disabled = false;
        del.textContent = 'Delete';
        alert(`Delete failed: ${err.message || err}`);
      }
    };

    row.append(btn, del);
    meetingList.appendChild(row);
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

  const preferredDeviceId = selectedMicId();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: meetingAudioConstraints(preferredDeviceId),
      video: false,
    });
  } catch (err) {
    // Bluetooth devices can disappear or receive a new browser deviceId.
    // Fall back to Android's active/default microphone rather than blocking recording.
    if (preferredDeviceId && (err?.name === 'OverconstrainedError' || err?.name === 'NotFoundError')) {
      localStorage.removeItem(MIC_STORAGE_KEY);
      audioInputSelect.value = '';
      updateSelectedMicUi();
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: meetingAudioConstraints(''),
          video: false,
        });
      } catch (fallbackErr) {
        alert(`Microphone access failed: ${fallbackErr.message || fallbackErr}`);
        return;
      }
    } else {
      alert(`Microphone access failed: ${err.message || err}`);
      return;
    }
  }

  const activeTrack = mediaStream.getAudioTracks()[0];
  const activeMicLabel = activeTrack?.label || 'Active microphone';
  if (recordingMic) recordingMic.textContent = `🎤 ${activeMicLabel}`;
  if (selectedMicPill) selectedMicPill.textContent = `🎤 Microphone: ${activeMicLabel}`;

  currentMeetingId = crypto.randomUUID();
  startedAt = Date.now();
  chunkIndex = 0;
  stopping = false;
  pendingChunkSaves = [];
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
    inputDeviceLabel: mediaStream.getAudioTracks()[0]?.label || 'Active microphone',
  };

  const mimeType = pickMimeType();
  currentMeeting.mimeType = mimeType || 'audio/webm';
  await putMeeting(currentMeeting);

  const recorderOptions = mimeType
    ? { mimeType, audioBitsPerSecond: 128000 }
    : { audioBitsPerSecond: 128000 };
  mediaRecorder = new MediaRecorder(mediaStream, recorderOptions);
  mediaRecorder.addEventListener('dataavailable', (event) => {
    if (!event.data || !event.data.size) return;
    const idx = chunkIndex++;
    const save = putChunk(currentMeetingId, idx, event.data)
      .then(() => { chunkStatus.textContent = `Saved chunk ${idx + 1} locally`; })
      .catch(() => { chunkStatus.textContent = 'Warning: local save failed'; });
    pendingChunkSaves.push(save);
  });
  mediaRecorder.addEventListener('stop', finishRecording);

  mediaRecorder.start(30000); // 30-second recoverable chunks
  await requestWakeLock();
  showView(recordView);
  document.body.classList.remove('ultra-dim');

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
  mediaRecorder.stop();
}

async function finishRecording() {
  clearInterval(timerHandle);
  timerHandle = null;
  await releaseWakeLock();
  mediaStream?.getTracks().forEach(t => t.stop());
  mediaStream = null;
  await Promise.allSettled(pendingChunkSaves);
  document.body.classList.remove('ultra-dim');
  $('#stopBtn').disabled = false;

  const endedAt = Date.now();
  currentMeeting.endedAt = endedAt;
  currentMeeting.durationSec = Math.max(1, Math.floor((endedAt - startedAt) / 1000));
  currentMeeting.status = 'recorded';
  await putMeeting(currentMeeting);
  idleTimerEl.textContent = '00:00';
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
  $('#questionInput').value = '';
  showView(meetingView);
}

async function doTranscribe() {
  if (!currentMeeting) return;
  const btn = $('#transcribeBtn');
  btn.disabled = true;
  try {
    const rows = await getChunks(currentMeeting.id);
    if (!rows.length) throw new Error('No recorded audio chunks found.');

    // MediaRecorder timeslice blobs are fragments of one continuous WebM/MP4
    // container. Later fragments often cannot be decoded by themselves on Android.
    // Reassemble the original recording first, then decode/transcribe it once.
    const mimeType = currentMeeting.mimeType || rows[0]?.blob?.type || 'audio/webm';
    const completeRecording = new Blob(rows.map(row => row.blob), { type: mimeType });

    const result = await transcribeBlob(completeRecording, {
      model: speechModelSelect.value,
      language: transcriptLanguageSelect.value,
      onProgress: (s) => $('#transcriptStatus').textContent = s,
    });

    currentMeeting.transcript = result.text || '';
    currentMeeting.transcriptChunks = result.chunks || [];
    currentMeeting.status = 'transcribed';
    await putMeeting(currentMeeting);
    $('#transcriptContent').textContent = currentMeeting.transcript;
    $('#transcriptStatus').textContent = currentMeeting.transcript
      ? `Done · ${currentMeeting.transcript.length.toLocaleString()} characters${result.filteredSegments ? ` · ${result.filteredSegments} unclear segment${result.filteredSegments === 1 ? '' : 's'} filtered` : ''}`
      : 'Done · no clear speech detected.';
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
$('#backBtn').onclick = async () => { showView(homeView); await refreshHistory(); };
$('#dimBtn').onclick = () => document.body.classList.toggle('ultra-dim');
$('#settingsBtn').onclick = async () => {
  await refreshAudioInputs();
  settingsDialog.showModal();
};
refreshMicsBtn.onclick = () => refreshAudioInputs({ requestPermission: true });
audioInputSelect.onchange = () => {
  const value = audioInputSelect.value || '';
  if (value) localStorage.setItem(MIC_STORAGE_KEY, value);
  else localStorage.removeItem(MIC_STORAGE_KEY);
  updateSelectedMicUi();
};
$('#historyBtn').onclick = async () => {
  await refreshHistory();
  historyDialog.showModal();
};
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

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', () => {
    if (!mediaStream) refreshAudioInputs().catch(() => {});
  });
}

if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
detectCapabilities();
refreshAudioInputs().catch(() => {});
refreshHistory();
