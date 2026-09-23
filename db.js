const DB_NAME = 'meeting-pocket-db';
const DB_VERSION = 1;
const MEETINGS = 'meetings';
const CHUNKS = 'chunks';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(MEETINGS)) {
        db.createObjectStore(MEETINGS, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(CHUNKS)) {
        const store = db.createObjectStore(CHUNKS, { keyPath: ['meetingId', 'index'] });
        store.createIndex('byMeeting', 'meetingId', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export async function putMeeting(meeting) {
  const db = await openDb();
  const tx = db.transaction(MEETINGS, 'readwrite');
  tx.objectStore(MEETINGS).put(meeting);
  await txDone(tx);
  db.close();
  return meeting;
}

export async function getMeeting(id) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MEETINGS, 'readonly');
    const req = tx.objectStore(MEETINGS).get(id);
    req.onsuccess = () => { resolve(req.result || null); db.close(); };
    req.onerror = () => { reject(req.error); db.close(); };
  });
}

export async function listMeetings() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MEETINGS, 'readonly');
    const req = tx.objectStore(MEETINGS).getAll();
    req.onsuccess = () => {
      const rows = (req.result || []).sort((a,b) => b.startedAt - a.startedAt);
      resolve(rows);
      db.close();
    };
    req.onerror = () => { reject(req.error); db.close(); };
  });
}

export async function putChunk(meetingId, index, blob) {
  const db = await openDb();
  const tx = db.transaction(CHUNKS, 'readwrite');
  tx.objectStore(CHUNKS).put({ meetingId, index, blob, size: blob.size, createdAt: Date.now() });
  await txDone(tx);
  db.close();
}

export async function getChunks(meetingId) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS, 'readonly');
    const idx = tx.objectStore(CHUNKS).index('byMeeting');
    const req = idx.getAll(IDBKeyRange.only(meetingId));
    req.onsuccess = () => {
      resolve((req.result || []).sort((a,b) => a.index - b.index));
      db.close();
    };
    req.onerror = () => { reject(req.error); db.close(); };
  });
}


export async function deleteMeetingFully(meetingId) {
  const db = await openDb();
  const tx = db.transaction([MEETINGS, CHUNKS], 'readwrite');
  const meetings = tx.objectStore(MEETINGS);
  const chunks = tx.objectStore(CHUNKS);
  const byMeeting = chunks.index('byMeeting');

  meetings.delete(meetingId);

  const cursorReq = byMeeting.openCursor(IDBKeyRange.only(meetingId));
  cursorReq.onsuccess = () => {
    const cursor = cursorReq.result;
    if (!cursor) return;
    cursor.delete();
    cursor.continue();
  };

  await txDone(tx);
  db.close();
}
