const CACHE = 'meeting-pocket-shell-v0.3.3-r1';
const SHELL = ['./','./index.html','./styles.css','./app.js','./db.js','./ai.js','./manifest.webmanifest','./icons/icon.svg'];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
    const clone = res.clone();
    caches.open(CACHE).then(c => c.put(req, clone));
    return res;
  })));
});
