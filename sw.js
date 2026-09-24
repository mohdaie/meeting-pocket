const CACHE = 'meeting-pocket-shell-v0.5.0-r1';
const SHELL = [
  './',
  './index.html',
  './styles.css?v=0.5.0',
  './app.js?v=0.5.0',
  './db.js?v=0.5.0',
  './ai.js?v=0.5.0',
  './groq-usage.js?v=0.5.0',
  './manifest.webmanifest',
  './icons/icon.svg'
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;

  // Network-first prevents installed PWA refreshes from getting stuck on an
  // older Meeting Pocket build. Cached shell remains available offline.
  event.respondWith(
    fetch(req, { cache: 'no-store' })
      .then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(req, clone));
        }
        return response;
      })
      .catch(async () => {
        const exact = await caches.match(req);
        if (exact) return exact;
        const withoutQuery = await caches.match(req, { ignoreSearch: true });
        if (withoutQuery) return withoutQuery;
        throw new Error('Offline resource unavailable');
      })
  );
});
