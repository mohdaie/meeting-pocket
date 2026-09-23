# Architecture — v0.4.0

```text
Microphone / earbuds microphone
   ↓
MediaRecorder (30 s recoverable chunks)
   ↓
IndexedDB ───────────────→ local meeting history / permanent delete
   ↓
Assemble meeting Blob after recording
   ↓
Transcription router (Auto)
   ├── Gemini 3.5 Transcribe
   │     └── EN / MS / ZH / TA code-switching, automatic multilingual STT
   ├── Groq Whisper Large V3 (optional fallback)
   └── Local Transformers.js Whisper (offline fallback)
            ↓
        Transcript
            ↓
Local Qwen2.5 0.5B
   ├── Meeting brief
   └── Ask This Meeting
```

## Reliability rules

1. Recording must never depend on AI availability.
2. Audio is saved locally before transcription starts.
3. Auto transcription degrades gracefully: Gemini → Groq → local Whisper.
4. API keys are never committed to the repository.
5. Provider failures must leave the original recording intact for retry.
6. PWA shell uses network-first refresh with cached offline fallback.

## Security note

The current static GitHub Pages deployment supports personal BYOK configuration stored on-device. A shared production credential must be moved behind a backend proxy or an equivalent server-side secret boundary.
