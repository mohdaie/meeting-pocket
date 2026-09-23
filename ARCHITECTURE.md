# Architecture

```text
Microphone
   ↓
MediaRecorder (30 s chunks)
   ↓
IndexedDB ───────────────→ crash recovery / meeting history
   ↓
Assemble meeting Blob
   ↓
Transformers.js
   ├── Whisper multilingual → transcript
   └── Qwen2.5 0.5B → chunk summaries → final brief
                           └→ Ask This Meeting
```

## Reliability rule
Recording is the critical path. AI must never prevent recording. Models are lazy-loaded only after a meeting is recorded.

## Recommended next iterations
- V0.2: background transcription queue without competing with MediaRecorder.
- V0.3: speaker-labeling/diarization experiment.
- V0.4: export Markdown / TXT / audio.
- V0.5: optional encrypted sync.
- V1.0: Android native wrapper only if true screen-off recording is required.
