# Meeting Pocket PWA — v0.1.0

A local-first meeting companion: record → transcribe → summarize → ask questions.

## Why this architecture

- **No paid AI API by default.** AI runs in the browser with Transformers.js.
- **Audio is stored locally in IndexedDB.** 30-second chunks reduce the chance of losing a long meeting after a crash.
- **Whisper multilingual** handles English/Malay mixed meetings better than English-only models.
- **Qwen2.5 0.5B Instruct** is the local summary/Q&A model, using quantized WebGPU inference when available.
- **Near-black recording UI** and no app-generated sound/vibration. The app does not attempt to hide OS microphone/privacy indicators.
- **Screen Wake Lock is optional and enabled by default** because phones may suspend browser work when the screen is off. The UI stays nearly black instead.

## First use

The first transcription/summary downloads local AI model assets from Hugging Face. Browser/model caching makes later runs much faster.

Default AI:
- Speech: `onnx-community/whisper-base`
- Summary/Q&A: `onnx-community/Qwen2.5-0.5B-Instruct`
- Browser runtime: `@huggingface/transformers@4.3.0`

For a faster first test, switch speech model in Settings to Whisper Tiny multilingual.

## Current V0.1 scope

1. Start meeting
2. Microphone recording
3. 30-second crash-recoverable IndexedDB chunks
4. Near-black recording UI
5. Optional screen wake lock
6. Local Whisper transcription
7. Chunked local Qwen meeting summary
8. Ask-this-meeting Q&A
9. PWA manifest and offline app-shell cache

## Important mobile limitation

A PWA cannot guarantee reliable microphone recording after the OS fully turns the screen off or aggressively suspends the browser. V0.1 keeps the display awake when possible while rendering an almost-black interface. If true screen-off/background recording becomes a hard requirement, the next step should be a native Android wrapper/service.

## Privacy / recording consent

The app is designed for local processing. Users remain responsible for workplace policies and applicable recording/consent rules. The app must not attempt to suppress Android/iOS microphone privacy indicators.
