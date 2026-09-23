# Meeting Pocket PWA — v0.4.0

A quiet meeting companion: record → multilingual transcribe → summarize → ask questions.

## v0.4.0 transcription architecture

Meeting Pocket no longer asks you to choose one meeting language.

**Auto mode** uses this order:

1. **Gemini 3.5 Transcribe** — preferred. Automatic multilingual/code-switching transcription for English, Malay, Mandarin and Tamil.
2. **Groq Whisper Large V3** — optional cloud fallback when a Groq key is configured.
3. **Local Whisper** — on-device fallback that works without an API key.

The provider that actually produced a transcript is shown in the transcript status.

## API keys

The public GitHub repository contains **no Gemini or Groq API key**.

For personal use, keys entered in Settings are stored in this browser's localStorage on this device and sent only to the selected provider when transcription runs. They are not written into meeting records, committed to GitHub, or synchronized by Meeting Pocket.

For a public multi-user production deployment, move provider credentials behind a backend proxy rather than embedding a shared key in the PWA.

## Recording and storage

- Audio recording uses MediaRecorder.
- 30-second recoverable chunks are stored locally in IndexedDB.
- A meeting is reassembled only when transcription is requested.
- Meeting audio, transcript and summary can be permanently deleted from IndexedDB.
- Cloud transcription sends the assembled recording to the configured provider.
- Gemini temporary uploaded files are deleted after transcription when the Files API path is used.
- Local mode keeps transcription entirely on-device.

## Local AI fallback

- Speech: `onnx-community/whisper-small` by default
- Summary/Q&A: `onnx-community/Qwen2.5-0.5B-Instruct`
- Runtime: `@huggingface/transformers@4.3.0`
- WebGPU is used when available; WASM is the fallback.

## Mobile reliability

Recording is always the critical path. AI is loaded or contacted only after the recording has stopped, so transcription failure should not prevent the audio from being saved.

A PWA cannot guarantee microphone recording after the OS fully turns the display off or suspends the browser. Meeting Pocket can request a screen wake lock and render a nearly black recording UI instead.

## Privacy / recording consent

Users remain responsible for workplace policies and applicable recording/consent rules. Meeting Pocket does not attempt to suppress Android/iOS microphone privacy indicators.
