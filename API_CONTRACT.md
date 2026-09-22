
# Sukoon Voice API Contract

## Status

Voice Module is working locally using FastAPI.

## Base URL (Local)

http://127.0.0.1:8000

## Endpoint

### POST /predict

Analyzes an uploaded speech recording and returns the detected emotion, confidence score, and transcript.

### Request

- Method: POST
- Content-Type: multipart/form-data
- Field name: file

### Supported Audio

- WAV
- MP3
- M4A (with FFmpeg)
- MP4 (audio extraction)

Maximum duration: 60 seconds.

### Response

```json
{
  "emotion": "happy",
  "confidence": 0.94,
  "transcript": "Today was a good day."
}
```

### Response Fields

- `emotion` – Final detected emotion
- `confidence` – Model confidence (0–1)
- `transcript` – Whisper transcription

## Example JavaScript Request

```javascript
const formData = new FormData();
formData.append("file", audioFile);

fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(console.log);
```