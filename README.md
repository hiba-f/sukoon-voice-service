
# Sukoon Voice Module API

FastAPI-based Voice Emotion Recognition service for the Sukoon mental wellbeing platform.

## Status

- WavLM emotion detection
- Whisper speech-to-text
- Emotion fusion
- FastAPI service
- Local API tested
- GitHub repository ready

## Local URL

`http://127.0.0.1:8000`

---

## Endpoint

### POST `/predict`

Predicts emotion from an uploaded speech recording.

### Request

- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Field:** `file`

### Supported Formats

- WAV
- MP3
- M4A *(requires FFmpeg when running locally)*
- MP4 *(audio extraction via FFmpeg)*

### Audio Limit

Maximum duration: **60 seconds**

---

## Response

Example:

```json
{
  "emotion": "happy",
  "confidence": 0.94,
  "transcript": "Today was a good day."
}
```

### Response Fields

| Field | Description |
|--------|-------------|
| emotion | Final detected emotion |
| confidence | Model confidence (0–1) |
| transcript | Whisper transcription |

---

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

---

## Tech Stack

- FastAPI
- WavLM Base
- OpenAI Whisper
- Hugging Face Transformers
- Librosa
- PyTorch
- Noisereduce