from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import pickle
import subprocess
import librosa
import torch
import torch.nn as nn
import noisereduce as nr
import whisper

from transformers import AutoFeatureExtractor, WavLMModel, pipeline

app = FastAPI(title="Sukoon Voice Module V2")

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Label Encoder ----------------
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- Load WavLM ----------------
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

# Freeze all layers
for p in wavlm.parameters():
    p.requires_grad = False

# Unfreeze last 4 layers
for layer in wavlm.encoder.layers[-4:]:
    for p in layer.parameters():
        p.requires_grad = True


# ---------------- Emotion Classifier ----------------
class EmotionClassifier(nn.Module):
    def __init__(self, wavlm, num_classes):
        super().__init__()
        self.wavlm = wavlm
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            wavlm.config.hidden_size,
            num_classes
        )

    def forward(self, input_values, attention_mask):
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask
        )

        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = self.dropout(embeddings)

        return self.classifier(embeddings)


model = EmotionClassifier(
    wavlm,
    len(label_encoder.classes_)
).to(device)

model.load_state_dict(
    torch.load(
        "models/sukoon_wavlm_v2.pt",
        map_location=device
    )
)

model.eval()

# ---------------- Whisper ----------------
whisper_model = whisper.load_model("small")

# ---------------- Text Emotion ----------------
text_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

TEXT_TO_WAVLM = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised"
}


# ---------------- Audio Preprocessing ----------------
def preprocess_audio(path):

    audio, sr = librosa.load(path, sr=None)

    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Noise reduction
    audio = nr.reduce_noise(
        y=audio,
        sr=16000
    )

    return audio


# ---------------- Voice Prediction ----------------
def predict_voice(audio_path):

    audio = preprocess_audio(audio_path)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        logits = model(
            inputs.input_values.to(device),
            inputs.attention_mask.to(device)
        )

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    emotion = label_encoder.inverse_transform([pred.item()])[0]

    return emotion, conf.item()


# ---------------- Home ----------------
@app.get("/")
def home():
    return {
        "message": "Sukoon Voice API is running"
    }


# ---------------- Predict ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    os.makedirs("temp", exist_ok=True)

    temp_path = os.path.join("temp", file.filename)

    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------- 60-second limit ----------
    duration = librosa.get_duration(path=temp_path)

    if duration > 60:
        os.remove(temp_path)
        raise HTTPException(
            status_code=400,
            detail="Audio must be 60 seconds or shorter."
        )

    ext = os.path.splitext(temp_path)[1].lower()

    # ---------- Convert MP3/M4A/MP4 ----------
    if ext in [".m4a", ".mp3", ".mp4"]:

        ffmpeg_path = shutil.which("ffmpeg")

        if ffmpeg_path is None:
            os.remove(temp_path)
            raise HTTPException(
                status_code=500,
                detail="FFmpeg is not available on this server."
            )

        wav_path = os.path.splitext(temp_path)[0] + ".wav"

        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i", temp_path,
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        os.remove(temp_path)
        temp_path = wav_path

    # ---------- Voice emotion ----------
    voice_emotion, voice_conf = predict_voice(temp_path)

    # ---------- Whisper transcription ----------
    # Use the already preprocessed waveform instead of letting Whisper call FFmpeg
    audio_for_whisper = preprocess_audio(temp_path)
    transcript = whisper_model.transcribe(audio_for_whisper)["text"].strip()

    # ---------- Text emotion ----------
    text_pred = text_model(transcript)[0][0]
    text_emotion = TEXT_TO_WAVLM.get(text_pred["label"], "neutral")
    text_conf = text_pred["score"]

    # ---------- Fusion ----------
    if text_conf > voice_conf:
        final_emotion = text_emotion
        confidence = text_conf
    else:
        final_emotion = voice_emotion
        confidence = voice_conf

    # ---------- Cleanup ----------
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "emotion": final_emotion,
        "confidence": round(confidence, 2),
        "transcript": transcript
    }