from faster_whisper import WhisperModel
import json

print("📦 Loading Whisper model...")
model = WhisperModel("medium", compute_type="int8", device="cpu")

print("🎧 Transcribing audio file...")
segments, info = model.transcribe("Senior Project Demo.mp3")

print(f"🔍 Transcription language: {info.language}")
print(f"🕒 Duration: {info.duration:.2f} seconds")

print("🧠 Collecting results...")
text = ""
data = []

for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")  # Print each segment
    text += seg.text + " "
    data.append({
        "start": seg.start,
        "end": seg.end,
        "text": seg.text
    })

print("💾 Saving transcript...")
with open("full_transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("segments.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ Transcription done! Output saved.")
