
import whisper
import json

# Load Whisper model
model = whisper.load_model("medium")

# Path to your audio file
audio_path = "Senior Project Demo.mp3"
result = model.transcribe(audio_path)

# Save full transcript to text file
with open("full_transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

# Save segments to JSON file
with open("segments.json", "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, indent=2)

print("✅ Transcription complete. Output saved to full_transcript.txt and segments.json")
