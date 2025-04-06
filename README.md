
# Meeting Analyzer

An automated meeting analysis tool that transcribes audio, identifies speakers, extracts action items, and generates detailed reports with personalized email distribution.

---

## 🚀 Features

- 🎙️ Audio transcription using Whisper
- 👥 Speaker identification and verification
- ✅ Action item and task extraction
- 📊 Markdown + PDF meeting report generation
- 📧 Email distribution of personalized summaries
- 👤 Speaker profile management (with email memory)
- 🔐 `.env` support for secure API key + credentials
- 🌐 Ready for n8n automation via webhooks (coming soon)

---

## 🔧 Prerequisites

1. Python 3.10.9 or later  
2. FFmpeg installed and available in PATH  
3. [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) for PDF generation  
4. OpenAI API key  
5. Gmail account with **App Password** for email sending  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/meeting-analyzer.git
cd meeting-analyzer
```

### 2. Set up a virtual environment
```bash
python -m venv whisper-env
.\whisper-env\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install `wkhtmltopdf`
- Download and install: [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html)
- Add the install folder (e.g., `C:\Program Files\wkhtmltopdf\bin`) to your **system PATH**

---

## 🔐 Configure Credentials

### Option 1: Using `.env` (Recommended)

1. Copy and rename `.env.example`:
```bash
cp .env.example .env
```

2. Open `.env` and fill in:
```ini
OPENAI_API_KEY=your-openai-key
EMAIL_USERNAME=your.email@gmail.com
EMAIL_PASSWORD=your-app-password
FROM_EMAIL=your.email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

> ✅ `.env` is ignored by Git and keeps your keys safe.

---

## 🧠 Usage

### ▶️ Option 1: Batch Files (Windows only)

#### `run_meeting.bat`
Runs the full analysis pipeline with default audio file (`Senior Project Demo.mp3`)
```bash
.\run_meeting.bat
```

#### `run_analyzer.bat`
Custom run with different settings or input
```bash
.\run_analyzer.bat
```

---

### 🐍 Option 2: Python Execution (More flexible)

Activate your environment:
```bash
.\whisper-env\Scripts\activate
```

Then run:
```bash
python meeting_analyzer.py "your_audio.mp3"
```

Optionally add:
```bash
--email-config email_config.json
```

> Will use `.env` values unless overridden with `--email-config`

---

## 📁 Output Files

- `meeting_report.md` → Markdown summary  
- `meeting_report.pdf` → Printable version  
- `full_transcript.txt` → Raw transcript  
- `segments.json` → Timestamped segments  
- `meeting_analysis.log` → Execution logs  
- `individual_summaries/` → Per-speaker task summaries

---

## ⚙️ Key Config Files

| File | Purpose |
|------|---------|
| `.env` | Main API + email config (recommended) |
| `.env.example` | Template for `.env` setup |
| `email_config.json` | Optional legacy email config |
| `attendees_db.json` | Stores speaker names + emails |
| `requirements.txt` | Python dependencies |

---

## 🔐 Security Notes

- ✅ `.env` is excluded from Git
- ❌ Never commit your credentials
- 🔒 Use Gmail **App Passwords** (not your main password)
- 🔄 You can remove `email_config.json` if `.env` is used

---

## 🔮 Roadmap (Coming Soon)

- 🌐 **n8n integration** via webhooks + JSON payloads  
- 🧠 AI task suggestion templates  
- 🔁 Audio highlight replays  
- 🧑‍💼 Dashboard summaries per participant  
- 📎 Google Calendar or Notion integration

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| ❌ PDF not generated | Ensure `wkhtmltopdf` is installed + in PATH |
| ❌ Emails not sending | Check app password + Gmail 2FA enabled |
| ❌ Transcription error | Check `ffmpeg` is installed + audio format |
| ❌ Batch file fails | Use full path, run as admin if needed |

---

## 🤝 Contributing

Pull Requests are welcome!  
Want to add integrations or custom formats? Fork and build on top!

---

## 📜 License

MIT or your license here

---

## 📬 Contact

Brendan Bryan & Brian Ford

---
