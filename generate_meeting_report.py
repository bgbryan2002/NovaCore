import os
import openai
from pathlib import Path

# ✅ Load API key from environment variable (set this in PowerShell beforehand!)
client = openai.OpenAI(api_key=os.getenv("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"))

# ✅ Load the transcript file
transcript_path = Path("full_transcript.txt")
transcript = transcript_path.read_text(encoding="utf-8")

# ✅ Define your structured prompt
prompt = f"""
You are an AI assistant that generates structured meeting reports.

Based on this transcript:

\"\"\"{transcript}\"\"\"

Write a report with the following sections:
1. A concise **Meeting Summary**
2. A list of 3–5 **Key Audio Highlights** with timestamps if mentioned
3. A list of **Tasks** (explicit or implied)
4. A list of **Action Items / Next Steps**

Format the report in Markdown.
"""

# ✅ Call the GPT-4 Turbo API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # ← Works on all API keys
    messages=[
        {"role": "system", "content": "You are a helpful meeting assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
)


# ✅ Extract and save the response
report_md = response.choices[0].message.content
Path("meeting_report.md").write_text(report_md, encoding="utf-8")

print("✅ Report saved as meeting_report.md")
