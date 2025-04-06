import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import openai
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from tqdm import tqdm
import markdown
import pdfkit
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meeting_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure OpenAI client with timeout and API key from environment
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.timeout = 30  # 30 seconds timeout

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai_with_retry(client, **kwargs):
    """Make OpenAI API call with retry logic."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise

class AttendeeDatabase:
    def __init__(self, db_file: str = "attendees_db.json"):
        self.db_file = db_file
        self.attendees = self.load_database()

    def load_database(self) -> Dict:
        """Load the attendee database from JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)["attendees"]
        return {}

    def save_database(self) -> None:
        """Save the current state of the database to JSON file."""
        with open(self.db_file, 'w') as f:
            json.dump({"attendees": self.attendees}, f, indent=4)

    def get_attendee(self, name: str) -> Optional[Dict]:
        """Get attendee information by name."""
        return self.attendees.get(name)

    def add_or_update_attendee(self, name: str, email: str, role: str = "Team Member") -> None:
        """Add a new attendee or update existing attendee information."""
        self.attendees[name] = {
            "name": name,
            "email": email,
            "role": role,
            "last_meeting": datetime.now().isoformat()
        }
        self.save_database()

@dataclass
class Speaker:
    name: str
    speaking_time: float
    segments: List[Dict]
    tasks: List[str]
    action_items: List[str]
    future_references: List[Dict]
    email: Optional[str] = None

class MeetingAnalyzer:
    def __init__(self, email_config):
        """Initialize the meeting analyzer with API keys and models."""
        self.openai_client = openai
        self.speakers: Dict[str, Speaker] = {}
        self.attendee_db = AttendeeDatabase()
        self.email_config = email_config
        
    def transcribe_audio(self, audio_file: str) -> Tuple[str, List[Dict]]:
        """Transcribe audio using faster_whisper."""
        logger.info(f"Transcribing audio file: {audio_file}")
        
        try:
            # Import here to avoid loading model unless needed
            from faster_whisper import WhisperModel
            print("\nLoading Whisper model...")
            with tqdm(total=1, desc="Loading model", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                model = WhisperModel("medium", compute_type="int8", device="cpu")
                pbar.update(1)
            
            print("\nProcessing audio file...")
            segments, info = model.transcribe(audio_file)
            logger.info(f"Processing audio with duration {info.duration:.2f}s")
            
            # Collect results
            full_text = ""
            segments_data = []
            
            # Convert segments iterator to list to get total count
            segments_list = list(segments)
            total_segments = len(segments_list)
            
            print("\nProcessing segments...")
            with tqdm(total=total_segments, desc="Processing segments", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for seg in segments_list:
                    full_text += seg.text + " "
                    segments_data.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    })
                    pbar.update(1)
            
            print("\nSaving outputs...")
            with tqdm(total=2, desc="Saving files", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                # Save outputs
                with open("full_transcript.txt", "w", encoding="utf-8") as f:
                    f.write(full_text)
                pbar.update(1)
                
                with open("segments.json", "w", encoding="utf-8") as f:
                    json.dump(segments_data, f, indent=2)
                pbar.update(1)
                
            logger.info(f"Transcription completed. Duration: {info.duration:.2f}s")
            return full_text, segments_data
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def identify_speakers(self, segments: List[Dict]) -> Dict[str, Speaker]:
        """Identify speakers from the transcript and manage their information in the database."""
        logger.info("Identifying speakers from transcript")
        print("\nAnalyzing transcript for speaker identification...")
        
        # Extract potential speaker names using GPT
        prompt = """
        Analyze this meeting transcript and identify the distinct speakers. 
        For each speaker found, provide:
        1. Their full name
        2. How you identified them (e.g., self-introduction, mentioned by others)
        3. Confidence level (high/medium/low)
        
        Format the response as JSON:
        {
            "speakers": [
                {
                    "name": "string",
                    "identification_method": "string",
                    "confidence": "string"
                }
            ]
        }
        """
        
        try:
            print("Sending transcript to OpenAI for speaker identification...")
            with tqdm(total=1, desc="Analyzing speakers", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                response = call_openai_with_retry(
                    self.openai_client,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a meeting transcript analyzer. Always respond with valid JSON."},
                        {"role": "user", "content": prompt + "\n\nTranscript:\n" + json.dumps(segments)}
                    ],
                    temperature=0.3
                )
                pbar.update(1)

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No response content from OpenAI API")

            try:
                speakers_info = json.loads(response.choices[0].message.content.strip('`').replace('json\n', ''))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response: {response.choices[0].message.content}")
                raise ValueError("Invalid JSON response from OpenAI API") from e

            if not isinstance(speakers_info, dict) or "speakers" not in speakers_info:
                raise ValueError("Unexpected response format from OpenAI API")

            print(f"\nIdentified {len(speakers_info['speakers'])} speakers:")
            
            # Initialize speaker objects and check database
            with tqdm(total=len(speakers_info["speakers"]), desc="Processing speakers", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                for speaker in speakers_info["speakers"]:
                    name = speaker["name"]
                    print(f"\n- {name} ({speaker['confidence']} confidence)")
                    
                    # Confirm speaker name
                    while True:
                        confirm = input(f"  Is '{name}' accurate for this speaker? (Y/N): ").strip().upper()
                        if confirm == 'Y':
                            break
                        elif confirm == 'N':
                            name = input("  Please enter the correct name: ").strip()
                            break
                        else:
                            print("  Please enter Y or N")
                    
                    # Check if speaker exists in database
                    attendee_info = self.attendee_db.get_attendee(name)
                    
                    if not attendee_info:
                        # New speaker found - prompt for information
                        logger.info(f"New speaker detected: {name}")
                        email = input(f"  Please enter email address for {name}: ")
                        role = input(f"  Please enter role for {name} (press Enter for 'Team Member'): ") or "Team Member"
                        self.attendee_db.add_or_update_attendee(name, email, role)
                        attendee_info = self.attendee_db.get_attendee(name)
                    
                    self.speakers[name] = Speaker(
                        name=name,
                        speaking_time=0,
                        segments=[],
                        tasks=[],
                        action_items=[],
                        future_references=[],
                        email=attendee_info["email"]
                    )
                    pbar.update(1)
            
            # Assign segments to speakers using GPT
            print("\nAnalyzing individual segments...")
            with tqdm(total=len(segments), desc="Processing segments", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for segment in segments:
                    prompt = f"""
                    Who is the most likely speaker of this segment? Choose from: {list(self.speakers.keys())}
                    
                    Segment: {segment['text']}
                    
                    Reply with just the speaker's full name.
                    """
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a speaker identification expert. Respond with only the speaker's name."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    
                    speaker_name = response.choices[0].message.content.strip()
                    if speaker_name in self.speakers:
                        self.speakers[speaker_name].segments.append(segment)
                        self.speakers[speaker_name].speaking_time += segment["end"] - segment["start"]
                    pbar.update(1)
            
            logger.info(f"Identified {len(self.speakers)} speakers")
            return self.speakers
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {str(e)}")
            raise

    def extract_action_items(self, speaker: Speaker) -> None:
        """Extract action items and tasks for a specific speaker."""
        logger.info(f"Extracting action items for {speaker.name}")
        
        # Split speaker text into chunks if it's too long
        max_chunk_size = 4000  # Maximum size for each chunk
        speaker_text = "\n".join([segment["text"] for segment in speaker.segments])
        text_chunks = [speaker_text[i:i + max_chunk_size] for i in range(0, len(speaker_text), max_chunk_size)]
        
        all_items = {
            "action_items": [],
            "tasks": [],
            "future_references": []
        }
        
        for chunk in text_chunks:
            prompt = f"""
            Analyze the following text from {speaker.name} in a meeting and identify:
            1. Action items or tasks they committed to do
            2. Future references or follow-ups they mentioned
            3. Any deadlines mentioned
            
            Format the response as JSON with this structure:
            {{
                "action_items": [
                    "specific action item or task"
                ],
                "tasks": [
                    "specific task"
                ],
                "future_references": [
                    {{
                        "description": "what needs to be done",
                        "deadline": "deadline if mentioned, or null",
                        "context": "relevant context"
                    }}
                ]
            }}
            
            If there are no items in a category, return an empty list. Ensure the response is valid JSON.
            """
            
            try:
                response = call_openai_with_retry(
                    self.openai_client,
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a meeting analyzer that extracts action items and tasks. Always respond with valid JSON."},
                        {"role": "user", "content": prompt + "\n\nText to analyze:\n" + chunk}
                    ],
                    temperature=0.3
                )
                
                if not response.choices or not response.choices[0].message.content:
                    logger.error(f"No response content from OpenAI API for {speaker.name}")
                    continue
                
                # Clean up the response content
                content = response.choices[0].message.content.strip()
                content = content.replace('```json\n', '').replace('```', '').strip()
                
                try:
                    items = json.loads(content)
                    
                    # Validate and merge items
                    if isinstance(items, dict):
                        all_items["action_items"].extend(items.get("action_items", []))
                        all_items["tasks"].extend(items.get("tasks", []))
                        all_items["future_references"].extend(items.get("future_references", []))
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse action items for {speaker.name}: {str(e)}\nResponse content: {content}")
                
            except Exception as e:
                logger.error(f"Action item extraction failed for chunk of {speaker.name}: {str(e)}")
        
        # Update speaker with combined results
        speaker.action_items = list(set(all_items["action_items"]))  # Remove duplicates
        speaker.tasks = list(set(all_items["tasks"]))
        speaker.future_references = all_items["future_references"]  # Keep all references

    def generate_meeting_report(self, output_file: str = "meeting_report.md") -> Tuple[str, bool]:
        """Generate a comprehensive meeting report in Markdown format and optionally PDF."""
        logger.info("Generating meeting report")
        
        report = "# Meeting Summary Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add participant summary
        report += "## Participants\n\n"
        for speaker in self.speakers.values():
            speaking_minutes = speaker.speaking_time / 60
            report += f"- **{speaker.name}** (Speaking time: {speaking_minutes:.1f} minutes)\n"
        
        # Add key points and action items by speaker
        report += "\n## Action Items and Tasks by Participant\n\n"
        for speaker in self.speakers.values():
            report += f"### {speaker.name}\n\n"
            
            if speaker.tasks:
                report += "#### Tasks:\n"
                for task in speaker.tasks:
                    report += f"- {task}\n"
                report += "\n"
            
            if speaker.action_items:
                report += "#### Action Items:\n"
                for item in speaker.action_items:
                    report += f"- {item}\n"
                report += "\n"
            
            if speaker.future_references:
                report += "#### Future References:\n"
                for ref in speaker.future_references:
                    deadline = ref["deadline"] if ref["deadline"] else "No deadline specified"
                    report += f"- {ref['description']} (Deadline: {deadline})\n"
                    report += f"  Context: {ref['context']}\n"
                report += "\n"
        
        # Save the markdown report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        # Generate PDF version
        pdf_generated = False
        pdf_file = output_file.replace('.md', '.pdf')
        try:
            # Convert markdown to HTML
            html = markdown.markdown(report)
            
            # Save HTML temporarily
            html_file = output_file.replace('.md', '.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #34495e; border-bottom: 1px solid #eee; }}
                        h3 {{ color: #7f8c8d; }}
                    </style>
                </head>
                <body>
                    {html}
                </body>
                </html>
                """)
            
            try:
                # Try to find wkhtmltopdf in common installation paths
                wkhtmltopdf_paths = [
                    r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                    r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
                    "wkhtmltopdf"  # Try system PATH
                ]
                
                config = None
                for path in wkhtmltopdf_paths:
                    if os.path.exists(path):
                        config = pdfkit.configuration(wkhtmltopdf=path)
                        break
                
                # Convert HTML to PDF
                if config:
                    pdfkit.from_file(html_file, pdf_file, configuration=config)
                    pdf_generated = True
                else:
                    raise FileNotFoundError("wkhtmltopdf not found in common installation paths")
                
            except Exception as e:
                logger.error(f"Failed to generate PDF with wkhtmltopdf: {str(e)}")
            
            # Clean up temporary HTML file
            os.remove(html_file)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
        
        logger.info(f"Meeting report saved to {output_file}" + (f" and {pdf_file}" if pdf_generated else " (PDF generation failed)"))
        return report, pdf_generated

    def send_email_summaries(self) -> None:
        """Send individual summaries to each participant."""
        logger.info("Sending email summaries to participants")
        
        try:
            # Setup SMTP connection
            smtp = smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port)
            smtp.starttls()
            smtp.login(self.email_config.username, self.email_config.password)
            
            # Read the full report files
            with open("meeting_report.md", "r", encoding="utf-8") as f:
                md_content = f.read()
            
            # Try to read PDF if it exists
            pdf_content = None
            try:
                with open("meeting_report.pdf", "rb") as f:
                    pdf_content = f.read()
            except FileNotFoundError:
                logger.warning("PDF report not found, will send only markdown version")
            
            for speaker in self.speakers.values():
                if not speaker.email:
                    logger.warning(f"No email address found for {speaker.name}, skipping summary")
                    continue
                
                # Create personalized summary
                subject = "Meeting Summary and Action Items"
                
                # Create task list
                tasks_list = "\n".join([f"- {task}" for task in speaker.tasks]) or "No tasks assigned"
                action_items_list = "\n".join([f"- {item}" for item in speaker.action_items]) or "No action items"
                future_list = "\n".join([f"- {ref['description']} (Deadline: {ref['deadline'] or 'Not specified'})" for ref in speaker.future_references]) or "No future commitments"
                
                body = f"""Hello {speaker.name},

Here's your summary from the recent meeting:

Your Tasks:
{tasks_list}

Your Action Items:
{action_items_list}

Your Future Commitments:
{future_list}

The complete meeting report is attached in {"both Markdown and PDF formats" if pdf_content else "Markdown format"} for your reference.

Best regards,
Meeting Analyzer"""
                
                # Create message
                msg = MIMEMultipart()
                msg["From"] = self.email_config.from_email
                msg["To"] = speaker.email
                msg["Subject"] = subject
                
                # Attach main message body
                msg.attach(MIMEText(body, "plain"))
                
                # Attach Markdown report
                md_attachment = MIMEText(md_content, "plain")
                md_attachment.add_header('Content-Disposition', 'attachment', filename='meeting_report.md')
                msg.attach(md_attachment)
                
                # Attach PDF report if available
                if pdf_content:
                    pdf_attachment = MIMEApplication(pdf_content, _subtype="pdf")
                    pdf_attachment.add_header('Content-Disposition', 'attachment', filename='meeting_report.pdf')
                    msg.attach(pdf_attachment)
                
                # Send email
                smtp.send_message(msg)
                logger.info(f"Sent summary email to {speaker.name} ({speaker.email})")
            
            smtp.quit()
            logger.info("All email summaries sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email summaries: {str(e)}")
            raise

def main(audio_file: str, openai_key: str = None, email_config_file: str = None):
    """Main function to process the audio file and generate reports."""
    try:
        # Set OpenAI key if provided, otherwise use environment variable
        if openai_key:
            openai.api_key = openai_key
        
        # Initialize email configuration
        email_config = EmailConfig(email_config_file)
        
        # Create analyzer instance
        analyzer = MeetingAnalyzer(email_config)
        
        # Process the audio file
        logger.info("Starting transcription")
        segments = analyzer.transcribe_audio(audio_file)
        
        # Identify speakers
        speakers = analyzer.identify_speakers(segments)
        
        # Extract action items for each speaker
        logger.info("Extracting action items")
        for speaker in speakers.values():
            analyzer.extract_action_items(speaker)
        
        # Generate and save the meeting report
        logger.info("Generating meeting report")
        md_file, pdf_generated = analyzer.generate_meeting_report()
        
        # Send email summaries
        logger.info("Sending email summaries")
        analyzer.send_email_summaries()
        
        logger.info("Meeting analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Meeting analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze meeting audio and generate reports")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--openai-key", help="OpenAI API key (optional, can use environment variable)")
    parser.add_argument("--email-config", help="Path to email configuration file (optional, can use environment variables)")
    args = parser.parse_args()
    
    main(args.audio_file, args.openai_key, args.email_config)
