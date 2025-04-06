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
import subprocess
import argparse
import whisper

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

class EmailConfig:
    def __init__(self, config_file: str = "email_config.json"):
        """Initialize email configuration from JSON file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email') or os.getenv('EMAIL_ADDRESS')
        self.sender_password = config.get('sender_password') or os.getenv('EMAIL_PASSWORD')
        
        if not self.sender_email or not self.sender_password:
            raise ValueError("Email credentials not found in config or environment variables")

class MeetingAnalyzer:
    def __init__(self, openai_key=None):
        """Initialize the meeting analyzer."""
        self.model = whisper.load_model("base")
        self.speakers = {}
        self.segments_data = []
        self.full_text = ""
        self.attendee_db = AttendeeDatabase()  # Use the AttendeeDatabase class
        
        # Set OpenAI key if provided
        if openai_key:
            openai.api_key = openai_key

    def process_audio_file(self, audio_file):
        """Process the audio file to transcribe and identify speakers."""
        logger.info("Starting transcription")
        self.full_text, self.segments_data = self.transcribe_audio(audio_file)
        
        # Identify speakers
        logger.info("Identifying speakers")
        self.speakers = self.identify_speakers(self.segments_data)
        
        # Extract action items for each speaker
        logger.info("Extracting action items")
        for speaker in self.speakers.values():
            self.extract_action_items(speaker)

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

    def identify_speakers(self, segments_data):
        """Identify speakers from transcript segments using batch processing."""
        logger.info("Identifying speakers from transcript")
        
        # Combine segments into batches of 10 for fewer API calls
        batch_size = 10
        batched_segments = [segments_data[i:i + batch_size] for i in range(0, len(segments_data), batch_size)]
        
        speakers = {}
        speaker_segments = {}  # Track which segments belong to which speaker
        
        logger.info("Analyzing transcript for speaker identification...")
        
        # First, identify all unique speakers
        full_text = "\n".join([seg["text"] for seg in segments_data])
        messages = [
            {"role": "system", "content": "You are a helpful assistant that identifies speakers in a conversation. List each speaker on a new line with their name and confidence level (high/medium/low)."},
            {"role": "user", "content": f"Analyze this conversation and list each unique speaker. Format your response as:\nSpeaker Name (confidence level)\n\nTranscript:\n{full_text}"}
        ]
        
        response = call_openai_with_retry(openai, model="gpt-3.5-turbo", messages=messages)
        speaker_list = response.choices[0].message.content
        
        # Process speaker list and get confirmations
        potential_speakers = []
        for line in speaker_list.split('\n'):
            if not line.strip():
                continue
            # Look for patterns like "Name (confidence)" or "- Name (confidence)"
            match = re.search(r'(?:-\s*)?([^(]+?)\s*\((high|medium|low)\s*(?:confidence)?\)', line.lower())
            if match:
                name = match.group(1).strip().title()  # Capitalize each word
                confidence = match.group(2).lower()
                potential_speakers.append((name, confidence))
        
        print(f"\nIdentified {len(potential_speakers)} speakers:")
        
        # Process batches of segments for speaker assignment
        with tqdm(total=len(batched_segments), desc="Processing segments") as pbar:
            for batch in batched_segments:
                batch_text = "\n".join([f"Segment {i}: {seg['text']}" for i, seg in enumerate(batch)])
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that identifies which speaker said each segment."},
                    {"role": "user", "content": f"For each segment, identify which of these speakers said it: {', '.join([s[0] for s in potential_speakers])}.\nRespond with segment number and speaker name only, one per line.\n\n{batch_text}"}
                ]
                
                response = call_openai_with_retry(openai, model="gpt-3.5-turbo", messages=messages)
                assignments = response.choices[0].message.content
                
                # Process speaker assignments
                for line in assignments.split('\n'):
                    if not line.strip():
                        continue
                    match = re.search(r'Segment\s*(\d+):\s*([^,]+)', line)
                    if match:
                        seg_idx = int(match.group(1))
                        speaker_name = match.group(2).strip()
                        if seg_idx < len(batch):
                            if speaker_name not in speaker_segments:
                                speaker_segments[speaker_name] = []
                            speaker_segments[speaker_name].append(batch[seg_idx])
                
                pbar.update(1)
        
        # Create Speaker objects
        for name, confidence in potential_speakers:
            print(f"- {name} ({confidence} confidence)")
            valid = input(f"  Is '{name}' accurate for this speaker? (Y/N): ").strip().upper() == 'Y'
            
            if valid:
                # Get or create attendee info
                attendee_info = self.get_attendee(name)
                if not attendee_info:
                    email = input(f"  Please enter email address for {name}: ").strip()
                    if email:  # Only add if email is provided
                        self.add_or_update_attendee(name, email)
                        attendee_info = self.get_attendee(name)
                
                # Calculate speaking time
                speaking_time = sum(float(seg['end']) - float(seg['start']) 
                                  for seg in speaker_segments.get(name, []))
                
                # Create Speaker object with their segments
                speakers[name] = Speaker(
                    name=name,
                    speaking_time=speaking_time,
                    segments=speaker_segments.get(name, []),
                    tasks=[],
                    action_items=[],
                    future_references=[],
                    email=attendee_info['email'] if attendee_info else None
                )
        
        return speakers

    def extract_action_items(self, speaker):
        """Extract action items for a speaker using a single API call."""
        if not speaker.segments:
            return
        
        # Combine all segments for the speaker
        full_text = "\n".join([seg["text"] for seg in speaker.segments])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that identifies action items and future references from meeting transcripts."},
            {"role": "user", "content": f"Analyze this speaker's ({speaker.name}) contributions and identify:\n1. Action items (tasks they need to do)\n2. Future references (deadlines, follow-ups, or important dates)\n3. Key discussion points\n\nText:\n{full_text}"}
        ]
        
        try:
            response = call_openai_with_retry(openai, model="gpt-3.5-turbo", messages=messages)
            content = response.choices[0].message.content
            
            # Parse the response
            action_items = []
            future_refs = []
            
            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if 'action item' in line.lower():
                    current_section = 'action'
                elif 'future reference' in line.lower():
                    current_section = 'future'
                elif 'key discussion' in line.lower():
                    current_section = 'discussion'
                elif line.startswith('-') or line.startswith('•'):
                    item = line[1:].strip()
                    if current_section == 'action':
                        action_items.append({
                            'description': item,
                            'context': 'From meeting discussion'
                        })
                    elif current_section == 'future':
                        # Try to extract deadline if present
                        deadline_match = re.search(r'by\s+([^,\.]+)', item)
                        deadline = deadline_match.group(1) if deadline_match else None
                        
                        future_refs.append({
                            'description': item,
                            'deadline': deadline,
                            'context': 'From meeting discussion'
                        })
            
            speaker.action_items = action_items
            speaker.future_references = future_refs
            
        except Exception as e:
            logger.error(f"Failed to extract action items for {speaker.name}: {str(e)}")
            raise

    def generate_meeting_report(self) -> Tuple[str, bool]:
        """Generate a meeting report in Markdown format and convert to PDF."""
        logger.info("Generating meeting report")
        
        # Create markdown content
        md_content = []
        
        # Add title and date
        md_content.append("# Meeting Summary Report\n")
        md_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add brief overview
        md_content.append("## Brief Overview\n")
        md_content.append("This meeting focused on the development and deployment of an AI-based system. Key topics discussed included:\n")
        md_content.append("- Docker containerization of AI/ML applications\n")
        md_content.append("- Project scope definition and data set characteristics\n")
        md_content.append("- Research plans for anomaly detection and fraud detection\n")
        md_content.append("- Task delegation and implementation planning\n\n")
        
        # Add participants section
        md_content.append("## Participants\n")
        for name, speaker in self.speakers.items():
            speaking_time = round(speaker.speaking_time / 60, 1)  # Convert to minutes
            md_content.append(f"- **{name}** (Speaking time: {speaking_time} minutes)\n")
        md_content.append("\n")
        
        # Add action items section
        md_content.append("## Action Items and Tasks by Participant\n\n")
        for name, speaker in self.speakers.items():
            md_content.append(f"### {name}\n\n")
            
            if speaker.action_items:
                md_content.append("#### Action Items:\n")
                for item in speaker.action_items:
                    if isinstance(item, dict):
                        md_content.append(f"- {item.get('description', 'No description provided')}\n")
                    else:
                        md_content.append(f"- {item}\n")
                md_content.append("\n")
            
            if speaker.future_references:
                md_content.append("#### Future References:\n")
                for ref in speaker.future_references:
                    deadline = ref.get('deadline', 'None')
                    context = ref.get('context', 'No context provided')
                    description = ref.get('description', ref.get('task', 'No description provided'))
                    md_content.append(f"- {description} (Deadline: {deadline})\n  Context: {context}\n")
                md_content.append("\n")
        
        # Write to markdown file
        md_file = "meeting_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(''.join(md_content))
        
        # Convert to HTML
        html_content = markdown.markdown(''.join(md_content))
        html_file = "meeting_report.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Meeting Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        ul {{ list-style-type: disc; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
''')
        
        # Convert to PDF using full path to wkhtmltopdf
        pdf_generated = False
        try:
            # Use absolute paths
            html_path = os.path.abspath(html_file)
            pdf_path = os.path.abspath('meeting_report.pdf')
            wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            
            # Add more options for better PDF generation
            cmd = [
                wkhtmltopdf_path,
                '--encoding', 'utf-8',
                '--page-size', 'A4',
                '--margin-top', '20',
                '--margin-bottom', '20',
                '--margin-left', '20',
                '--margin-right', '20',
                '--load-error-handling', 'ignore',
                html_path,
                pdf_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.returncode == 0:
                pdf_generated = True
                logger.info("PDF generated successfully")
            else:
                logger.error(f"PDF generation failed with output: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
        
        logger.info(f"Meeting report saved to {md_file}" + (" and meeting_report.pdf" if pdf_generated else ""))
        return md_file, pdf_generated

    def send_email_summaries(self, email_config):
        """Send individual summaries to each participant."""
        logger.info("Sending email summaries to participants")
        
        try:
            # Setup SMTP connection
            smtp = smtplib.SMTP(email_config.smtp_server, email_config.smtp_port)
            smtp.starttls()
            smtp.login(email_config.sender_email, email_config.sender_password)
            
            # Read the full report files
            with open("meeting_report.md", "r", encoding="utf-8") as f:
                md_content = f.read()
            
            # Try to read PDF if it exists
            pdf_content = None
            try:
                with open("meeting_report.pdf", "rb") as f:
                    pdf_content = f.read()
                    logger.info("PDF file loaded successfully")
            except FileNotFoundError:
                logger.warning("PDF report not found, will send only markdown version")
            
            for speaker in self.speakers.values():
                if not speaker.email:
                    logger.warning(f"No email address found for {speaker.name}, skipping summary")
                    continue
                
                # Create personalized summary
                subject = "Meeting Summary and Action Items"
                
                # Create task list
                action_items_list = []
                for item in speaker.action_items:
                    if isinstance(item, dict):
                        action_items_list.append(f"- {item.get('description', 'No description provided')}")
                    else:
                        action_items_list.append(f"- {item}")
                
                future_list = []
                for ref in speaker.future_references:
                    desc = ref.get('description', ref.get('task', 'No description provided'))
                    deadline = ref.get('deadline', 'Not specified')
                    future_list.append(f"- {desc} (Deadline: {deadline})")
                
                body = f"""Hello {speaker.name},

Here's your summary from the recent meeting:

Your Action Items:
{chr(10).join(action_items_list) if action_items_list else "No action items"}

Your Future Commitments:
{chr(10).join(future_list) if future_list else "No future commitments"}

The complete meeting report is attached in {"both Markdown and PDF formats" if pdf_content else "Markdown format"} for your reference.

Best regards,
Meeting Analyzer"""
                
                # Create message
                msg = MIMEMultipart()
                msg["From"] = email_config.sender_email
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

    def get_attendee(self, name):
        """Get attendee information from the database."""
        return self.attendee_db.get_attendee(name)

    def add_or_update_attendee(self, name, email, role):
        """Add or update an attendee in the database."""
        self.attendee_db.add_or_update_attendee(name, email, role)

def main(audio_file, openai_key=None, email_config_file=None, send_emails=False):
    """
    Main function to process an audio file and generate a meeting report.
    
    Args:
        audio_file (str): Path to the audio file
        openai_key (str, optional): OpenAI API key
        email_config_file (str, optional): Path to email configuration file
        send_emails (bool, optional): Whether to send email summaries
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Initialize the meeting analyzer
        analyzer = MeetingAnalyzer(openai_key)
        
        # Process the audio file
        analyzer.process_audio_file(audio_file)
        
        # Generate the meeting report
        analyzer.generate_meeting_report()
        
        # Send email summaries if requested
        if send_emails and email_config_file:
            try:
                email_config = EmailConfig(email_config_file)
                analyzer.send_email_summaries(email_config)
            except Exception as e:
                logging.warning(f"Failed to send email summaries: {str(e)}")
                logging.warning("Continuing without sending emails...")
        
        logging.info("Meeting analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Meeting analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze meeting audio and generate report")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--email-config", dest="email_config_file", help="Path to email configuration file")
    parser.add_argument("--send-emails", action="store_true", help="Send email summaries to participants")
    
    args = parser.parse_args()
    main(args.audio_file, args.openai_key, args.email_config_file, args.send_emails)
