from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import meeting_analyzer
from dotenv import load_dotenv
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
import redis
import hashlib
import aiohttp
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

app = FastAPI(
    title="Meeting Analyzer API",
    description="API for analyzing meeting recordings and generating structured summaries",
    version="1.0.0"
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def verify_api_key(api_key: str = Header(..., alias=API_KEY_NAME)):
    """Verify the API key."""
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

class MeetingRequest(BaseModel):
    audio_file: str
    email_config_file: Optional[str] = None

class Participant(BaseModel):
    name: str
    email: str
    speaking_time: float
    tasks: List[str]
    action_items: List[str]
    future_references: List[Dict]

class MeetingResponse(BaseModel):
    meeting_summary: str
    participants: List[Participant]
    action_items: Dict[str, List[str]]
    tasks: Dict[str, List[str]]
    future_references: Dict[str, List[Dict]]
    key_highlights: Optional[List[str]]
    timestamp: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    timestamp: str

class N8nMeetingRequest(BaseModel):
    """n8n specific request model that handles various input formats"""
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_file_path: Optional[str] = None
    email_config: Optional[Dict] = None
    webhook_url: Optional[str] = None  # For n8n callback

class N8nMeetingResponse(BaseModel):
    """n8n friendly response format"""
    success: bool
    execution_id: str
    data: Optional[MeetingResponse] = None
    status: str
    callback_url: Optional[str] = None
    error: Optional[str] = None

def load_meeting_outputs() -> Dict:
    """Load and parse meeting analysis output files."""
    try:
        # Load meeting report
        with open("meeting_report.md", "r", encoding="utf-8") as f:
            meeting_summary = f.read()

        # Load segments if available
        segments_data = {}
        if os.path.exists("segments.json"):
            with open("segments.json", "r", encoding="utf-8") as f:
                segments_data = json.load(f)

        return {
            "meeting_summary": meeting_summary,
            "segments": segments_data
        }
    except Exception as e:
        logger.error(f"Error loading meeting outputs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load meeting analysis outputs")

@app.post("/analyze-meeting", response_model=MeetingResponse)
@limiter.limit("5/minute")
async def analyze_meeting(
    request: MeetingRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze a meeting recording and return structured results.
    
    Rate limit: 5 requests per minute per IP address
    Authentication: Requires API key in X-API-Key header
    """
    try:
        # Validate audio file exists
        if not os.path.exists(request.audio_file):
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "Audio file not found",
                    "error_code": "FILE_NOT_FOUND",
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Check cache
        cache_key = hashlib.md5(
            f"{request.audio_file}:{request.email_config_file}".encode()
        ).hexdigest()
        
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        # Run meeting analysis
        meeting_analyzer.main(
            audio_file=request.audio_file,
            openai_key=None,  # Use from environment
            email_config_file=request.email_config_file
        )

        # Load analysis outputs
        outputs = load_meeting_outputs()

        # Extract participants and their details from the analyzer's database
        analyzer = meeting_analyzer.MeetingAnalyzer(
            email_config=meeting_analyzer.EmailConfig(request.email_config_file)
        )
        
        participants = []
        action_items = {}
        tasks = {}
        future_references = {}

        for name, speaker in analyzer.speakers.items():
            participant = Participant(
                name=speaker.name,
                email=speaker.email or "",
                speaking_time=speaker.speaking_time,
                tasks=speaker.tasks,
                action_items=speaker.action_items,
                future_references=speaker.future_references
            )
            participants.append(participant)

            # Group items by participant
            action_items[name] = speaker.action_items
            tasks[name] = speaker.tasks
            future_references[name] = speaker.future_references

        # Extract key highlights (this could be enhanced with more sophisticated analysis)
        key_highlights = []
        for speaker in analyzer.speakers.values():
            if speaker.action_items:
                key_highlights.extend(speaker.action_items)

        # Create response
        response = MeetingResponse(
            meeting_summary=outputs["meeting_summary"],
            participants=participants,
            action_items=action_items,
            tasks=tasks,
            future_references=future_references,
            key_highlights=key_highlights[:5] if key_highlights else None,
            timestamp=datetime.now().isoformat()
        )

        # Cache the result for 1 hour
        redis_client.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(response.dict())
        )

        return response

    except Exception as e:
        logger.error(f"Error processing meeting analysis request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "detail": str(e),
                "error_code": "INTERNAL_SERVER_ERROR",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/ping")
@limiter.limit("60/minute")
async def ping():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/n8n/analyze-meeting", response_model=N8nMeetingResponse)
@limiter.limit("5/minute")
async def n8n_analyze_meeting(
    request: N8nMeetingRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    n8n-optimized endpoint for meeting analysis.
    Supports multiple input formats and asynchronous processing.
    """
    try:
        execution_id = hashlib.md5(f"{datetime.now().isoformat()}:{request.json()}".encode()).hexdigest()
        
        # Handle different input types
        audio_file = None
        if request.audio_url:
            # Download from URL
            audio_file = f"temp_{execution_id}.mp3"
            await download_audio(request.audio_url, audio_file)
        elif request.audio_base64:
            # Decode base64
            audio_file = f"temp_{execution_id}.mp3"
            decode_base64_audio(request.audio_base64, audio_file)
        elif request.audio_file_path:
            audio_file = request.audio_file_path
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "detail": "No audio input provided",
                    "error_code": "NO_AUDIO_INPUT",
                    "timestamp": datetime.now().isoformat()
                }
            )

        # Create temporary email config if provided
        email_config_file = None
        if request.email_config:
            email_config_file = f"temp_email_config_{execution_id}.json"
            with open(email_config_file, 'w') as f:
                json.dump(request.email_config, f)

        # Start analysis in background task
        background_tasks.add_task(
            process_meeting_analysis,
            audio_file=audio_file,
            email_config_file=email_config_file,
            execution_id=execution_id,
            callback_url=request.webhook_url
        )

        return N8nMeetingResponse(
            success=True,
            execution_id=execution_id,
            status="processing",
            callback_url=request.webhook_url
        )

    except Exception as e:
        logger.error(f"Error processing n8n meeting analysis request: {str(e)}")
        return N8nMeetingResponse(
            success=False,
            execution_id=execution_id if 'execution_id' in locals() else None,
            status="error",
            error=str(e)
        )

@app.get("/n8n/status/{execution_id}")
async def get_analysis_status(
    execution_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Check the status of a meeting analysis job"""
    try:
        # Check Redis for status
        status = redis_client.get(f"status:{execution_id}")
        if not status:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        status = json.loads(status)
        return N8nMeetingResponse(**status)
    
    except Exception as e:
        logger.error(f"Error checking analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_meeting_analysis(
    audio_file: str,
    email_config_file: Optional[str],
    execution_id: str,
    callback_url: Optional[str]
):
    """Background task to process meeting analysis"""
    try:
        # Update status to processing
        redis_client.set(
            f"status:{execution_id}",
            json.dumps({
                "success": True,
                "execution_id": execution_id,
                "status": "processing"
            })
        )

        # Run analysis
        analyzer = meeting_analyzer.MeetingAnalyzer()
        analyzer.process_audio_file(audio_file)
        
        # Generate report
        md_file, pdf_generated = analyzer.generate_meeting_report()
        
        # Create response
        response = create_meeting_response(analyzer, md_file)
        
        # Update status with results
        redis_client.set(
            f"status:{execution_id}",
            json.dumps({
                "success": True,
                "execution_id": execution_id,
                "status": "completed",
                "data": response.dict()
            }),
            ex=3600  # Expire in 1 hour
        )

        # Send callback if URL provided
        if callback_url:
            await send_callback(callback_url, response.dict())

    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        redis_client.set(
            f"status:{execution_id}",
            json.dumps({
                "success": False,
                "execution_id": execution_id,
                "status": "error",
                "error": str(e)
            }),
            ex=3600
        )

async def download_audio(url: str, target_path: str):
    """Download audio file from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(target_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                raise HTTPException(status_code=400, detail="Failed to download audio file")

def decode_base64_audio(base64_string: str, target_path: str):
    """Decode base64 audio data to file"""
    try:
        audio_data = base64.b64decode(base64_string)
        with open(target_path, 'wb') as f:
            f.write(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 audio: {str(e)}")

async def send_callback(url: str, data: Dict):
    """Send callback to n8n webhook"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Callback failed with status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send callback: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "webhook_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 