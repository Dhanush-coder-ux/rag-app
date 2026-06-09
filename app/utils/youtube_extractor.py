"""
YouTube transcript extraction utility for RAG pipeline
"""
import re
from typing import Optional
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


def extract_youtube_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - VIDEO_ID (raw ID)
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


async def get_video_metadata(video_id: str) -> dict:
    """
    Get YouTube video metadata (title, description, duration, etc.)
    """
    try:
        # Using the noembed.com API (no auth required)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data.get("title", "Unknown Title"),
                    "author": data.get("author_name", "Unknown Author"),
                    "description": data.get("description", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    
    # Fallback
    return {
        "title": f"YouTube Video {video_id}",
        "author": "Unknown Author",
        "description": "",
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }


async def extract_transcript(youtube_url: str) -> tuple[str, dict]:
    """
    Extract transcript from YouTube video.
    
    Returns:
        Tuple of (transcript_text, metadata)
    
    Raises:
        ValueError: If URL is invalid or transcript unavailable
    """
    video_id = extract_youtube_id(youtube_url)
    
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")
    
    try:
        api = YouTubeTranscriptApi()
        
        # Get the transcript - try English first, then any available language
        try:
            transcript = api.fetch(video_id, languages=['en'])
        except Exception:
            # If English not available, get any available transcript
            transcript_list = api.list(video_id)
            
            # Try to get first available transcript
            # Access via __iter__ or get all and pick first
            try:
                # Get first manually created or generated transcript
                available = list(transcript_list)
                transcript = None
                errors = []
                
                # Try each available transcript
                for trans in available:
                    try:
                        transcript = trans.fetch()
                        break  # Success, exit loop
                    except Exception as e:
                        errors.append(f"{trans.language}: {str(e)}")
                        continue
                
                if not transcript:
                    error_msg = "; ".join(errors) if errors else "Unknown error"
                    raise ValueError(f"No usable transcripts for video {video_id}. Errors: {error_msg}")
            except (AttributeError, StopIteration, IndexError):
                raise ValueError(f"No transcripts found for video {video_id}")
        
        if not transcript:
            raise ValueError(f"No transcripts found for video {video_id}")
        
        # Convert transcript list to formatted text
        transcript_text = "\n".join(
            [f"[{item.start:.1f}s] {item.text}" for item in transcript]
        )
        
        # Get metadata
        metadata = await get_video_metadata(video_id)
        
        return transcript_text, metadata
    
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for this video: {youtube_url}")
    except NoTranscriptFound:
        raise ValueError(f"No transcripts found for this video: {youtube_url}")
    except Exception as e:
        raise ValueError(f"Error extracting transcript: {str(e)}")


def format_transcript_for_ingestion(
    transcript_text: str,
    metadata: dict
) -> str:
    """
    Format transcript with metadata for ingestion into RAG pipeline.
    """
    formatted = f"""
# {metadata['title']}

**Author:** {metadata['author']}  
**Source:** {metadata['url']}

---

## Transcript

{transcript_text}
"""
    return formatted.strip()
