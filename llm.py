"""
LLM Integration via AI Pipe (OpenAI-compatible API)
"""
import os
import base64
import logging
import requests
from openai import OpenAI

from config import AIPIPE_TOKEN, OPENAI_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

# Lazy initialization - don't create client at import time
_client = None

def get_client():
    """Get or create the OpenAI client (lazy initialization)"""
    global _client
    if _client is None:
        api_key = AIPIPE_TOKEN or os.getenv("AIPIPE_TOKEN")
        if not api_key:
            raise ValueError("AIPIPE_TOKEN environment variable is not set!")
        _client = OpenAI(
            api_key=api_key,
            base_url=OPENAI_BASE_URL
        )
    return _client


def ask_llm(prompt: str, model: str = None, system_prompt: str = None) -> str:
    """
    Send a prompt to the LLM and get a response
    """
    model = model or LLM_MODEL
    client = get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        logger.info(f"Sending prompt to {model} (length: {len(prompt)} chars)")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096
        )
        
        result = response.choices[0].message.content
        logger.info(f"Received response (length: {len(result)} chars)")
        return result
        
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise


def ask_llm_with_image(prompt: str, image_url: str = None, image_base64: str = None, model: str = "gpt-4o") -> str:
    """
    Send a prompt with an image to the LLM (vision capability)
    """
    client = get_client()
    content = [{"type": "text", "text": prompt}]
    
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    elif image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })
    
    try:
        logger.info(f"Sending vision prompt to {model}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096
        )
        
        result = response.choices[0].message.content
        logger.info(f"Received vision response (length: {len(result)} chars)")
        return result
        
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        raise


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using gpt-4o-audio-preview via chat completions.
    AI Pipe supports audio in chat messages.
    """
    api_key = AIPIPE_TOKEN or os.getenv("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN not set")
    
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Read and encode audio as base64
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Determine audio format from extension
        ext = os.path.splitext(audio_path)[1].lower()
        audio_format = {
            '.opus': 'opus',
            '.mp3': 'mp3',
            '.wav': 'wav',
            '.ogg': 'ogg',
            '.m4a': 'm4a',
        }.get(ext, 'opus')
        
        logger.info(f"Audio format: {audio_format}, base64 length: {len(audio_data)}")
        
        # Use chat completions with audio input
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try using gpt-4o-audio-preview with audio input
        payload = {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": audio_format
                            }
                        },
                        {
                            "type": "text",
                            "text": "Transcribe this audio exactly. Return only the spoken text, nothing else."
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        logger.info(f"Audio chat response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            logger.info(f"Transcription complete: {text}")
            return text.strip()
        else:
            logger.error(f"Audio chat failed: {response.text}")
            raise Exception(f"Audio API error: {response.status_code} - {response.text}")
        
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        raise
