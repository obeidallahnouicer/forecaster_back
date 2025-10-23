"""
Optimized LLM client for Groq API (Llama 3.3 70B).

Provides:
- Real Groq API integration with proper error handling
- Rate limiting and cooldown management
- Token tracking and cost estimation
- Efficient prompt formatting
"""
import logging
import os
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("llm.client")

# Groq rate limits (conservative estimates)
GROQ_RPM = 30  # requests per minute
GROQ_TPM = 6000  # tokens per minute (Llama 3.3 70B on Groq)
COOLDOWN_FILE = Path("cache/groq_cooldown.json")


class RateLimiter:
    """Simple rate limiter with cooldown tracking."""
    
    def __init__(self, rpm: int = GROQ_RPM):
        self.rpm = rpm
        self.calls = []
        self.cooldown_until = 0
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        now = time.time()
        
        # Check cooldown
        if now < self.cooldown_until:
            wait_time = self.cooldown_until - now
            logger.warning(f"Rate limit cooldown: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            now = time.time()
        
        # Clean old calls (older than 60s)
        self.calls = [t for t in self.calls if now - t < 60]
        
        # If at limit, wait
        if len(self.calls) >= self.rpm:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self.calls.append(time.time())
    
    def set_cooldown(self, seconds: int):
        """Set a cooldown period after rate limit error."""
        self.cooldown_until = time.time() + seconds
        # Persist cooldown
        COOLDOWN_FILE.parent.mkdir(exist_ok=True)
        with open(COOLDOWN_FILE, 'w') as f:
            json.dump({'cooldown_until': self.cooldown_until}, f)
    
    def load_cooldown(self):
        """Load cooldown from disk if exists."""
        if COOLDOWN_FILE.exists():
            try:
                with open(COOLDOWN_FILE, 'r') as f:
                    data = json.load(f)
                    self.cooldown_until = data.get('cooldown_until', 0)
            except Exception:
                pass


# Global rate limiter
_rate_limiter = RateLimiter()
_rate_limiter.load_cooldown()


def get_llm_response(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 500,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get LLM response from Groq API (Llama 3.3 70B).
    
    Args:
        prompt: User/human prompt
        model: Model name (default: llama-3.3-70b-versatile)
        max_tokens: Max completion tokens
        temperature: Sampling temperature (0.0-1.0)
        system_prompt: Optional system prompt
    
    Returns:
        {
            'content': str,
            'prompt_tokens': int,
            'completion_tokens': int,
            'total_tokens': int,
            'model': str
        }
    """
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        logger.warning("GROQ_API_KEY not set; returning fallback response")
        return {
            'content': "LLM not configured. Set GROQ_API_KEY environment variable.",
            'prompt_tokens': len(prompt) // 4,
            'completion_tokens': 0,
            'total_tokens': len(prompt) // 4,
            'model': 'fallback'
        }
    
    try:
        # Import Groq SDK
        try:
            from groq import Groq
        except ImportError:
            logger.error("Groq SDK not installed. Run: pip install groq")
            return {
                'content': "Groq SDK not installed.",
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'model': 'error'
            }
        
        # Wait if rate limited
        _rate_limiter.wait_if_needed()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call Groq API
        client = Groq(api_key=api_key)
        start = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        elapsed = time.time() - start
        content = response.choices[0].message.content
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else len(prompt) // 4
        completion_tokens = usage.completion_tokens if usage else len(content) // 4
        total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
        
        logger.info(f"Groq API: {model}, {total_tokens} tokens, {elapsed:.2f}s")
        
        return {
            'content': content,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'model': model
        }
    
    except Exception as e:
        error_str = str(e)
        
        # Handle rate limit errors
        if 'rate_limit' in error_str.lower() or '429' in error_str:
            logger.error(f"Rate limit hit: {e}")
            _rate_limiter.set_cooldown(60)  # Wait 1 minute
            return {
                'content': "Rate limit exceeded. Please try again in a moment.",
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'model': 'rate_limited'
            }
        
        logger.exception(f"Groq API call failed: {e}")
        return {
            'content': f"LLM error: {error_str}",
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'model': 'error'
        }
