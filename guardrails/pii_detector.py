"""
PII Input Validator

This module provides STRICT INPUT VALIDATION for user queries.

PURPOSE:
- Detect PII/sensitive data in user input BEFORE sending to LLM
- REJECT unsafe input immediately - do not proceed to LLM
- Only allow safe, sanitized input to continue the workflow

WORKFLOW:
User input → validate() → PASS (safe) or FAIL (contains PII)
- PASS: Continue to LLM for SQL generation
- FAIL: Return error message, log incident, STOP workflow

No masking, no proceeding with unsafe data.
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIIDetectionResult:
    """Result of PII detection validation."""
    is_safe: bool
    detected_types: List[str]
    message: str
    original_input: str


class PIIInputValidator:
    """
    Input validator that detects PII in user queries.
    
    If PII is detected, the input is REJECTED and workflow stops.
    No data is masked or modified - only validated.
    """
    
    # PII detection patterns
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
    }
    
    # Sensitive keywords that might indicate PII requests
    SENSITIVE_KEYWORDS = [
        'password', 'secret', 'credential', 'token', 'api_key',
        'social security', 'ssn', 'credit card', 'card number',
        'personal information', 'private data', 'confidential',
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize PII input validator.
        
        Args:
            strict_mode: If True, reject on any PII detection. If False, only log warnings.
        """
        self.strict_mode = strict_mode
        logger.info(f"PIIInputValidator initialized (strict_mode={strict_mode})")
    
    def validate(self, user_input: str) -> PIIDetectionResult:
        """
        Validate user input for PII/sensitive data.
        
        Args:
            user_input: The raw user query to validate
            
        Returns:
            PIIDetectionResult with is_safe=True (pass) or False (reject)
        """
        if not user_input or not user_input.strip():
            return PIIDetectionResult(
                is_safe=True,
                detected_types=[],
                message="Empty input - safe",
                original_input=user_input
            )
        
        detected_types = []
        
        # Check patterns
        for pii_type, pattern in self.PATTERNS.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                detected_types.append(pii_type)
                logger.warning(f"PII detected: {pii_type} in user input")
        
        # Check sensitive keywords
        lower_input = user_input.lower()
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in lower_input:
                detected_types.append(f"keyword:{keyword}")
                logger.warning(f"Sensitive keyword detected: {keyword}")
        
        # Determine if input is safe
        if detected_types:
            if self.strict_mode:
                # REJECT - do not proceed
                message = (
                    f"INPUT REJECTED: PII or sensitive data detected ({', '.join(detected_types)}). "
                    "Please rephrase your query without including personal information."
                )
                logger.error(f"Input validation FAILED: {detected_types}")
                return PIIDetectionResult(
                    is_safe=False,
                    detected_types=detected_types,
                    message=message,
                    original_input=user_input
                )
            else:
                # WARNING mode - allow but log
                message = f"WARNING: Potential PII detected ({', '.join(detected_types)})"
                logger.warning(message)
                return PIIDetectionResult(
                    is_safe=True,
                    detected_types=detected_types,
                    message=message,
                    original_input=user_input
                )
        
        # No PII detected - PASS
        logger.info("Input validation PASSED: No PII detected")
        return PIIDetectionResult(
            is_safe=True,
            detected_types=[],
            message="Input is safe - no PII detected",
            original_input=user_input
        )
    
    def validate_strict(self, user_input: str) -> Tuple[bool, str]:
        """
        Strict validation that returns simple pass/fail.
        
        Args:
            user_input: The user query to validate
            
        Returns:
            Tuple of (is_safe, message)
            - (True, "safe") if no PII
            - (False, "rejection reason") if PII detected
        """
        result = self.validate(user_input)
        return result.is_safe, result.message


# Convenience function for quick validation
def validate_input_for_pii(user_input: str, strict: bool = True) -> Tuple[bool, str]:
    """
    Quick validation function.
    
    Args:
        user_input: User query to validate
        strict: Whether to reject on PII detection
        
    Returns:
        (is_safe, message) tuple
    """
    validator = PIIInputValidator(strict_mode=strict)
    return validator.validate_strict(user_input)
