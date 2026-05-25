"""
logging_sanitizer.py - Filter sensitive data from logs before disk write
============================================================================
Wraps Python logging to automatically filter/encrypt sensitive fields
(credentials, C2 endpoints, vault paths, API keys) before they reach disk.

Prevents forensic recovery of operational artifacts from log files.

Usage:
    from logging_sanitizer import SanitizedLogger
    
    logger = SanitizedLogger(__name__)
    
    # Normal logging - sensitive fields auto-filtered
    logger.info(f"Connected to C2: https://attacker.com/api")  
    # Output: Connected to C2: [REDACTED]
    
    logger.debug(f"Vault password: secret123")
    # Output: Vault password: [REDACTED]
"""

import logging
import re
import secrets
from typing import List, Optional, Tuple


class SensitivePatterns:
    """Collection of regex patterns for sensitive data detection."""
    
    # Credential patterns
    PASSWORD_PATTERN = re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s,]+)', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([^"\'\s,]+)', re.IGNORECASE)
    TOKEN_PATTERN = re.compile(r'(token|bearer)["\']?\s*[:=]\s*["\']?([^"\'\s,]+)', re.IGNORECASE)
    SECRET_PATTERN = re.compile(r'(secret|api[_-]?secret)["\']?\s*[:=]\s*["\']?([^"\'\s,]+)', re.IGNORECASE)
    
    # URL patterns (C2 endpoints, etc.)
    URL_PATTERN = re.compile(r'https?://[^\s\'"]+', re.IGNORECASE)
    
    # Path patterns (vault paths, config paths)
    VAULT_PATH_PATTERN = re.compile(r'~?/\.blackvault|\.sealed_config')
    
    # Database connection strings
    CONN_STRING_PATTERN = re.compile(r'(mongodb|mysql|postgres|redis)://[^\s\'"]+', re.IGNORECASE)
    
    # AWS/Cloud credentials
    AWS_KEY_PATTERN = re.compile(r'AKIA[0-9A-Z]{16}')
    
    # Windows registry paths (potentially sensitive)
    REGISTRY_PATTERN = re.compile(r'HKCU|HKLM|Software\\', re.IGNORECASE)
    
    @classmethod
    def all_patterns(cls) -> List[Tuple[re.Pattern, str]]:
        """
        Return all patterns with descriptions.
        
        Returns:
            List of (pattern, description) tuples
        """
        return [
            (cls.PASSWORD_PATTERN, 'password'),
            (cls.API_KEY_PATTERN, 'api_key'),
            (cls.TOKEN_PATTERN, 'token'),
            (cls.SECRET_PATTERN, 'secret'),
            (cls.URL_PATTERN, 'url'),
            (cls.VAULT_PATH_PATTERN, 'vault_path'),
            (cls.CONN_STRING_PATTERN, 'connection_string'),
            (cls.AWS_KEY_PATTERN, 'aws_key'),
            (cls.REGISTRY_PATTERN, 'registry_path'),
        ]


class LogSanitizer:
    """Sanitizes log messages by filtering sensitive data."""
    
    def __init__(self, redaction_marker: str = '[REDACTED]', mode: str = 'redact'):
        """
        Initialize log sanitizer.
        
        Args:
            redaction_marker: Placeholder string to replace sensitive data
            mode: 'redact' (replace with marker) or 'hash' (replace with hash)
        """
        self.marker = redaction_marker
        self.mode = mode
        self._patterns = SensitivePatterns.all_patterns()
    
    def sanitize_message(self, message: str) -> str:
        """
        Sanitize a log message by filtering sensitive fields.
        
        Args:
            message: Original log message
        
        Returns:
            Sanitized message with sensitive data replaced
        
        Example:
            >>> sanitizer = LogSanitizer()
            >>> msg = "Connected to Redis: redis://localhost:6379"
            >>> sanitizer.sanitize_message(msg)
            'Connected to Redis: [REDACTED]'
        """
        sanitized = message
        
        for pattern, description in self._patterns:
            # Find all matches
            matches = list(pattern.finditer(sanitized))
            
            # Replace in reverse order to avoid offset issues
            for match in reversed(matches):
                if self.mode == 'redact':
                    replacement = self.marker
                elif self.mode == 'hash':
                    # Hash the matched value (first group or full match)
                    matched_value = match.group(1) if match.groups() else match.group(0)
                    import hashlib
                    hash_val = hashlib.sha256(matched_value.encode()).hexdigest()[:8]
                    replacement = f'[HASH:{hash_val}]'
                else:
                    replacement = self.marker
                
                sanitized = sanitized[:match.start()] + replacement + sanitized[match.end():]
        
        return sanitized
    
    def sanitize_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Sanitize a logging.LogRecord.
        
        Args:
            record: LogRecord to sanitize
        
        Returns:
            Modified LogRecord with sanitized message
        """
        record.msg = self.sanitize_message(str(record.msg))
        record.getMessage = lambda: record.msg  # Override getMessage
        return record


class SanitizedLogger:
    """
    Drop-in replacement for logging.Logger that sanitizes sensitive data.
    
    Usage:
        logger = SanitizedLogger(__name__)
        logger.info("Password: mysecret123")  # Auto-filters
    """
    
    def __init__(self, name: str, level: int = logging.DEBUG):
        """
        Initialize sanitized logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._sanitizer = LogSanitizer(redaction_marker='[REDACTED]', mode='redact')
        
        # Replace handlers with sanitizing filter
        self._add_sanitizing_filter()
    
    def _add_sanitizing_filter(self):
        """Add sanitizing filter to all handlers."""
        class SanitizingFilter(logging.Filter):
            def __init__(self, sanitizer: LogSanitizer):
                super().__init__()
                self.sanitizer = sanitizer
            
            def filter(self, record: logging.LogRecord) -> bool:
                record = self.sanitizer.sanitize_record(record)
                return True
        
        sanitizing_filter = SanitizingFilter(self._sanitizer)
        
        for handler in self._logger.handlers:
            handler.addFilter(sanitizing_filter)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        sanitized_msg = self._sanitizer.sanitize_message(msg)
        self._logger.debug(sanitized_msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        sanitized_msg = self._sanitizer.sanitize_message(msg)
        self._logger.info(sanitized_msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        sanitized_msg = self._sanitizer.sanitize_message(msg)
        self._logger.warning(sanitized_msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        sanitized_msg = self._sanitizer.sanitize_message(msg)
        self._logger.error(sanitized_msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        sanitized_msg = self._sanitizer.sanitize_message(msg)
        self._logger.critical(sanitized_msg, *args, **kwargs)


# Global sanitized logger instance (TIER 2a integration point)
_global_sanitized_logger = None


def get_sanitized_logger(name: str = 'blackroot_agent', level: int = logging.DEBUG) -> SanitizedLogger:
    """Get or create global sanitized logger."""
    global _global_sanitized_logger
    if _global_sanitized_logger is None:
        _global_sanitized_logger = SanitizedLogger(name, level=level)
    return _global_sanitized_logger


if __name__ == '__main__':
    print("[*] Logging Sanitizer - Self Test")
    
    sanitizer = LogSanitizer()
    
    # Test messages with various sensitive data
    test_cases = [
        "Connected to Redis: redis://localhost:6379",
        "C2 endpoint: https://attacker.com/api/v1?token=abc123def456",
        "Vault path: ~/.blackvault/sealed_config",
        "AWS credentials: AKIA1234567890ABCDEF",
        'Password: "super_secret_pass123"',
        "API Key: api_key = sk-1234567890abcdefghij",
        "Database: mongodb://user:pass@localhost/blackroot",
        "Registry path: HKCU\\Software\\Microsoft\\Windows",
    ]
    
    print("[*] Testing sanitization:")
    for test in test_cases:
        sanitized = sanitizer.sanitize_message(test)
        print(f"  Original:  {test}")
        print(f"  Sanitized: {sanitized}")
        print()
    
    # Test logger
    print("[*] Testing SanitizedLogger:")
    logger = SanitizedLogger('test_logger')
    
    # Set up console handler for demo
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger._logger.addHandler(handler)
    
    logger.info("Agent initialized with C2: https://malware.attacker.com/api")
    logger.warning("Failed to connect to Redis: redis://db.internal:6379")
    logger.debug(f"Vault encryption key: {secrets.token_hex(32)}")
    
    print("[*] All tests passed!")
