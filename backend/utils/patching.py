# utils/patching.py
"""
Utility module for applying patches to dependencies.
This keeps the main application code clean.
"""
import sys
import logging

logger = logging.getLogger(__name__)

def apply_anyio_patch():
    """Apply patch for anyio to avoid uvloop on Windows."""
    if sys.platform == 'win32':
        try:
            # Try file-based patching first
            from patch_anyio import patch_anyio
            patch_anyio()
            logger.info("Applied anyio patch using patch_anyio module")
        except ImportError:
            logger.warning("patch_anyio module not found. Trying runtime patching.")
            
            try:
                # Try runtime patching as a fallback
                from runtime_patch import patch_anyio_at_runtime
                patch_anyio_at_runtime()
                logger.info("Applied anyio patch using runtime_patch module")
            except ImportError:
                logger.warning("runtime_patch module not found. uvloop errors may occur on Windows.")

def apply_logging_patch():
    """Apply patch for logging circular import issue."""
    try:
        from logging_patch import apply_logging_patch
        apply_logging_patch()
        logger.info("Applied logging patch successfully")
    except ImportError:
        logger.warning("logging_patch module not found. Circular import errors may occur.")