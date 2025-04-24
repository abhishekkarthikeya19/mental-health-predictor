# logging_patch.py
# This file patches the uvicorn logging module to avoid circular imports

def apply_logging_patch():
    """
    Apply a patch to fix the circular import issue in uvicorn's logging module.
    This ensures that the standard library logging module is fully initialized
    before uvicorn's logging module tries to use it.
    """
    import sys
    import importlib
    
    # First, ensure the standard library logging is fully imported
    import logging
    
    # If uvicorn.logging is already in sys.modules, remove it to force a clean import
    if 'uvicorn.logging' in sys.modules:
        del sys.modules['uvicorn.logging']
    
    # Now, when uvicorn.logging is imported again, it should work correctly
    print("Logging patch applied successfully")