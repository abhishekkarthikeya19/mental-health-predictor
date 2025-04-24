# run.py
import sys
import os

# Apply patch for anyio to avoid uvloop on Windows
if sys.platform == 'win32':
    try:
        # Try file-based patching first
        from patch_anyio import patch_anyio
        patch_anyio()
    except ImportError:
        print("Warning: patch_anyio module not found. Trying runtime patching.")
    
    try:
        # Try runtime patching as a fallback
        from runtime_patch import patch_anyio_at_runtime
        patch_anyio_at_runtime()
    except ImportError:
        print("Warning: runtime_patch module not found. uvloop errors may occur on Windows.")

# Run the uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)