import anyio
import os
import sys

print(f"Anyio path: {os.path.dirname(anyio.__file__)}")
print(f"Anyio backend path: {os.path.join(os.path.dirname(anyio.__file__), '_backends', '_asyncio.py')}")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")