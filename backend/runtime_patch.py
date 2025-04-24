# runtime_patch.py
import sys
import types

def patch_anyio_at_runtime():
    """
    Patch anyio at runtime to avoid using uvloop on Windows systems.
    This function monkey patches the anyio._backends._asyncio module to avoid importing uvloop on Windows.
    """
    if sys.platform == 'win32':
        try:
            import anyio._backends._asyncio
            
            # Define a new function to replace the original one
            def patched_init(self, *, debug=None, use_uvloop=False, loop_factory=None):
                if use_uvloop and loop_factory is None and sys.platform != 'win32':
                    import uvloop
                    loop_factory = uvloop.new_event_loop
                
                self._runner = anyio._backends._asyncio.Runner(debug=debug, loop_factory=loop_factory)
            
            # Get the original class
            original_class = anyio._backends._asyncio.AsyncIOBackend
            
            # Replace the __init__ method
            original_class.__init__ = patched_init
            
            print("Successfully patched anyio at runtime")
        except ImportError:
            print("anyio module not found")
        except Exception as e:
            print(f"Error patching anyio: {e}")

if __name__ == "__main__":
    patch_anyio_at_runtime()