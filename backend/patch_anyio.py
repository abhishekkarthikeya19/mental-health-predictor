# patch_anyio.py
import sys
import os
import glob

def patch_anyio():
    """
    Patch anyio to avoid using uvloop on Windows systems.
    """
    if sys.platform == 'win32':
        # Find all possible anyio installations
        try:
            import anyio
            possible_paths = []
            
            # Add the main anyio path
            anyio_path = os.path.dirname(anyio.__file__)
            possible_paths.append(os.path.join(anyio_path, '_backends', '_asyncio.py'))
            
            # Look for other possible installations
            site_packages_dirs = []
            for path in sys.path:
                if 'site-packages' in path or 'dist-packages' in path:
                    site_packages_dirs.append(path)
            
            # Add explicit path for backend/venv
            backend_venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'lib', 'python3.12', 'site-packages')
            if os.path.exists(backend_venv_path):
                site_packages_dirs.append(backend_venv_path)
            
            for site_dir in site_packages_dirs:
                possible_paths.extend(glob.glob(os.path.join(site_dir, 'anyio', '_backends', '_asyncio.py')))
            
            # Remove duplicates
            possible_paths = list(set(possible_paths))
            
            patched_count = 0
            for asyncio_backend_path in possible_paths:
                if os.path.exists(asyncio_backend_path):
                    print(f"Checking file at: {asyncio_backend_path}")
                    
                    # Read the file
                    with open(asyncio_backend_path, 'r') as f:
                        content = f.read()
                    
                    # Check if the file needs patching
                    if 'import uvloop' in content and 'if use_uvloop and loop_factory is None:' in content:
                        # Modify the content to avoid importing uvloop on Windows
                        modified_content = content.replace(
                            'if use_uvloop and loop_factory is None:',
                            'if use_uvloop and loop_factory is None and sys.platform != "win32":'
                        )
                        
                        # Also add sys import if it's not already there
                        if 'import sys' not in content:
                            modified_content = 'import sys\n' + modified_content
                        
                        # Write the modified content back
                        with open(asyncio_backend_path, 'w') as f:
                            f.write(modified_content)
                        
                        print(f"Successfully patched anyio at {asyncio_backend_path}")
                        patched_count += 1
            
            if patched_count > 0:
                print(f"Successfully patched {patched_count} anyio installations")
            else:
                print("No anyio installations needed patching")
                
        except ImportError:
            print("anyio module not found")

if __name__ == "__main__":
    patch_anyio()