try:
    from pydantic_settings import BaseSettings
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")