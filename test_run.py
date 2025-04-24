"""
Test script to verify that the run module imports correctly.
"""
from backend.run import parse_arguments

print("Run module imports successfully!")
args = parse_arguments()
print(f"Default host: {args.host}")
print(f"Default port: {args.port}")
print(f"Default environment: {args.env}")