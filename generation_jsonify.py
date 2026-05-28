import argparse
from GenerationLogger import GenerationLogger

parser = argparse.ArgumentParser(description="Convert a generation log HDF5 file to JSON.")
parser.add_argument("--file", required=True, help="Path to the .h5 generation log file")
args = parser.parse_args()

output = GenerationLogger.jsonnify(args.file)
print(f"Written to: {output}")
