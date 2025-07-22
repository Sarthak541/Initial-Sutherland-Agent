#!/bin/bash

# This script runs the python scripts for the pipeline in order.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

# Check if an input file argument was provided.
if [ -z "$1" ]; then
  echo "Error: No input file specified."
  echo "Usage: ./run_pipeline.sh /path/to/your/file.csv"
  exit 1
fi

INPUT_FILE=$1

echo "🚀 Starting ingestion for: $INPUT_FILE"
# Pass the input file as an argument to ingestion.py
python3 ingestion.py "$INPUT_FILE"
echo "✅ Ingestion complete."

echo "🚀 Starting agent extraction..."
python3 agent_extractor.py
echo "✅ Agent extraction complete."

echo "🚀 Starting SQL agent..."
python3 sql_agent.py
echo "✅ SQL agent complete."

echo "🎉 Pipeline finished successfully!"