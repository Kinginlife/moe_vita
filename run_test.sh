#!/bin/bash
# Quick test script for VITA continual learning data processing

echo "=========================================="
echo "VITA Continual Learning Data Test"
echo "=========================================="

cd "$(dirname "$0")"

echo ""
echo "Running data processing tests..."
python test_continual_data.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
