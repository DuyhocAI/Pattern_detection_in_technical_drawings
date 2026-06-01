#!/bin/bash

# Push to HuggingFace Spaces
# Prerequisites: HF token configured in git credentials

echo "Setting up HuggingFace Space..."

# Add HF remote
git remote add huggingface https://huggingface.co/spaces/DuyhocAI/Pattern_detection_in_technical_drawings.git 2>/dev/null || \
git remote set-url huggingface https://huggingface.co/spaces/DuyhocAI/Pattern_detection_in_technical_drawings.git

echo ""
echo "Pushing code to HuggingFace Spaces..."
echo "You may be prompted to enter HF token"
echo ""

git push huggingface main

if [ $? -eq 0 ]; then
    echo ""
    echo "Push successful!"
    echo ""
    echo "HuggingFace will now:"
    echo "1. Build Docker image (3-5 min)"
    echo "2. Deploy to Space"
    echo "3. Make it live at:"
    echo "   https://huggingface.co/spaces/DuyhocAI/Pattern_detection_in_technical_drawings"
    echo ""
    echo "Monitor the build here:"
    echo "   https://huggingface.co/spaces/DuyhocAI/Pattern_detection_in_technical_drawings/settings"
else
    echo "Push failed - check authentication"
fi
