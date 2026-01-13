#!/bin/bash
# =============================================================================
# UNet Architecture Deep Dive Script
# =============================================================================
# This script runs the understand_unet.py to explore the architecture
# of the UNet2DConditionModel used in Stable Diffusion.
#
# Usage: ./explore_unet.sh
# =============================================================================

set -e

echo "=============================================="
echo "🔍 Deep Diving into UNet Architecture"
echo "=============================================="
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Run the UNet exploration script in exploration mode
python understand_unet.py --explore_unet \
    --instance_data_dir="./dummy_data" \
    --instance_prompt="dummy prompt"
