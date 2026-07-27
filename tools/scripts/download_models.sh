#!/usr/bin/env bash
# Pre-fetch model weights so the first pipeline run is not blocked on downloads.
# All weights are public and otherwise download automatically on first use.
#
#   MolmoPoint-Vid-4B  allenai/MolmoPoint-Vid-4B   (HuggingFace, cached in HF_HOME)
#   Molmo2-8B          allenai/Molmo2-8B           (HuggingFace, cached in HF_HOME)
#   Qwen3-0.6B         Qwen/Qwen3-0.6B             (HuggingFace, cached in HF_HOME)
#   SAM 3              facebook/sam3               (HuggingFace, cached in HF_HOME)
#   AllTracker         aharley/alltracker          (torch.hub, cached in TORCH_HOME)
#
# Set HF_HOME / TORCH_HOME to control cache locations.
set -euo pipefail

echo "Pre-fetching AllTracker checkpoint into the torch.hub cache..."
python - <<'PY'
import torch
url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
torch.hub.load_state_dict_from_url(url, map_location="cpu")
print("AllTracker checkpoint cached.")
PY

echo "Pre-fetching HuggingFace checkpoints (this may take a while)..."
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["allenai/MolmoPoint-Vid-4B", "allenai/Molmo2-8B",
             "Qwen/Qwen3-0.6B", "facebook/sam3"]:
    print(f"  {repo} ...")
    snapshot_download(repo)
print("All HuggingFace checkpoints cached.")
PY

echo "Done."
