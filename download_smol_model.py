from huggingface_hub import snapshot_download
from pathlib import Path

destination = Path(__file__).resolve().parent / "models" / "SmolLM2-360M-Instruct"
destination.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    local_dir=str(destination),
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
)
print(f"SmolLM2-360M-Instruct downloaded to: {destination}")
