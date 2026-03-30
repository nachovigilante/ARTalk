#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <wav_directory> [options]

Run ARTalk inference on all .wav files in a directory.

Arguments:
  <wav_directory>       Path to directory containing .wav files

Options:
  -s, --style ID        Style motion ID (default: natural_0)
  -i, --shape ID        Shape/appearance ID (default: mesh)
  -l, --clip-length N   Max frames to render (0 = full audio length, default: 0)
  -o, --output DIR      Output directory (default: render_results/batch)
  -h, --help            Show this help message
EOF
    exit "${1:-0}"
}

# Defaults
STYLE_ID="natural_0"
SHAPE_ID="mesh"
CLIP_LENGTH=0
OUTPUT_DIR=""

# Parse args
WAV_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--style)      STYLE_ID="$2"; shift 2 ;;
        -i|--shape)      SHAPE_ID="$2"; shift 2 ;;
        -l|--clip-length) CLIP_LENGTH="$2"; shift 2 ;;
        -o|--output)     OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)       usage 0 ;;
        -*)              echo "Unknown option: $1" >&2; usage 1 ;;
        *)
            if [[ -z "$WAV_DIR" ]]; then
                WAV_DIR="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; usage 1
            fi
            ;;
    esac
done

if [[ -z "$WAV_DIR" ]]; then
    echo "Error: wav directory is required." >&2
    usage 1
fi

if [[ ! -d "$WAV_DIR" ]]; then
    echo "Error: '$WAV_DIR' is not a directory." >&2
    exit 1
fi

# Resolve project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: venv not found at $PROJECT_DIR/.venv — run 'uv pip install -e .' first." >&2
    exit 1
fi

# Collect wav files
shopt -s nullglob
WAV_FILES=("$WAV_DIR"/*.wav "$WAV_DIR"/*.WAV)
shopt -u nullglob

if [[ ${#WAV_FILES[@]} -eq 0 ]]; then
    echo "No .wav files found in '$WAV_DIR'." >&2
    exit 1
fi

echo "Found ${#WAV_FILES[@]} wav file(s) in '$WAV_DIR'"
echo "Style: $STYLE_ID | Shape: $SHAPE_ID | Clip length: $CLIP_LENGTH"
echo "---"

# Run batch inference via a single Python process to avoid reloading the model per file
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/render_results/batch}"
exec "$VENV_PYTHON" - "$PROJECT_DIR" "$STYLE_ID" "$SHAPE_ID" "$CLIP_LENGTH" "$OUTPUT_DIR" "${WAV_FILES[@]}" <<'PYEOF'
import sys
import os
import torch
import torchaudio

# Parse arguments passed from bash
args = sys.argv[1:]
project_dir = args[0]
style_id = args[1]
shape_id = args[2]
clip_length = int(args[3])
output_dir = args[4]
wav_files = args[5:]

os.chdir(project_dir)
sys.path.insert(0, ".")

torch.set_float32_matmul_precision("high")

from inference import ARTAvatarInferEngine

os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
engine = ARTAvatarInferEngine(load_gaga=False, fix_pose=False, clip_length=999999)
engine.output_dir = output_dir
print(f"Device: {engine.device}")

if style_id != "default":
    engine.set_style_motion(style_id)

total = len(wav_files)
for idx, wav_path in enumerate(wav_files, 1):
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    save_name = f"{basename}_{style_id}_{shape_id}"
    print(f"\n[{idx}/{total}] Processing: {os.path.basename(wav_path)}")

    audio, sr = torchaudio.load(wav_path)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

    import math
    effective_clip = clip_length if clip_length > 0 else math.ceil(audio.shape[0] / 16000 * 25)
    pred_motions = engine.inference(audio, clip_length=effective_clip)
    engine.rendering(audio, pred_motions, shape_id=shape_id, save_name=save_name)
    torch.save(pred_motions.float().cpu(), os.path.join(output_dir, f"{save_name}_motions.pt"))
    print(f"  -> {output_dir}/{save_name}.mp4")

print(f"\nDone! {total} video(s) saved to {output_dir}")
PYEOF
