#!/bin/sh
# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause

# fetch-models.sh downloads the official checkpoints that gophonic's tests,
# benchmarks, and examples use and converts them into the models directory:
# models/ at the repository root, or $GOPHONIC_MODELS. Files already present
# are kept, so the script is safe to rerun.
#
# Usage: tools/fetch-models.sh [whisper] [turn] [qwen3] [clm] (default: whisper turn)
#
# It needs curl and Python 3; the converters' packages (NumPy, ONNX, PyTorch,
# Transformers) are installed on the fly with uv when uv is available.
# Checkpoints are verified against the SHA-256 digests the converters pin.
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${GOPHONIC_MODELS:-$repo/models}
cache=$dir/.downloads
mkdir -p "$dir" "$cache"

# py runs a converter with the Python packages it needs.
py() {
	packages=$1
	shift
	if command -v uv >/dev/null 2>&1; then
		with=""
		for p in $packages; do with="$with --with $p"; done
		# shellcheck disable=SC2086
		uv run --quiet --no-project $with python "$@"
	else
		python3 "$@"
	fi
}

# hf downloads a Hugging Face repository (or some of its files) into a directory.
hf() {
	if command -v uvx >/dev/null 2>&1; then
		uvx --quiet --from huggingface_hub hf download "$@"
	else
		command hf download "$@"
	fi
}

fetch() { # URL FILE
	[ -f "$2" ] && return
	curl -fL --retry 3 -o "$2.part" "$1"
	mv "$2.part" "$2"
}

whisper() {
	for name in tiny.en base.en; do
		out=$dir/$name.gophonic
		[ -e "$out" ] && continue
		case $name in
		tiny.en) sum=d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03 ;;
		base.en) sum=25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead ;;
		esac
		fetch "https://openaipublic.azureedge.net/main/whisper/models/$sum/$name.pt" "$cache/$name.pt"
		py numpy "$repo/tools/whisper_pt_to_gophonic.py" "$cache/$name.pt" "$out"
	done
}

turn() {
	if [ ! -e "$dir/smart-turn-v3.2.gophonic" ]; then
		fetch https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx "$cache/smart-turn-v3.2-gpu.onnx"
		py "numpy onnx" "$repo/tools/onnx_to_gophonic.py" "$cache/smart-turn-v3.2-gpu.onnx" "$dir/smart-turn-v3.2.gophonic"
	fi
	if [ ! -e "$dir/tinymel.gophonic" ]; then
		fetch https://huggingface.co/deveshu/hinglish-turn-detector/resolve/main/model_tinymel_int8.onnx "$cache/model_tinymel_int8.onnx"
		py "numpy onnx" "$repo/tools/tinymel_to_gophonic.py" "$cache/model_tinymel_int8.onnx" --bundle "$dir/tinymel.gophonic"
	fi
}

qwen3() {
	[ -e "$dir/Qwen3-8B" ] || hf Qwen/Qwen3-8B --local-dir "$dir/Qwen3-8B"
	[ -e "$dir/qwen3-8b-hello-reference.f32" ] ||
		py "torch transformers accelerate" "$repo/qwen3/tools/reference_hidden.py" "$dir/Qwen3-8B" "$dir/qwen3-8b-hello-reference.f32"
}

clm() {
	[ -e "$dir/CLM_v0.1-8B.gclm" ] && return
	hf Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --revision 87655cb835bd76fd66c2da78e1e3709f7fa11a94 --local-dir "$cache"
	py "torch numpy" "$repo/tools/clm_pt_to_gophonic.py" "$cache/CLM_v0.1-8B.pt" "$dir/CLM_v0.1-8B.gclm" \
		--expected-sha256 b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5 \
		--source-revision 87655cb835bd76fd66c2da78e1e3709f7fa11a94
}

[ $# -gt 0 ] || set -- whisper turn
for target in "$@"; do
	case $target in
	whisper | turn | qwen3 | clm) "$target" ;;
	*)
		echo "fetch-models.sh: unknown target $target (want whisper, turn, qwen3, or clm)" >&2
		exit 2
		;;
	esac
done
echo "models in $dir:"
ls "$dir"
