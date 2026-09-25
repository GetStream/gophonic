# Copyright 2026 The gophonic authors
# SPDX-License-Identifier: BSD-2-Clause
#
# reference.py writes the greedy FP32 fixtures of the official qwen-tts
# package that qwen3tts's tests compare against:
#
#   uv run --with qwen-tts --with soundfile python qwen3tts/tools/reference.py \
#       models/Qwen3-TTS-12Hz-1.7B-CustomVoice models/qwen3tts-reference
#
# For each case it records the prompt ids, the talker's prompt rows and
# first three step inputs and states, the code predictor's inputs,
# projection, final state and first-codebook logits for three frames, every
# frame's sixteen codes, and the decoded waveform.
import json, os, sys
import numpy as np, torch
from qwen_tts import Qwen3TTSModel

model_dir, out = sys.argv[1], sys.argv[2]
tts = Qwen3TTSModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
model = tts.model
talker, cp = model.talker, model.talker.code_predictor
cases = [("hello", "Hello! I'm Gopher, and I run entirely on this laptop.", "english", "ryan")]

for name, text, lang, spk in cases:
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    seen = {"rows": [], "hidden": [], "cp_in": [], "cp_logits": []}

    def talker_spy(orig):
        def f(*a, **kw):
            e = kw.get("inputs_embeds")
            r = orig(*a, **kw)
            if e is not None and len(seen["rows"]) < 4:
                seen["rows"].append(e[0].detach().float().numpy().copy())
                seen["hidden"].append(r.last_hidden_state[0, -1].detach().float().numpy().copy())
            return r
        return f

    def cp_spy(orig):
        def f(*a, **kw):
            e = kw.get("inputs_embeds")
            r = orig(*a, **kw)
            if e is not None and e.shape[1] == 2 and len(seen["cp_in"]) < 3:
                seen["cp_in"].append(e[0].detach().float().numpy().copy())
                seen["cp_logits"].append(r.logits[0, -1].detach().float().numpy().copy())
            return r
        return f

    talker.model.forward = talker_spy(talker.model.forward)
    cp.forward = cp_spy(cp.forward)
    ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
    codes, _ = model.generate(input_ids=[ids], instruct_ids=[None], languages=[lang], speakers=[spk],
                              non_streaming_mode=False, do_sample=False, subtalker_dosample=False, max_new_tokens=400)
    codes = codes[0].numpy().astype(np.int32)
    wavs, sr = model.speech_tokenizer.decode([{"audio_codes": torch.tensor(codes)}])
    wav = np.asarray(wavs[0], dtype=np.float32)
    with torch.no_grad():
        x = torch.tensor(seen["cp_in"][0][None])
        proj = cp.small_to_mtp_projection(x)
        hidden = cp.model(inputs_embeds=proj, use_cache=False).last_hidden_state
    json.dump({"text": text, "language": lang, "speaker": spk, "ids": ids[0].tolist(), "frames": int(codes.shape[0]),
               "sample_rate": sr}, open(os.path.join(d, "case.json"), "w"))
    save = lambda n, a: np.asarray(a).astype("<f4").tofile(os.path.join(d, n))
    save("prefill.f32", seen["rows"][0])
    save("step_rows.f32", np.concatenate(seen["rows"][1:4]))
    save("step_hidden.f32", np.stack(seen["hidden"][:4]))
    save("cp_inputs.f32", np.stack(seen["cp_in"]))
    save("cp_logits.f32", np.stack(seen["cp_logits"]))
    save("cp_proj0.f32", proj.numpy())
    save("cp_hidden0.f32", hidden[0, -1].numpy())
    codes.astype("<i4").tofile(os.path.join(d, "codes.i32"))
    save("wav.f32", wav)
    print(name, codes.shape, len(wav) / sr, "s")
