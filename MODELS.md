# Pretrained Models

This repository's own code is licensed BSD-3-Clause (see each package's
`package.xml`/`LICENSE`). That license covers the code in this repository —
it does **not** extend to the third-party pretrained model weights the code
loads at runtime. Those weights carry their own, separate licenses from
their respective authors, and several are copyleft (GPL-3.0 / AGPL-3.0)
rather than permissive.

None of the `.onnx`/checkpoint files below are committed to this repo —
`models/` directories are gitignored (see `.gitignore`). Each package's
README has setup instructions for downloading or placing the actual weight
file; this document exists so that provenance and license are recorded in
one place instead of scattered across per-package feature bullets.

## Locally-run models (weight file loaded at runtime)

| Model | Task | Used by | Checkpoint | License | Source |
|---|---|---|---|---|---|
| YOLO11m (Ultralytics) | Person detection | `person_detection` | `person_detection_yolov11m.onnx` | **AGPL-3.0** (or Ultralytics Enterprise License for closed-source use) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| Gold-YOLO | Face detection | `face_detection` | `face_detection_goldYOLO.onnx` | **GPL-3.0** | [huawei-noah/Efficient-Computing (Gold-YOLO)](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) · [paper](https://ar5iv.labs.arxiv.org/html/2309.11331) |
| 6DRepNet360 (SixDRepNet) | Head-pose estimation / mutual gaze | `face_detection` | `face_detection_sixdrepnet360.onnx` | MIT | [thohemp/6DRepNet360](https://github.com/thohemp/6DRepNet360) |
| MiVOLO | Age/gender estimation | `face_detection` (`age_gender_detection` node) | `face_detection_mivolo_agegender.onnx` | Apache-2.0 (verify the repo's `LICENSE` file directly before commercial use — sources disagreed on this one during review) | [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO) |
| Silero VAD | Voice activity detection | `speech_event` | `silero_vad.onnx` | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| faster-whisper-large-v3-turbo (CT2) | Speech-to-text (Whisper ASR) | `speech_event` | fetched by ID `deepdml/faster-whisper-large-v3-turbo-ct2` (HuggingFace Hub cache, not stored in `models/`) | MIT (both the `faster-whisper` runtime and this checkpoint) | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) |
| Kokoro-82M | Local text-to-speech (`kokoro_local`/`kokoro_pepper` backends) | `text_to_speech` | fetched via the `kokoro` pip package (not stored in `models/`) | Apache-2.0 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| all-MiniLM-L6-v2 (sentence-transformers) | Text embedding for RAG retrieval | `conversation_manager` | fetched via `sentence-transformers` (HuggingFace Hub cache) | Apache-2.0 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

## Algorithm/code (no separate weight file)

| Component | Task | Used by | License | Source |
|---|---|---|---|---|
| ByteTrack | Multi-object tracking (Kalman filter + IoU association) | `person_detection`, `face_detection` (`byte_tracker.cpp`, adapted) | MIT | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) |

## API-based (no weights run or shipped locally)

| Service | Task | Used by | Notes |
|---|---|---|---|
| DeepSeek (`deepseek-chat`) | LLM for RAG responses | `conversation_manager` | Any OpenAI-compatible API works (`llm.base_url`/`llm.model` in config); usage governed by [DeepSeek's](https://www.deepseek.com) API terms, not an open-source license. |
| ElevenLabs | Cloud text-to-speech (`elevenlabs_local`/`elevenlabs_pepper` backends) | `text_to_speech` | Usage governed by [ElevenLabs'](https://elevenlabs.io) API terms, not an open-source license. |

## Why this matters here specifically

Two of the locally-run models are copyleft, not permissive, and their
`.onnx` files are plain data as far as this repo's code is concerned (loaded
through ONNX Runtime, not linked against GPL/AGPL source) — but if you ever
redistribute those weight files themselves, or build a networked product
around YOLO11m without an Ultralytics Enterprise License, the GPL-3.0/AGPL-3.0
terms of the weights apply independently of this repo's own BSD-3-Clause
code license:

- **YOLO11m** (`person_detection`) — AGPL-3.0. Ultralytics requires either
  open-sourcing the full project under AGPL-3.0 or purchasing an Enterprise
  License for any non-AGPL use, including internal/SaaS deployments.
- **Gold-YOLO** (`face_detection`) — GPL-3.0.

If either of these ever needs to ship in a closed-source or commercial
context, swap the detector rather than rely on this repo's BSD-3-Clause
license to cover it — it doesn't.
