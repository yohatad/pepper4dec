# Maps each dec_system node to the Python interpreter it runs under.
# Sourced by the ros2 run launcher scripts in */scripts/.
# Each variable is an interpreter path, invoked directly as: "$VENV_X" -m <module>

VENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# speech_event: faster-whisper, pyroomacoustics, onnxruntime, silero VAD
export VENV_SOUND="$VENVS_DIR/sound/bin/python3"

# conversation_manager: chromadb, sentence-transformers, openai
export VENV_LANGUAGE_MODEL="$VENVS_DIR/conversation/bin/python3"

# text_to_speech: Kokoro-82M, misaki, sounddevice
export VENV_TEXT_TO_SPEECH="$VENVS_DIR/tts_virtual_env/bin/python3"
