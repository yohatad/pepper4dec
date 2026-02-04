#!/usr/bin/env python3.10

import os
import time
import threading
import subprocess
import tempfile
import wave
import numpy as np
from queue import Queue
from ament_index_python.packages import get_package_share_directory, get_package_prefix

try:
    from RealtimeTTS import TextToAudioStream, CoquiEngine
    REALTIME_TTS_AVAILABLE = True
except ImportError:
    REALTIME_TTS_AVAILABLE = False

class TTSProcessor:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()
        self.chunk_queue = Queue()
        self.generation_complete = threading.Event()
        self.is_generating = False
        
        # ROS 2 Parameters
        self.nao_ip = node.declare_parameter("nao_ip", "172.29.111.240").value
        self.nao_port = node.declare_parameter("nao_port", 9559).value
        self.chunk_duration = node.declare_parameter("chunk_duration", 15.0).value
        
        # Paths - FIXED
        package_share = get_package_share_directory("tts")
        package_prefix = get_package_prefix("tts")
        
        # Python2 script is installed in lib directory
        self.python2_script = os.path.join(package_prefix, "lib", "tts", "send_and_play_audio.py")
        
        # Voice clone is in share directory
        voice_path = os.path.join(package_share, "voice_clones", "pepper.wav")
        
        # Log the paths for debugging
        self.logger.info(f"Python2 script path: {self.python2_script}")
        self.logger.info(f"Voice clone path: {voice_path}")
        
        # Verify files exist
        if not os.path.exists(self.python2_script):
            self.logger.error(f"Python2 script not found at: {self.python2_script}")
        if not os.path.exists(voice_path):
            self.logger.error(f"Voice clone not found at: {voice_path}")

        if not REALTIME_TTS_AVAILABLE:
            self.logger.error("RealtimeTTS not installed.")
            return

        engine = CoquiEngine(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            voice=voice_path, language="en"
        )
        self.stream = TextToAudioStream(engine, muted=True)

    def _on_audio_chunk(self, chunk):
        if self.is_generating:
            self.chunk_queue.put(chunk)

    def _send_chunk_to_nao(self, audio_data, sample_rate, chunk_id):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_filename = f.name
            
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            # Call the Python 2 script for NAO communication
            subprocess.Popen([
                '/usr/bin/python2', self.python2_script,
                '--ip', self.nao_ip, '--port', str(self.nao_port),
                '--file', temp_filename
            ])
            return temp_filename
        except Exception as e:
            self.logger.error(f"Failed to send chunk: {e}")
            return None

    def process_and_stream(self, text, feedback_callback):
        self.is_generating = True
        self.generation_complete.clear()
        while not self.chunk_queue.empty(): self.chunk_queue.get()

        # Thread 1: Generation
        def generate():
            self.stream.feed(text)
            self.stream.play_async(on_audio_chunk=self._on_audio_chunk, muted=True)
            while self.stream.is_playing(): time.sleep(0.1)
            self.is_generating = False
            self.generation_complete.set()

        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        # Thread 2: Streaming to Robot
        sample_rate = 24000
        chunk_size = int(sample_rate * self.chunk_duration * 2)
        audio_buffer = b''
        temp_files = []

        try:
            while not self.generation_complete.is_set() or not self.chunk_queue.empty():
                try:
                    chunk = self.chunk_queue.get(timeout=0.1)
                    chunk_bytes = (chunk * 32767).astype(np.int16).tobytes() if isinstance(chunk, np.ndarray) else chunk
                    audio_buffer += chunk_bytes
                    
                    if len(audio_buffer) >= chunk_size:
                        tmp = self._send_chunk_to_nao(audio_buffer, sample_rate, "mid")
                        if tmp: temp_files.append(tmp)
                        audio_buffer = b''
                        feedback_callback("Streaming chunk...")
                except: continue
            
            if audio_buffer:
                self._send_chunk_to_nao(audio_buffer, sample_rate, "final")
            
            return True, "Success"
        finally:
            gen_thread.join()