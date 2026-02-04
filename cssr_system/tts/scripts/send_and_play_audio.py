#!/usr/bin/python2
from naoqi import ALProxy
import subprocess
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description='Send and play audio on NAO robot.')

parser.add_argument('--ip', type=str, required=True, help='IP address of the NAO robot')
parser.add_argument('--port', type=int, required=True, help='Port of the NAO robot')
parser.add_argument('--file', type=str, required=True, help='Path to the temporary WAV file')

args = parser.parse_args()

IP = args.ip
PORT = args.port
audio_path = args.file
audio_name = audio_path.split("/")[-1]

print "Local audio file:", audio_path
print "Audio file name:", audio_name

# Transfer file to NAO robot - USE FULL PATHS
print "Transferring audio file to NAO..."
result = subprocess.call([
    "/usr/bin/sshpass", "-p", "nao", 
    "/usr/bin/scp",
    "-o", "StrictHostKeyChecking=no",
    audio_path, 
    "nao@" + IP + ":/home/nao/" + audio_name
])

if result != 0:
    print "Error: File transfer failed with exit code:", result
    sys.exit(1)

print "File transferred successfully"

# Small delay to ensure filesystem sync
time.sleep(0.2)

# Verify file exists on NAO - USE FULL PATHS
print "Verifying file on NAO..."
verify_result = subprocess.call([
    "/usr/bin/sshpass", "-p", "nao", 
    "/usr/bin/ssh",
    "-o", "StrictHostKeyChecking=no",
    "nao@" + IP,
    "test -f /home/nao/" + audio_name
])

if verify_result != 0:
    print "Error: File not found on NAO after transfer!"
    sys.exit(1)

print "File verified on NAO"
print "Initializing the audio player"

try:
    audio_player = ALProxy("ALAudioPlayer", IP, PORT)
    remote_file_path = "/home/nao/" + audio_name
    print "Playing file:", remote_file_path
    
    taskid = audio_player.post.playFile(remote_file_path)  # Non-blocking call
    
    # Wait for playback to finish
    while audio_player.isRunning(taskid):
        time.sleep(0.1)
    
    print "Audio playback completed"
except Exception as e:
    print "Error playing audio:", str(e)
    sys.exit(1)

print "Cleaning up..."
# Clean up the file on NAO - USE FULL PATHS
subprocess.call([
    "/usr/bin/sshpass", "-p", "nao", 
    "/usr/bin/ssh",
    "-o", "StrictHostKeyChecking=no",
    "nao@" + IP, 
    "rm -f /home/nao/" + audio_name
])

print "Done"
            # Local playback using aplay (Linux) or ffplay (cross-platform)
            # subprocess.run(["aplay", audio_file])   