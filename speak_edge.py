import sys
import subprocess
import os
import uuid
import glob

# Clean up any leftover mp3 files from previous runs
for f in glob.glob("temp_speech_*.mp3"):
    try:
        os.remove(f)
    except:
        pass

if len(sys.argv) > 1:
    text = sys.argv[1]
    
    # Generate a unique filename to prevent playsound file locking bugs
    unique_id = uuid.uuid4().hex
    audio_file = f"temp_speech_{unique_id}.mp3"
    
    # Generate audio with edge-tts using a sweet, melodious voice
    subprocess.run(["edge-tts", "--voice", "en-US-AriaNeural", "--rate", "+5%", "--text", text, "--write-media", audio_file], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(audio_file):
        from playsound import playsound
        try:
            playsound(audio_file)
        except Exception as e:
            print(f"[Playsound Error]: {e}")
        
        # Optionally remove the file so it doesn't accumulate
        try:
            os.remove(audio_file)
        except:
            pass
