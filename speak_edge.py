import sys
import subprocess
import os

if len(sys.argv) > 1:
    text = sys.argv[1]
    audio_file = "temp_speech.mp3"
    
    # If a pre-rendered audio path is provided, use it!
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        audio_file = sys.argv[2]
    else:
        # Fallback to generating it on the fly if pre-rendering is missing
        subprocess.run(["edge-tts", "--voice", "en-GB-SoniaNeural", "--text", text, "--write-media", audio_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(audio_file):
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        pygame.mixer.quit()
