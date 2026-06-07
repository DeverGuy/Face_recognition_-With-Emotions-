import os
import pickle
import subprocess
import json

# Define the exact same messages structure, replacing {nm} with a placeholder
MESSAGES = {
    "neutral": [
        "HEY {nm}, YOU LOOK CALM AND FOCUSED.",
        "JUST CHILLING, ARE WE, {nm}?",
        "LOOKING VERY NEUTRAL TODAY, {nm}."
    ],
    "happiness": [
        "LOOKING GOOD, {nm}! LOVE THE SMILE!",
        "GLAD TO SEE YOU SO HAPPY, {nm}!",
        "THAT SMILE LOOKS GREAT ON YOU, {nm}!"
    ],
    "surprise": [
        "WOW! WHAT SURPRISED YOU, {nm}?",
        "DID I STARTLE YOU, {nm}?",
        "YOU LOOK SHOCKED, {nm}!"
    ],
    "sadness": [
        "HEY {nm}, YOU LOOK A BIT DOWN. CHEER UP! YOU'VE GOT THIS!",
        "DON'T BE SAD, {nm}. THINGS WILL GET BETTER!",
        "I'M HERE FOR YOU, {nm}. KEEP YOUR HEAD UP!"
    ],
    "anger": [
        "WHOA {nm}, TAKE A DEEP BREATH! CHILL OUT!",
        "YOU LOOK FURIOUS, {nm}. RELAX!",
        "EASY THERE, {nm}. NO NEED TO BE ANGRY!"
    ],
    "disgust": [
        "YUCK! SAW SOMETHING GROSS, {nm}?",
        "YOU LOOK DISGUSTED, {nm}.",
        "NOT A FAN OF THAT, HUH {nm}?"
    ],
    "fear": [
        "DON'T PANIC, {nm}! EVERYTHING IS FINE!",
        "YOU LOOK TERRIFIED, {nm}! BREATHE!",
        "IT'S OKAY, {nm}. THERE'S NOTHING TO FEAR."
    ],
    "contempt": [
        "WHY THE CONTEMPT, {nm}?",
        "YOU LOOK LIKE YOU'RE JUDGING ME, {nm}!",
        "THAT'S A VERY SCORNFUL LOOK, {nm}."
    ],
    "exhausted": [
        "YOU LOOK EXHAUSTED, {nm}. GET SOME SLEEP!",
        "LONG DAY, {nm}? YOU LOOK TIRED.",
        "COFFEE TIME, {nm}! YOU'RE FALLING ASLEEP!"
    ],
    "shocked": [
        "JAW-DROPPING, ISN'T IT, {nm}?",
        "I KNOW, CRAZY RIGHT, {nm}?",
        "YOU LOOK COMPLETELY STUNNED, {nm}!"
    ],
    "suspicious": [
        "WHY THE SUSPICIOUS LOOK, {nm}?",
        "I PROMISE I'M NOT HIDING ANYTHING, {nm}!",
        "YOU DON'T TRUST ME, DO YOU, {nm}?"
    ],
    "confused": [
        "ARE YOU CONFUSED, {nm}?",
        "LET ME EXPLAIN IT AGAIN, {nm}.",
        "YOU LOOK LIKE YOU HAVE A QUESTION, {nm}."
    ],
    "frustrated": [
        "DON'T LET IT FRUSTRATE YOU, {nm}!",
        "TAKE A BREAK, {nm}. YOU LOOK FRUSTRATED.",
        "DEEP BREATHS, {nm}. FRUSTRATION WON'T HELP!"
    ],
    "euphoric": [
        "YOU ARE GLOWING, {nm}! SO HAPPY!",
        "ABSOLUTELY BEAMING TODAY, {nm}!",
        "LOVE THE ENERGY, {nm}!"
    ],
    "bored": [
        "ZONING OUT ALREADY, {nm}?",
        "AM I BORING YOU, {nm}?",
        "WAKE UP, {nm}! PAY ATTENTION!"
    ],
    "flirty": [
        "WINKING AT ME, {nm}?",
        "OH, YOU'RE FLIRTING NOW, {nm}?",
        "I SAW THAT WINK, {nm}!"
    ],
    "yawning": [
        "ROUGH NIGHT, {nm}? YOU'RE YAWNING!",
        "AM I THAT BORING THAT YOU'RE YAWNING, {nm}?",
        "GET SOME REST, {nm}. BIG YAWN!"
    ],
    "smirking": [
        "WHAT'S WITH THAT SMIRK, {nm}?",
        "YOU THINK YOU'RE CLEVER, {nm}?",
        "THAT'S A SNEAKY SMIRK, {nm}."
    ]
}

def main():
    print("[INFO] Starting Zero-Latency Audio Pre-renderer...")
    
    # Load database to get user names
    names = ["Unknown"]
    if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
        with open("embeddings.pkl", "rb") as f:
            database = pickle.load(f)
            names.extend(list(database.keys()))
            
    os.makedirs("audio", exist_ok=True)
    
    total_messages = len(names) * sum(len(msgs) for msgs in MESSAGES.values())
    current = 0
    
    print(f"[INFO] Need to generate {total_messages} audio files.")
    
    for name in names:
        safe_name = name.upper()
        for emotion, msgs in MESSAGES.items():
            for i, msg_template in enumerate(msgs):
                current += 1
                final_text = msg_template.replace("{nm}", safe_name)
                # Create a safe filename
                filename = f"audio/{name}_{emotion}_{i}.mp3"
                
                if os.path.exists(filename):
                    continue
                    
                print(f"[{current}/{total_messages}] Rendering: {final_text}")
                # Render with edge-tts
                cmd = ["edge-tts", "--voice", "en-GB-SoniaNeural", "--text", final_text, "--write-media", filename]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
    print("[INFO] Audio pre-rendering complete!")

if __name__ == "__main__":
    main()
