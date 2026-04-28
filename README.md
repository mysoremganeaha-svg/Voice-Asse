# Voice-Asse
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import pyttsx3
import tempfile
import queue
import sys

# Audio settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds per recording

# Initialize TTS
engine = pyttsx3.init()

# Load Whisper model
model = whisper.load_model("base")

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=DURATION):
    print("Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()
    return audio

def save_wav(audio):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, SAMPLE_RATE, audio)
    return temp_file.name

def transcribe(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

def process_command(text):
    text = text.lower()

    if "hello" in text:
        return "Hello! How can I help in your meeting?"
    elif "summary" in text:
        return "I cannot summarize yet, but I will soon!"
    elif "action item" in text:
        return "Noted. I will track action items."
    elif "exit" in text or "stop" in text:
        speak("Goodbye!")
        sys.exit()
    else:
        return "I heard you say: " + text

def main():
    speak("Voice assistant started")

    while True:
        audio = record_audio()
        audio_file = save_wav(audio)

        try:
            text = transcribe(audio_file)
            print("You:", text)

            if text.strip() == "":
                continue

            response = process_command(text)
            speak(response)

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
