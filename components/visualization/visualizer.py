from os import path
from audio2numpy import open_audio

def load_audio(audio_file):
    if not path.exists(audio_file):
        raise FileNotFoundError("Audio file does not exist")
    else:
        waveform, sample_rate = open_audio(audio_file)
        return waveform, sample_rate