import moviepy.editor
import numpy as np
from functools import reduce
from pathlib import Path
from pydub import AudioSegment

SOUNDTRACK_FILE = Path("soundtrack.wav")
RNG = np.random.default_rng()


def save_soundtrack(noises, duration=50_000):
    quiet = AudioSegment.silent(duration=duration)
    soundtrack = reduce(lambda a, b: a.overlay(b), noises, quiet)
    soundtrack.export(SOUNDTRACK_FILE, format="wav")


def add_soundtrack():
    soundtrack = moviepy.editor.AudioFileClip(str(SOUNDTRACK_FILE))
    feature_film = moviepy.editor.VideoFileClip("starry.mp4")
    scape = feature_film.set_audio(soundtrack)
    scape.write_videofile("starry_song.mp4", audio_codec="aac")
    soundtrack.close()
    feature_film.close()
    scape.close()


def random_start(noise, fade=True):
    start = RNG.integers(len(noise))
    print(f'{start=}')
    return (noise[start:] + noise[:start])



def random_soundtrack():
    engine_noise = RNG.choice(
        list(Path("audio", "Engine Output").glob("*.wav")), RNG.integers(1, 3)
    ).tolist()
    sensor_sound = RNG.choice(
        list(Path("audio", "Sensor Output").glob("*.wav")), RNG.integers(2)
    ).tolist()
    print(f"Soundtrack = {engine_noise} + {sensor_sound}")
    return [random_start(AudioSegment.from_file(fp)) for fp in engine_noise] + [
        random_start(AudioSegment.from_file(fp)) - 3 for fp in sensor_sound
    ]


def normalize_dBFS(noises):
    quietest = 0
    for n in noises:
        quietest = min(quietest, n.dBFS)
        print(quietest)
    return [n - n.dBFS + quietest - i for i, n in enumerate(noises)]


def jacn_sound_go():
    soundtrack = normalize_dBFS(random_soundtrack())
    print(f"{soundtrack[0].dBFS=}")
    save_soundtrack(soundtrack)
    add_soundtrack()
