import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

cleaned_txt = set([x.replace("\n", "") for x in open("./Data/Infore_dev/files.txt", "r").readlines() if x.replace("\n", "") != ""])

def prepare_align(config):
    in_dir = "./Data/Infore_raw/Infore"
    out_dir = "./Data/Infore/"
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    speaker = "Infore"
    tk = tqdm(cleaned_txt)
    for base_name in tk:
        base_name = base_name.strip()

        wav_path = os.path.join(in_dir, f"{base_name}.wav")
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, sr = librosa.load(wav_path, sr=sampling_rate)
            wav = ((wav / max(abs(wav)))) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, f"{base_name}.wav"),
                sampling_rate,
                wav.astype(np.int16),
            )
            tk.set_postfix(max_wave_v = max(abs(wav)), max_v=max_wav_value, sample_rate=sr)

import yaml

config = "config/Infore/preprocess.yaml"

config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
prepare_align(config)
