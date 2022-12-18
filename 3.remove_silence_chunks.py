from pathlib import Path

import numpy as np
from scipy.io import wavfile
import tgt
from tqdm.auto import tqdm

special_phonemes = ["sil", "sp", "spn", " "]

gt_files = sorted(Path("preprocessed_data/Infore/TextGrid/Infore").glob("*.TextGrid"))

for gt_fn in tqdm(gt_files):
    textgrid = tgt.io.read_textgrid(gt_fn, include_empty_intervals=True)
    wav_fn = f"./Data/Infore_raw/Infore/{gt_fn.stem}.wav"
    sr, y = wavfile.read(wav_fn)
    y = np.copy(y)

    for t in textgrid.get_tier_by_name("phones")._objects:
        s, e, p = t.start_time, t.end_time, t.text
        if len(p) == 0:
            p = "sil"
        if p in special_phonemes:
            l = int(s * sr)
            r = int(e * sr)
            y[l:r] = 0
    out_file = f"./Data/Infore/Infore/{gt_fn.stem}.wav"
    wavfile.write(out_file, sr, y)
