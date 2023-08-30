import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("spn")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

from viphoneme import vi2IPA_split

N_onsets = { u'b' : u'b', u't' : u't', u'th' : u'tʰ', u'đ' : u'd', u'ch' : u'c', 
            u'kh' : u'x', u'g' : u'ɣ', u'l' : u'l', u'm' : u'm', u'n': u'n', 
            u'ngh': u'ŋ', u'nh' : u'ɲ', u'ng' : u'ŋ', u'ph' : u'f', u'v' : u'v', 
            u'x' : u's', u'd' : u'z', u'h' : u'h', u'p' : u'p', u'qu' : u'kw',
            u'gi' : u'z', u'tr' : u'c', u'k' : u'k', u'c' : u'k', u'gh' : u'ɣ', 
            u'r' : u'z', u's' : u's', u'gi': u'z'}

pad = list(set(N_onsets.values()))
Special=['tʰ','tʰw','p','pw','t','tw','ʈ','ʈw','c','cw','k','kw','k','kw','kw',"'",'b','bw',
        'd','dw','f','fw','s','sw','ʂ', 'ʂw', 'x', 'xw', 'h', 'hw', 'v', 'vw', 'z', 'zw', 
        'j', 'jw', 'ʐ', 'ʐw', 'ɣ', 'ɣw', 'ɣ', 'ɣw', 'm', 'mw', 'n', 'nw', 'ɲ', 'ɲw', 
        'ŋ', 'ŋw', 'ŋ', 'ŋw', 'l', 'lw',"'"]
pad += Special

def g2p(w):
    w = w.strip()
    a = vi2IPA_split(w,"/")\
            .replace("/./","")\
            .strip()\
            .split("/")[1:-1]
    
    a = "|".join(a).replace("_", " ")
    ps = a.split("| |")
    res = []
    for a in ps:
        a = a.split("|")
        tone = a[-1] if a[-1].isdigit() else None
        
        if tone is not None:
            if int(tone) in [0,1,2,3,4,5,6]:
                if a[0] in pad:
                    for i in range(1,len(a)):
                        a[i] += tone
                else:
                    for i in range(0,len(a)):
                        a[i] += tone
                    a = ["silp"] + a
                a = a[:-1]
        res.extend(a)
    return res

from text.viphone import valid_symbols

def preprocess_vietnamese(text, preprocess_config):
    phones = []
    words = text.strip().split(" ")
    phonemes = [g2p(word) for word in words]
    for w in phonemes:
        w_p = []
        for p in w:
            if p not in valid_symbols:
                w_p.append("sil")
                break
            else:
                w_p.append(p)
        phones.append(" ".join(w_p))

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control, alophone_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                alophone_control=alophone_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--alophone_control",
        type=float,
        default=0.2,
        help="control the variant of the phoneme, larger value more change of changing a phoneme to other",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "vi":
            texts = np.array([preprocess_vietnamese(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control, args.alophone_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
