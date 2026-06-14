# run using mic
# python test_model.py --model_dir phoneme_model_out --duration 4.0

# exisitng wave file
# python test_model.py --model_dir phoneme_model_out --wav examples/test.wav

import argparse, sounddevice as sd, soundfile as sf, torch, numpy as np, os, json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

def collapse_and_remove_pad(ids, pad_id):
    out = []
    prev = None
    for i in ids:
        if i == prev:
            continue
        prev = i
        if i == pad_id:
            continue
        out.append(int(i))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="phoneme_wav2vec2_out")
    p.add_argument("--wav", default=None, help="optional wav file to infer; if not given, records mic")
    p.add_argument("--duration", type=float, default=3.0, help="mic record seconds")
    args = p.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pad_id = processor.tokenizer.pad_token_id

    with open(os.path.join(args.model_dir, "phoneme2id.json"), "r", encoding="utf8") as f:
        phoneme2id = json.load(f)
    id2phoneme = {int(v): k for k, v in phoneme2id.items()}

    if args.wav:
        speech, sr = sf.read(args.wav)
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            speech = torchaudio.functional.resample(torch.tensor(speech), sr, 16000).numpy()
            sr = 16000
    else:
        sr = 16000
        print(f"Recording {args.duration}s ...")
        rec = sd.rec(int(args.duration*sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        speech = rec.squeeze()

    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
    collapsed = collapse_and_remove_pad(pred_ids, pad_id)
    phonemes = [id2phoneme[i] for i in collapsed]

    print("Predicted phonemes:", phonemes)
