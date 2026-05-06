#!/usr/bin/env python3
"""One-shot normalization of all processor files from mojibake to true UTF-8 IPA."""
import json
import os

# ── MASTER MAPPING: mojibake string → true IPA ──
# Built from verified cp1252 decoding. The \x90 byte entries are long vowels.
MOJIBAKE_TO_IPA = {
    'b\u00ca\u00b2': 'b\u02b2',
    'c\u00ca\u00b7': 'c\u02b7',
    'd\u00ca\u2019': 'd\u0292',
    'd\u00cc\u00aa': 'd\u032a',
    'e\u00cb\x90': 'e\u02d0',
    'f\u00ca\u00b2': 'f\u02b2',
    'i\u00cb\x90': 'i\u02d0',
    'k\u00ca\u00b7': 'k\u02b7',
    'm\u00ca\u00b2': 'm\u02b2',
    'o\u00cb\x90': 'o\u02d0',
    'p\u00ca\u00b2': 'p\u02b2',
    't\u00ca\u0192': 't\u0283',
    't\u00cc\u00aa': 't\u032a',
    '\u00c3\u00a7': '\u00e7',
    '\u00c5\u2039': '\u014b',
    '\u00c9\u00a1': '\u0261',
    '\u00c9\u00a1\u00ca\u00b7': '\u0261\u02b7',
    '\u00c9\u00aa': '\u026a',
    '\u00c9\u00b2': '\u0272',
    '\u00c9\u00b9': '\u0279',
    '\u00c9\u00be': '\u027e',
    '\u00c9\u0153': '\u025c',
    '\u00c9\u0153\u00cb\x90': '\u025c\u02d0',
    '\u00c9\u0178': '\u025f',
    '\u00c9\u0178\u00ca\u00b7': '\u025f\u02b7',
    '\u00c9\u2013': '\u0256',
    '\u00c9\u2018': '\u0251',
    '\u00c9\u2018\u00cb\x90': '\u0251\u02d0',
    '\u00c9\u2019': '\u0252',
    '\u00c9\u2019\u00cb\x90': '\u0252\u02d0',
    '\u00c9\u201dj': '\u0254j',
    '\u00c9\u203a': '\u025b',
    '\u00c9\u203a\u00cb\x90': '\u025b\u02d0',
    '\u00c9\u2122': '\u0259',
    '\u00ca\u0160': '\u028a',
    '\u00ca\u017d': '\u028e',
    '\u00ca\u0192': '\u0283',
    '\u00ca\u02c6': '\u0288',
    '\u00ca\u02c6\u00ca\u00b2': '\u0288\u02b2',
    '\u00ca\u02c6\u00ca\u00b7': '\u0288\u02b7',
    '\u00ca\u2019': '\u0292',
    '\u00ca\u2030': '\u0289',
    '\u00ca\u2030\u00cb\x90': '\u0289\u02d0',
    '\u00ca\u2039': '\u028b',
}

def normalize(s):
    """Replace all mojibake sequences with true IPA, longest-match first."""
    for moji, ipa in sorted(MOJIBAKE_TO_IPA.items(), key=lambda x: -len(x[0])):
        s = s.replace(moji, ipa)
    return s


def main():
    base = 'processor_dir'

    # ── 1. Fix vocab.json ──
    vp = os.path.join(base, 'vocab.json')
    v = json.load(open(vp, 'r', encoding='utf-8'))
    new_v = {}
    for k, val in v.items():
        new_v[normalize(k)] = val
    with open(vp, 'w', encoding='utf-8') as f:
        json.dump(new_v, f, ensure_ascii=False, indent=2)
    print(f'\u2705 vocab.json: {len(new_v)} entries normalized')

    # ── 2. Fix tokenizer_config.json ──
    tp = os.path.join(base, 'tokenizer_config.json')
    raw = open(tp, 'r', encoding='utf-8').read()
    normalized_raw = normalize(raw)
    with open(tp, 'w', encoding='utf-8') as f:
        f.write(normalized_raw)
    tc = json.loads(normalized_raw)
    print(f'\u2705 tokenizer_config.json: {len(tc["added_tokens_decoder"])} tokens normalized')

    # ── 3. Verify sync ──
    for token_id, token_info in tc['added_tokens_decoder'].items():
        content = token_info['content']
        if content in new_v:
            assert new_v[content] == int(token_id), f'ID mismatch for {content}: vocab={new_v[content]}, config={token_id}'
        elif content in ('<s>', '</s>'):
            pass
    print('\u2705 vocab.json and tokenizer_config.json are in sync!')

    # ── 4. Verify no mojibake remains ──
    remaining = []
    for k in new_v.keys():
        for c in k:
            cp = ord(c)
            # True IPA chars live in 0x0250-0x02FF and 0x0300-0x036F
            # Mojibake chars use Latin-1 Supplement (0x00C0-0x00FF) and Windows-1252 extras
            if cp > 127 and cp < 0x0250 and c not in '\u00e7\u014b':
                remaining.append((k, c, hex(cp)))
                break

    if remaining:
        print(f'\u26a0\ufe0f  {len(remaining)} keys still have suspect chars:')
        for k, c, h in remaining:
            print(f'   {k!r}: {c!r} ({h})')
    else:
        print('\u2705 Zero mojibake remaining in vocab!')

    # ── 5. Final readout ──
    print('\n\u2500\u2500 Final Vocabulary \u2500\u2500')
    for k, v_id in sorted(new_v.items(), key=lambda x: x[1]):
        print(f'  {v_id:3d}: {k}')


if __name__ == '__main__':
    main()
