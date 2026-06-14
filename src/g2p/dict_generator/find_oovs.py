# find_oovs.py
dict_words = {line.split()[0].lower() for line in open("output.dict", encoding="utf8")}
corpus_words = {line.strip().lower() for line in open("words.txt", encoding="utf8")}

oovs = sorted(corpus_words - dict_words)
with open("oovs_found.txt", "w", encoding="utf8") as f:
    f.write("\n".join(oovs))

print(f"Found {len(oovs)} OOV words. Wrote to oovs_found.txt")
