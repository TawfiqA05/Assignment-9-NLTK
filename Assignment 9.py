from pathlib import Path
import argparse, re, sys
from collections import Counter
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, ngrams, FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tabulate import tabulate

FILES = [f"texts/Text_{i}.txt" for i in range(1,5)]
stemmer, lemmatizer = SnowballStemmer("english"), WordNetLemmatizer()

def ensure_nltk():
    for p in ["punkt","averaged_perceptron_tagger","maxent_ne_chunker","words","wordnet"]:
        try: nltk.data.find(f"tokenizers/{p}" if p=="punkt" else f"corpora/{p}")
        except LookupError: nltk.download(p, quiet=True)

def load(p): return Path(p).read_text(encoding="utf-8")
def preprocess(text): return word_tokenize(re.sub(r"[^\w\s]"," ",text.lower()))
def top20(t): return Counter(t).most_common(20)
def stems_lemmas(tok20): return [(stemmer.stem(w), lemmatizer.lemmatize(w)) for w,_ in tok20]
def ner(text):
    return sum(1 for s in nltk.sent_tokenize(text)
                 for n in ne_chunk(pos_tag(word_tokenize(s))) if hasattr(n,"label"))
def trigrams(toks,k=10): return [(" ".join(g),c) for g,c in FreqDist(ngrams(toks,3)).most_common(k)]
def subject(tok20): return ", ".join([w for w,_ in tok20 if w.isalpha()][:3])

def main(show_ner=False):
    ensure_nltk(); tri_sets, summary = {}, []
    for f in FILES:
        raw, toks = load(f), preprocess(load(f))
        top = top20(toks); tri_sets[f] = set(" ".join(g) for g,_ in trigrams(toks,25))
        print(f"\n=== {Path(f).name} ===")
        print(tabulate(top,headers=["token","count"]))
        print("\nStem → Lemma"); print(tabulate(stems_lemmas(top)))
        print("\nTop 10 trigrams"); print(tabulate(trigrams(toks),headers=["trigram","count"]))
        summary.append({"file":Path(f).name,
                        "named_entities": ner(raw) if show_ner else "-",
                        "subject_guess": subject(top)})
    print("\n--- SUMMARY ---"); print(tabulate(summary,headers="keys"))
    base = {k:v for k,v in tri_sets.items() if "Text_4" not in k}
    best = max(base,key=lambda k:len(base[k]&tri_sets["texts/Text_4.txt"]))
    print(f"\nAuthorship hint: Text_4 most resembles {Path(best).name}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-ner",action="store_true")
    try: main(ap.parse_args().show_ner)
    except Exception as e: sys.exit(f"Error: {e}")