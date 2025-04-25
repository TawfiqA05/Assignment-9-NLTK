# Assignment-9-NLTK

Analyze four text files with NLTK to:

1. list the 20 most-common tokens (plus stems & lemmas)  
2. count named entities (optional flag)  
3. list top trigrams and guess whether any author of Text 1-3 wrote Text 4

---

## Quick Start

```bash
# set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run analysis
python main.py           # core output
python main.py --show-ner  # include named-entity counts

# run tiny unit tests
pytest
