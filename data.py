import random
import numpy as np
random.seed(0)
np.random.seed(0)

from collections import Counter
from pprint import pprint as pp

import nltk
import pickle

if __name__ == "__main__":
    nltk.download('punkt_tab')

    txt = open("hp.txt", "rb").read().decode("windows-1251")
    txt = txt.lower()

    tokens = nltk.tokenize.word_tokenize(txt)
    c = Counter(tokens)
    stop_words = { ',', '.', 'the', "''", '``', 'an', 'a', 'of', "'s", "n't", "and", "?", "!", "said", "he", "him", "his", "her", "she", "at", "to", "as", "was", "i", "it", "had", "-", "you", "on", "in", "they", "did", "that", "have", "but", "--", "...", "were", "are", "*"}
    tokens = [t for t in tokens if t not in stop_words and c[t] > 5]
    i2w = {i: w for i, w in enumerate(set(tokens))}
    w2i = {w: i for i, w in i2w.items()}
    # print(tokens[:10])

    pickle.dump(i2w, open("i2w.pkl", "wb"))
    pickle.dump(w2i, open("w2i.pkl", "wb"))
    pickle.dump(tokens, open("tokens.pkl", "wb"))
