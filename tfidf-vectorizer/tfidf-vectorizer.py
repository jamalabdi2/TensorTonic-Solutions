import numpy as np
from collections import Counter

def tokenize(documents):
    return [document.strip().lower().split() for document in documents]

def get_vocab(tokens):
    vocab = set()
    for token in tokens:
        vocab.update(token)
    vocab = sorted(vocab)
    return vocab, {word: idx for idx, word in enumerate(vocab)}

def idf(vocab, tokenize_docs):
    df = np.zeros(len(vocab))
    N = len(tokenize_docs)

    for i, word in enumerate(vocab):
        df[i] = sum(1 for doc in tokenize_docs if word in doc)

    return np.log(N / (df + 1e-9))  

def tfidf_vectorizer(documents):
    tokens = tokenize(documents)
    vocab, vocab_idx = get_vocab(tokens)

    tf_matrix = np.zeros((len(documents), len(vocab)))

    for index, token in enumerate(tokens):
        counts = Counter(token)
        total_terms = len(token)

        for word, count in counts.items():
            tf_matrix[index, vocab_idx[word]] = count / total_terms  

    idf_vals = idf(vocab, tokens)

    tfidf = tf_matrix * idf_vals

    return tfidf, vocab  


