"""Four n-gram views of a passage, stacked into one sparse matrix.

- char: raw characters, 1 to 5 grams. Catches em dashes, curly quotes and spacing.
- word: words and punctuation marks as tokens, 1 and 2 grams.
- shape: every word turned into its shape (Xxxx, dd), so habits of case and length show.
- function: every content word replaced by CONTENT, so only grammar and punctuation are left.

The last two hide the topic, which matters because test prompts are new.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

TOKEN_PATTERN = r"\w+|[^\w\s]"
TOKEN = re.compile(TOKEN_PATTERN)
FUNCTION_WORDS = ENGLISH_STOP_WORDS | {"s", "t", "d", "ll", "re", "ve", "m"}


def word_shape(text):
    shapes = []
    for token in TOKEN.findall(text):
        s = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in token)
        shapes.append(re.sub(r"(.)\1{4,}", r"\1\1\1\1", s))
    return " ".join(shapes)


def function_words(text):
    return " ".join(
        t if t.lower() in FUNCTION_WORDS or not t[0].isalnum() else "CONTENT" for t in TOKEN.findall(text)
    )


def four_views(min_df=2):
    common = dict(sublinear_tf=True, min_df=min_df, dtype=np.float32)
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, **common)),
        ("word", TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), lowercase=True, **common)),
        ("shape", TfidfVectorizer(preprocessor=word_shape, token_pattern=r"\S+", ngram_range=(1, 3), lowercase=False, **common)),
        ("function", TfidfVectorizer(preprocessor=function_words, token_pattern=r"\S+", ngram_range=(1, 3), lowercase=False, **common)),
    ])
