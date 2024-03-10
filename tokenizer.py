import re
from typing import List, Tuple

# import nltk
#
# nltk.download("stopwords")

from nltk.corpus import stopwords
from collections import Counter

_english_stopwords = frozenset(stopwords.words('english'))
_corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
_RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

_all_stopwords = _english_stopwords.union(_corpus_stopwords)

from nltk.stem.porter import *

stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    tokens = [token.group() for token in _RE_WORD.finditer(text.lower())]
    filtered = [token for token in tokens if token not in _all_stopwords]
    return filtered


def word_count(text: str, id: int) -> List[Tuple[str, Tuple[int, int]]]:
    ''' Count the frequency of each word in `text` (tf) that is not included in
    `all_stopwords` and return entries that will go into our posting lists.
    Parameters:
    -----------
      text: str
        Text of one document
      id: int
        Document id
    Returns:
    --------
      List of tuples
        A list of (token, (doc_id, tf)) pairs
        for example: [("Anarchism", (12, 5)), ...]
    '''
    tokens = tokenize(text)
    counter = Counter(tokens)

    return [(token, (id, tf)) for token, tf in counter.items()]
