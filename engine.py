from collections import defaultdict
from typing import Optional

from index.tf_idf_index import TfIdfIndex
from index.title_index import TitleIndex
from index.token_index import TokenIndexer
from tokenizer import tokenize


class QueryEngine:

    def __init__(self, bucket_name: Optional[str] = None):
        self.token_index = TokenIndexer()
        self.tf_idf_index = TfIdfIndex(bucket_name)
        self.title_index = TitleIndex(bucket_name)

    def query(self, text: str):
        tokens = tokenize(text)
        token_indices = [self.token_index.index_of(token) for token in tokens]
        token_indices = [x for x in token_indices if x is not None]

        scores = defaultdict(list)
        for token_idx in token_indices:
            for (doc_id, score) in self.tf_idf_index.values_of(token_idx):
                if self.title_index.length_of(doc_id) > 100:
                    scores[doc_id].append(score)

        combined = [(doc_id, sum(x) * len(x) / len(token_indices)) for doc_id, x in scores.items()]
        sorted_scores = sorted(combined, key=lambda x: x[1], reverse=True)[:100]
        return [(f"https://en.wikipedia.org/?curid={k}", self.title_index.title_of(k), str(v)) for k, v in sorted_scores]
        # return [(self.title_index.title_of(k), str(v)) for k, v in sorted_scores]


if __name__ == '__main__':
    engine = QueryEngine()
    while True:
        query = input("Enter query:\n")
        result = engine.query(query)
        for r in result:
            print(r)
