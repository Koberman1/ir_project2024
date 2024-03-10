import math
import pickle
import struct
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from tqdm import tqdm

from index.token_index import TokenIndexer
from utils.parquet_utils import parquet_file_iterator
from tokenizer import tokenize
from utils.file_utils import MultiFileWriter, get_bucket, _open, MultiFileReader

_NUM_BUCKETS = 1024


def _process_record(record: Dict, indexer: TokenIndexer) -> Dict[int, float]:
    text = record['text']
    tokens = tokenize(text)
    tokens = [indexer.index_of(token) for token in tokens]
    tokens = [token for token in tokens if token is not None]
    total_tokens_in_doc = len(tokens)
    token_counts = Counter(tokens)

    return {token_id: count / total_tokens_in_doc for (token_id, count) in token_counts.items()}


class TfIdfIndex:

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.tmp_dir = Path("../db/tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = Path("../db/tf_idf/index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("../db/tf_idf/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.index = [(0, None) for _ in range(_NUM_BUCKETS)]
        for bucket_id in range(_NUM_BUCKETS):
            path = str(self.index_dir / f'{bucket_id:05d}.index')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'rb', bucket) as f:
                data = pickle.load(f)
                self.index[bucket_id] = {k: (v[0][2], [x[:2] for x in v]) for k, v in data.items()}

    def values_of(self, token_id: int) -> List[Tuple[int, float]]:
        result = []
        with closing(MultiFileReader(self.data_dir, self.bucket_name)) as reader:
            size, locs = self.index[token_id % _NUM_BUCKETS][token_id]
            b = reader.read(locs, size * 8)
            for i in range(size):
                doc_id = int.from_bytes(b[i * 8:i * 8 + 4], 'big')
                tf = struct.unpack("f", b[i * 8 + 4:(i + 1) * 8])[0]
                result.append((doc_id, tf))
        return result

    def _idf(self, token_idx: int, token_indexer: TokenIndexer) -> float:
        idf = (token_indexer.n_docs + 1) / (token_indexer.doc_amount_of(token_idx) + 1)
        return math.log(idf)

    def import_data(self, path: Path, token_indexer: TokenIndexer):
        msg = f"Reading file '{path.name}'..."
        tf_idf = defaultdict(list)
        for record in parquet_file_iterator(str(path.absolute()), ["id", "text"], 1000, msg):
            doc_id = record["id"]
            tfs = _process_record(record, token_indexer)
            for token_idx, tf in tfs.items():
                tf_idf[token_idx].append((doc_id, tf * self._idf(token_idx, token_indexer)))
        batches = [defaultdict(list) for _ in range(1024)]
        for token_idx, values in tqdm(tf_idf.items(), desc="Splitting to chunks..."):
            batches[token_idx % _NUM_BUCKETS][token_idx] = values
        for i, batch in tqdm(enumerate(batches), total=len(batches), desc="Writing tf-idf"):
            with (self.tmp_dir / f"{i:05d}.dat").open("ba") as fp:
                for token_idx, values in batch.items():
                    data = []
                    for doc_id, value in values:
                        data.append(token_idx.to_bytes(4, 'big'))
                        data.append(doc_id.to_bytes(4, 'big'))
                        data.append(bytearray(struct.pack("f", value)))
                    fp.write(b''.join(data))

    def _commit_file(self, file: Path, bucket_name: Optional[str] = None):
        tf_idf = defaultdict(list)
        with file.open("rb") as fp:
            data = fp.read()
            for offset in range(0, len(data), 12):
                token_idx = int.from_bytes(data[offset:offset + 4], 'big')
                doc_id = int.from_bytes(data[offset + 4:offset + 8], 'big')
                value = struct.unpack("f", data[offset + 8:offset + 12])[0]
                tf_idf[token_idx].append((doc_id, value))
        bucket_id = file.name.split(".")[0]
        token_index = defaultdict(list)
        with closing(MultiFileWriter(str(self.data_dir.absolute()), bucket_id, bucket_name)) as writer:
            for token_idx, values in tf_idf.items():
                # convert to bytes
                data = []
                for doc_id, value in values:
                    data.append(doc_id.to_bytes(4, 'big'))
                    data.append(bytearray(struct.pack("f", value)))
                # write to file(s)
                locs = writer.write(b''.join(data))
                locs = [x + (len(values),) for x in locs]
                # save file locations to index
                token_index[token_idx].extend(locs)
            path = str(self.index_dir / f'{bucket_id}.index')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(token_index, f)

    def commit(self):
        for file in tqdm(list(self.tmp_dir.iterdir()), desc="Committing tf-idf"):
            self._commit_file(file)


if __name__ == '__main__':
    # token_indexer = TokenIndexer()
    # file = Path('../wikidump')

    ig = TfIdfIndex()
    # for f in file.iterdir():
    #     ig.import_data(f, token_indexer)
    ig.commit()
    print("loaded")
