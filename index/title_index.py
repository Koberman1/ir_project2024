import pickle
from pathlib import Path
from typing import Optional

from utils.parquet_utils import parquet_file_iterator
from utils.file_utils import get_bucket, _open


class TitleIndex:

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.title_index_file = Path("../db/titles.index")
        self.titles = dict()
        bucket = None if self.bucket_name is None else get_bucket(self.bucket_name)
        with _open(str(self.title_index_file.absolute()), 'rb', bucket) as f:
            self.titles = pickle.load(f)

    def import_data(self, path: Path):
        msg = f"Reading file '{path.name}'..."
        for record in parquet_file_iterator(str(path.absolute()), ["id", "title", "text"], 1000, msg):
            self.titles[record['id']] = (record['title'], len(record['text'].split()))

    def title_of(self, doc_id: int) -> Optional[str]:
        if self.titles.get(doc_id, None) is None:
            return None
        else:
            return self.titles.get(doc_id)[0]

    def length_of(self, doc_id: int) -> Optional[int]:
        if self.titles.get(doc_id, None) is None:
            return None
        else:
            return self.titles.get(doc_id)[1]

    def commit(self):
        bucket = None if self.bucket_name is None else get_bucket(self.bucket_name)
        with _open(str(self.title_index_file.absolute()), 'wb', bucket) as f:
            pickle.dump(self.titles, f)


if __name__ == '__main__':
    file = Path('../wikidump')

    ig = TitleIndex()
    for f in file.iterdir():
        ig.import_data(f)
    ig.commit()
    print("loaded")
