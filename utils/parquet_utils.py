from typing import List

from tqdm import tqdm
import pyarrow.parquet as pa


def parquet_file_iterator(
        file_name: str,
        columns: List[str] = None,
        batch_size: int = 64,
        descr: str = f"Reading file..."
):
    parquet_file = pa.ParquetFile(file_name)
    n_batches = ((parquet_file.metadata.num_rows - 1) // batch_size) + 1
    for record_batch in tqdm(parquet_file.iter_batches(batch_size=batch_size, columns=columns), total=n_batches,
                             desc=descr):
        for d in record_batch.to_pylist():
            yield d
