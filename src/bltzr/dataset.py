import json
import hashlib
import os.path
from pathlib import Path
import torch
import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import sql
from torch.nn.utils.rnn import pad_sequence 
from dataclasses import dataclass
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

@dataclass
class SqlDatasetConfig:
    db_host: str
    db_user: str
    db_pass: str
    db_name: str
    with_metadata: bool = False
    dataset_table: str = "dataset"
    window_size: int = 8192

class SqlDataset(Dataset):
    def _get_cache_path(self):
        # Create a unique cache file name based on dataset parameters
        cache_key = f"{self.config.dataset_table}_{self.config.window_size}_{self.with_metadata}"
        # Add database name to make sure we don't mix indices from different databases
        cache_key = f"{self.config.db_name}_{cache_key}"
        # Create a hash to keep filename reasonable length
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        # Use a dedicated directory for cache files
        cache_dir = Path.home() / ".cache" / "dataset_indices"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"dataset_index_{cache_hash}.json"

    def _save_index(self):
        cache_path = self._get_cache_path()
        print(f"Saving dataset index to {cache_path}")
        # Convert chunks to a serializable format
        serializable_chunks = []
        for chunk in self.chunks:
            # Create a new dict with all fields except any that might not be JSON serializable
            serializable_chunk = {
                "src": chunk["src"],
                "last": chunk.get("last", False)
            }
            serializable_chunks.append(serializable_chunk)

        with open(cache_path, 'w') as f:
            json.dump(serializable_chunks, f)

    def _load_index(self):
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        print(f"Loading dataset index from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                self.chunks = json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load cache file: {e}")
            return False

    def get_conn(self):
        return self.pool.getconn()

    def release_conn(self, conn):
        self.pool.putconn(conn)

    def __init__(self, config):
        super(SqlDataset, self).__init__()
        self.config = config
        self.tokenizer = Tokenizer()
        self.chunks = []

        # Create a connection pool
        self.pool = SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            dsn="dbname={} user={} password={} host={}".format(config.db_name, config.db_user, config.db_pass, config.db_host)
        )
        # Try to load the index from cache first
        if self._load_index():
            return
        conn = self.get_conn()
        print("Building dataset index, hold on...")
        with conn.cursor() as cur:
             # Create a temporary table to store lengths
            cur.execute("""
              CREATE TEMP TABLE item_lengths AS
              SELECT
                  d.tbl,
                  d.ref_id,
                  get_dataset_item_len(d.tbl, d.ref_id, %s) as length
              FROM {} d
            """.format(config.dataset_table), (self.config.with_metadata,))
            # Fetch all lengths at once
            cur.execute("SELECT tbl, ref_id, length FROM item_lengths ORDER BY tbl, ref_id")
            chunk = {"src": []}
            payload_added = 0

            for tbl, ref_id, txt_len in cur.fetchall():
                remaining_len = txt_len
                ofs = 0
                while remaining_len > 0:
                    payload_len = min(self.config.window_size - payload_added, remaining_len)
                    chunk["src"].append({ "tbl": tbl, "ref": ref_id, "ofs": ofs, "len": payload_len })
                    payload_added += payload_len
                    remaining_len -= payload_len
                    ofs += payload_len

                    if payload_added >= self.config.window_size:
                        self.chunks.append(chunk)
                        chunk = {"src": []}
                        payload_added = 0

            if payload_added > 0:
                chunk["last"] = True
                self.chunks.append(chunk)
            # Clean up
            cur.execute("DROP TABLE item_lengths")
        print("Done!")
        self.release_conn(conn)
        # Save the index to cache
        self._save_index()

    def __getitem__(self, i):
        chunk = self.chunks[i]
        conn = self.get_conn()
        chunk_data = []
        with conn.cursor() as cur:
            for idx, s in enumerate(chunk['src']):
                tbl = s['tbl']
                ref = s['ref']
                cur.execute(f"SELECT * FROM get_dataset_item('{tbl}', '{ref}', {self.config.with_metadata})")
                msgs = cur.fetchone()[0]
                chunk_data.extend(self.tokenizer.encode(msgs)[s['ofs']:s['ofs']+s['len']])
        self.release_conn(conn)

        input_ids = torch.tensor(chunk_data, dtype=torch.long)
        labels = torch.cat([input_ids[1:], torch.tensor([-100])])
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.chunks)

@dataclass
class DataCollatorForSqlDataset(object):

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, instances):

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.pad_token_id),
            'labels': labels,
        }

class SqlDataModule():
    def __init__(self, config: SqlDatasetConfig):
        self.dataset = SqlDataset(config)
        tokenizer = Tokenizer()
        self.data_collator = DataCollatorForSqlDataset(tokenizer.get_token_id('<PAD>'))
