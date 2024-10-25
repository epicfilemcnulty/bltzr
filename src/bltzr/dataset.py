import json
import hashlib
import os.path
from pathlib import Path
import torch
from tqdm import tqdm
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
    cache_save_percent: int = 25
    batch_size: int = 100

class SqlDataset(Dataset):

    def _get_cache_path(self):
        cache_key = f"{self.config.db_name}_{self.config.dataset_table}_{self.config.window_size}_{self.config.with_metadata}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = Path.home() / ".cache" / "dataset_indices"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"dataset_index_{cache_hash}.json"

    def _save_index(self, complete=True):
        cache_path = self._get_cache_path()
        if complete:
            print(f"Saving dataset index to {cache_path}")

        serializable_chunks = []
        for i, chunk in enumerate(self.chunks):
            serializable_chunk = {
                "src": chunk["src"]
            }
            # Only add 'last' field to the final chunk
            if i == len(self.chunks) - 1:
                serializable_chunk["last"] = chunk.get("last", False)

            serializable_chunks.append(serializable_chunk)

        cache_data = {
            "chunks": serializable_chunks,
            "complete": complete,
            "processed_rows": getattr(self, 'processed_rows', 0)
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

    def _load_index(self):
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False, 0

        print(f"Loading dataset index from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            self.chunks = cache_data["chunks"]
            is_complete = cache_data.get("complete", False)
            processed_rows = cache_data.get("processed_rows", 0)

            if is_complete:
                print("Loaded complete index from cache")
                return True, processed_rows
            else:
                print(f"Loaded partial index from cache (processed {processed_rows} rows)")
                return False, processed_rows

        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load cache file: {e}")
            return False, 0

    def get_conn(self):
        return self.pool.getconn()

    def release_conn(self, conn):
        self.pool.putconn(conn)

    def __init__(self, config):
        super(SqlDataset, self).__init__()
        self.config = config
        self.tokenizer = Tokenizer()
        self.chunks = []

        self.pool = SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            dsn="dbname={} user={} password={} host={}".format(
                config.db_name, config.db_user, config.db_pass, config.db_host
            )
        )
        # Try to load the index from cache
        is_complete, processed_rows = self._load_index()
        if is_complete:
            return

        conn = self.get_conn()
        print("Building dataset index, hold on...")

        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT tbl, ref_id FROM {}").format(
                    sql.Identifier(config.dataset_table)
                ))
                rows = cur.fetchall()
                total_rows = len(rows)

                # Skip already processed rows
                rows = rows[processed_rows:]
              
                # Initialize or continue with last chunk
                chunk = {"src": []} if not self.chunks else self.chunks[-1]
                payload_added = sum(s['len'] for s in chunk["src"]) if chunk["src"] else 0

                last_save_percentage = (processed_rows / total_rows) * 100 if total_rows > 0 else 0                

                for i in tqdm(range(0, len(rows), self.config.batch_size)):
                    batch_rows = rows[i:i + self.config.batch_size]
                    tbls = [row[0] for row in batch_rows]
                    ref_ids = [row[1] for row in batch_rows]
                    
                    # Create arrays for the PL/Lua function
                    query = f"SELECT * FROM get_dataset_items_len(ARRAY[{','.join(f"""'{tbl}'""" for tbl in tbls)}]::text[], ARRAY[{','.join(str(ref_id) for ref_id in ref_ids)}]::bigint[], {self.config.with_metadata})"
                    cur.execute(query)
                    lengths = cur.fetchone()[0]

                    # Process each item in the batch
                    for j, txt_len in enumerate(lengths):
                        remaining_len = txt_len
                        ofs = 0
                        while remaining_len > 0:
                            payload_len = min(
                                self.config.window_size - payload_added,
                                remaining_len
                            )
                            chunk["src"].append({
                                "tbl": batch_rows[j][0],
                                "ref": batch_rows[j][1],
                                "ofs": ofs,
                                "len": payload_len
                            })
                            payload_added += payload_len
                            remaining_len -= payload_len
                            ofs += payload_len

                            if payload_added >= self.config.window_size:
                                self.chunks.append(chunk)
                                chunk = {"src": []}
                                payload_added = 0

                        # Update processed rows count
                        self.processed_rows = processed_rows + i + 1

                        # Save progress every `cache_save_percent`
                        current_percentage = (self.processed_rows / total_rows) * 100
                        if current_percentage - last_save_percentage >= self.config.cache_save_percent:
                            self._save_index(complete=False)
                            last_save_percentage = current_percentage
                if payload_added > 0:
                    chunk["last"] = True
                    self.chunks.append(chunk)
                    
        finally:
            self.release_conn(conn)
        print("Done!")
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
