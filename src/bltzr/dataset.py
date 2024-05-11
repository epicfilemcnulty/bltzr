import torch
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import sql
from torch.nn.utils.rnn import pad_sequence 
from dataclasses import dataclass
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

# This is not an optimal implementation -- although
# the actual data lives in Postgres, when we init the class
# we need to calculate the length of the whole dataset, so we iterate all
# the elements in it. (It's probably worth storing the length of `content` fields in the table too)
# And we build a mapping index (as one document can span several chunks (window_size that is)
# and store it in memory.

@dataclass
class SqlDatasetConfig:
    db_host: str
    db_user: str
    db_name: str
    dataset_table: str = "dataset"
    window_size: int = 8192

class SqlDataset(Dataset):

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
            dsn="dbname={} user={} host={}".format(config.db_name, config.db_user, config.db_host)
        )
        conn = self.get_conn()
        print("Calculating dataset length, hold on...")
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT tbl, ref_id FROM {}").format(sql.Identifier(config.dataset_table)))
            rows = cur.fetchall()
            for row in rows:
                if row[0] == 'chats':
                    cur.execute("SELECT len FROM chats WHERE id = %s", (row[1],))
                    chat_raw = cur.fetchone()
                    total_len = int(chat_raw[0])
                    chunks = total_len // self.config.window_size
                    rest_bytes = total_len - chunks * self.config.window_size
                    if rest_bytes > 0:
                        chunks += 1
                    self.chunks.append({"chunks": chunks, "tbl": row[0], "ref": row[1]})
                else:
                    cur.execute(sql.SQL("SELECT octet_length(content) FROM {} WHERE id = %s").format(sql.Identifier(row[0])), (row[1],))
                    txt_len = cur.fetchone()
                    chunks = (txt_len[0] + 2) // self.config.window_size
                    rest_bytes = (txt_len[0] + 2) - chunks * self.config.window_size
                    if rest_bytes > 0:
                        chunks += 1
                    self.chunks.append({"chunks": chunks, "tbl": row[0], "ref": row[1]})
        print("Done!")
        self.release_conn(conn)

    def __len__(self):
        total = 0
        for chunk in self.chunks:
            total += chunk['chunks']
        return total

    def map_idx(self, i):
        accum_chunks = 0
        for idx, chunk in enumerate(self.chunks):
            if accum_chunks + chunk['chunks'] > i:
                return idx, i - accum_chunks
            accum_chunks += chunk['chunks']
        raise ValueError(f"Index {i} out of range")

    def __getitem__(self, i):
        idx, offset = self.map_idx(i)
        chunk = self.chunks[idx]
        conn = self.get_conn()
        with conn.cursor() as cur:
            if chunk['tbl'] == 'chats':
                cur.execute("SELECT chat FROM chats WHERE id = %s", (chunk['ref'],))
                resp = cur.fetchone()
                data = self.tokenizer.encode(resp[0])
            else:
                cur.execute(sql.SQL("SELECT content FROM {} WHERE id = %s").format(sql.Identifier(chunk['tbl'])), (chunk['ref'],))
                txt = cur.fetchone()
                data = self.tokenizer.encode_text({'content':txt[0]})
        self.release_conn(conn)
        start = offset * self.config.window_size
        end = start + self.config.window_size
        input_ids = data[start:end]

        input_ids_tensor = torch.tensor(input_ids[:self.config.window_size], dtype=torch.long)
        # Pad sequence to desired length.
        padding_size = self.config.window_size - len(input_ids_tensor)
        padding_tensor = torch.full((padding_size,), self.tokenizer.get_token_id('<PAD>'))
        padded_input_ids = torch.cat([input_ids_tensor, padding_tensor])

        # Shift labels by one position for language model training
        labels_tensor = torch.cat([padded_input_ids[1:],torch.tensor([-100])])
        return dict(input_ids=padded_input_ids, labels=labels_tensor)

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
