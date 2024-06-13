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
            chunk = {"src": []}
            payload_added = 0
            for row in rows:
                cur.execute(sql.SQL("SELECT octet_length(content) FROM {} WHERE id = %s").format(sql.Identifier(row[0])), (row[1],))
                txt_len = cur.fetchone()[0]
                remaining_len = txt_len
                ofs = 0
                while remaining_len > 0:
                    payload_len = min(self.config.window_size - payload_added, remaining_len)
                    chunk["src"].append({"tbl": row[0], "ref": row[1], "ofs": ofs, "len": payload_len})
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
        print("Done!")
        self.release_conn(conn)

    def __getitem__(self, i):
        chunk = self.chunks[i]
        conn = self.get_conn()
        chunk_data = []
        with conn.cursor() as cur:
            for idx, s in enumerate(chunk['src']):
                cur.execute(sql.SQL("SELECT content FROM {} WHERE id = %s").format(sql.Identifier(s['tbl'])), (s['ref'],))
                txt = cur.fetchone()[0]
                if idx == 0 and s['ofs'] == 0:
                    chunk_data.append(self.tokenizer.get_token_id('<TXT>'))
                elif s['ofs'] == 0:
                    chunk_data.append(self.tokenizer.get_token_id('</TXT>'))
                    chunk_data.append(self.tokenizer.get_token_id('<TXT>'))
                chunk_data.extend(self.tokenizer.encode_simple(txt)[s['ofs']:s['ofs']+s['len']])
                if idx == len(chunk['src']) - 1 and s['ofs'] + s['len'] == len(self.tokenizer.encode_simple(txt)):
                    chunk_data.append(self.tokenizer.get_token_id('</TXT>'))
        self.release_conn(conn)
        input_ids = chunk_data

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.cat([input_ids_tensor[1:], torch.tensor([-100])])
        return dict(input_ids=input_ids_tensor, labels=labels_tensor)

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
