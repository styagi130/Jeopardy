from torch.utils import data
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class JeopardyDataset(data.Dataset):
    """
        A Class to sample jeopardy dataset
    """
    def __init__(self, df, max_seq_length=256, device="cuda"):
        self.df = df
        self.max_seq_length = max_seq_length
        self.device = device
    
    def __getitem__(self,idx):
        tokenized_text = tokenizer.tokenize(self.df.loc[:, "text_feats"].values[idx])
        ### Tokenize input text
        if len(tokenized_text) > self.max_seq_length:
            tokenized_text = tokenized_text[:self.max_seq_length]
        padding = ["pad"] * (self.max_seq_length - len(tokenized_text))
        tokenized_text += padding

        ids_text  = tokenizer.convert_tokens_to_ids(tokenized_text)
        ### Pad Sequence and generate mask 
        attn_mask = [0 if ids==0 else 1 for ids in ids_text]

        assert len(ids_text) == self.max_seq_length
        ids_text = torch.tensor(ids_text).to(self.device)
        attn_mask = torch.tensor(attn_mask).to(self.device)

        air_date = torch.tensor(self.df.loc[:, " Air Date"].values[idx]).to(self.device)
        round_ = torch.tensor(self.df.loc[:, " Round"].values[idx]).to(self.device)
        value = torch.tensor(self.df.loc[:, " Value"].values[idx]).to(self.device)
        return ids_text, attn_mask, air_date, round_, value
    
    def __len__(self):
        return len(self.df)