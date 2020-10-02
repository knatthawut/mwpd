import pandas as pd
import re
import torch
from torch.utils.data import Dataset

class ProductCorpus(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = pd.read_json(dataset, lines=True)
        self.dataset['title_left_clean'] = self.dataset.loc[:,'title_left'].map(self._clean_text)
        self.dataset['title_right_clean'] = self.dataset.loc[:,'title_right'].map(self._clean_text)
        self.tokenizer = tokenizer

    def _clean_text(self, text):
        if text == None:
            return 'None'
        text = re.sub(r'\\.', '', text)  # Remove all \n \t etc..
        text = re.sub(r'[^\w\s]*', '', text)  # Remove anything not a digit, letter, or space
        return text.strip().lower()

    def __len__(self):
        return len(self.dataset)

    def getColumn(self,col_name):
        return self.dataset[col_name]

    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataset.iloc[idx, :]

        encoded_dict = self.tokenizer.encode_plus(
            row['title_left_clean'], row['title_right_clean'],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=165,  # Pad and Truncate sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attention masks.
            return_token_type_ids=True,  # token type ids
            return_tensors='pt',  # Return tensors.
            truncation=True
        )

        return {'input': encoded_dict['input_ids'],
                'attn': encoded_dict['attention_mask'],
                'token': encoded_dict['token_type_ids'],
                'label': row['label']}
