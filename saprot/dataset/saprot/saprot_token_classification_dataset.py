import json
import random
import torch

from ..data_interface import register_dataset
from transformers import AutoTokenizer, EsmTokenizer
from ..lmdb_dataset import *
from data.data_transform import pad_sequences


@register_dataset
class SaprotTokenClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.is_saprot_model = 'saprot' in tokenizer.lower() if isinstance(tokenizer, str) else False

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length-2]

        if self.is_saprot_model:
            processed_seq = []
            for aa in seq:
                processed_seq.append(aa + "#")
            seq = processed_seq
        
        seq = " ".join(seq)

        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)
        
        # Add -1 to the start and end of the label to ignore the cls token
        label = [-1] + entry["label"][:self.max_length-2] + [-1]
        # print('seq and label len', len(seq), len(label))
        label = torch.tensor(label, dtype=torch.long)
        
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))

        label_ids = pad_sequences(label_ids, constant_value=-1)
        labels = {"labels": label_ids}

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}
 
        return inputs, labels