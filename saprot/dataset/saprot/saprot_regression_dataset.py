import torch
import json
import random

from ..data_interface import register_dataset
from transformers import AutoTokenizer, EsmTokenizer	
from ..lmdb_dataset import *
from ..lmdb_dataset import *
from utils.others import setup_seed


@register_dataset
class SaprotRegressionDataset(LMDBDataset):
	def __init__(self,
	             tokenizer: str,
	             max_length: int = 1024,
	             min_clip: [float, float] = None,
	             mix_max_norm: [float, float] = None,
				 mask_struc_ratio: float = None,
	             plddt_threshold: float = None,
	             **kwargs):
		"""
		
		Args:
			tokenizer: ESM tokenizer

			max_length: Maximum length of the sequence

			min_clip: [given_value, clip_value]
					  Set the fitness value to a fixed value if it is less than a given value
			
			mix_max_norm: [min_norm, max_norm]
						  Normalize the fitness value to [0, 1] by min-max normalization

			mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
			
			plddt_threshold: If not None, mask structure tokens with pLDDT < threshold

			**kwargs:
		"""
		
		super().__init__(**kwargs)
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
		self.max_length = max_length
		self.min_clip = min_clip
		self.mix_max_norm = mix_max_norm
		self.mask_struc_ratio = mask_struc_ratio
		self.plddt_threshold = plddt_threshold
	
	def __getitem__(self, index):
		entry = json.loads(self._get(index))
		seq = entry['seq'][:self.max_length-2]
		
		# Add spaces between amino acids for non-ESM models
		if not isinstance(self.tokenizer, EsmTokenizer):
			seq = " ".join(seq)
			
		# Mask structure tokens
		if self.mask_struc_ratio is not None:
			tokens = self.tokenizer.tokenize(seq)
			mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]

			# Randomly shuffle the mask candidates and set seed to ensure mask is consistent
			setup_seed(20000812)
			random.shuffle(mask_candi)

			# Mask first n structure tokens
			mask_num = int(len(mask_candi) * self.mask_struc_ratio)
			for i in range(mask_num):
				idx = mask_candi[i]
				tokens[idx] = tokens[idx][:-1] + "#"

			seq = "".join(tokens)
		
		# Mask structure tokens with pLDDT < threshold
		if self.plddt_threshold is not None:
			plddt = entry["plddt"]
			tokens = self.tokenizer.tokenize(seq)
			seq = ""
			for token, score in zip(tokens, plddt):
				if score < self.plddt_threshold:
					seq += token[:-1] + "#"
				else:
					seq += token

		tokens = self.tokenizer.tokenize(seq)[:self.max_length]
		seq = " ".join(tokens)
	
		if self.min_clip is not None:
			given_min, clip_value = self.min_clip
			if entry['fitness'] < given_min:
				entry['fitness'] = clip_value
		
		if self.mix_max_norm is not None:
			min_norm, max_norm = self.mix_max_norm
			entry['fitness'] = (entry['fitness'] - min_norm) / (max_norm - min_norm)
				
		label = entry['fitness']

		return seq, label
	
	def __len__(self):
		return int(self._get("length"))
	
	def collate_fn(self, batch):
		seqs, labels = tuple(zip(*batch))
		labels = torch.tensor(labels)
		labels = {"labels": labels}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}

		# 添加调试信息 - 检查输入的token IDs
		print("=" * 50)
		print("数据集调试信息:")
		print(f"批次大小: {len(seqs)}")
		print(f"序列示例 (前3个): {seqs[:3]}")
		print(f"标签: {labels['labels']}")
		print(f"Token IDs shape: {encoder_info['input_ids'].shape}")
		print(f"Token IDs 范围: {encoder_info['input_ids'].min().item()} - {encoder_info['input_ids'].max().item()}")
		
		# 检查是否有UNK token
		if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
			unk_count = (encoder_info['input_ids'] == self.tokenizer.unk_token_id).sum().item()
			print(f"UNK token 数量: {unk_count}")
		
		# 检查词汇表大小
		vocab_size = len(self.tokenizer.get_vocab())
		print(f"词汇表大小: {vocab_size}")
		
		# 检查是否有超出词汇表范围的token
		invalid_tokens = (encoder_info['input_ids'] >= vocab_size).sum().item()
		print(f"超出词汇表范围的token数量: {invalid_tokens}")
		print("=" * 50)
		
		return inputs, labels