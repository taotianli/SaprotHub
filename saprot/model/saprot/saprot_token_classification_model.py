import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotTokenClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)
    
    def compute_mcc(self, preds, target):
        tp = (preds * target).sum()
        tn = ((1 - preds) * (1 - target)).sum()
        fp = (preds * (1 - target)).sum()
        fn = ((1 - preds) * target).sum()
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        return tp, tn, fp, fn, mcc
    
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}
    
    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)
        
        else:
            # 检查输入的token IDs是否在有效范围内
            if hasattr(self.model, "esm"):
                vocab_size = self.model.esm.embeddings.word_embeddings.num_embeddings
                input_ids = inputs["input_ids"]
                if torch.max(input_ids) >= vocab_size:
                    print(f"Warning: Found token IDs exceeding vocabulary size. Max ID: {torch.max(input_ids).item()}, Vocab size: {vocab_size}")
                    # 将超出范围的ID替换为UNK token ID
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
            elif hasattr(self.model, "bert"):
                vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
                input_ids = inputs["input_ids"]
                if torch.max(input_ids) >= vocab_size:
                    print(f"Warning: Found token IDs exceeding vocabulary size. Max ID: {torch.max(input_ids).item()}, Vocab size: {vocab_size}")
                    # 将超出范围的ID替换为UNK token ID
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
            
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        
        # 将logits和label展平为2D和1D
        batch_size, seq_len, num_labels = logits.shape
        logits = logits.view(-1, num_labels)  # [batch_size * seq_len, num_labels]
        label = label.view(-1)  # [batch_size * seq_len]
        
        # 计算损失
        loss = cross_entropy(logits, label, ignore_index=-1)
        
        # 移除被忽略的索引位置
        mask = label != -1
        label = label[mask]
        logits = logits[mask]
        
        # 非训练阶段时保存预测结果
        if stage != "train":
            preds = logits.argmax(dim=-1)
            self.preds.append(preds)
            self.targets.append(label)
        
        # 更新指标
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)
        
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # 重置训练指标
            self.reset_metrics("train")
        
        return loss
    
    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        
        preds = torch.cat(self.preds, dim=-1)
        target = torch.cat(self.targets, dim=-1)
        tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
        
        # Gather results
        # tmp = torch.tensor([tp, tn, fp, fn])
        # tp, tn, fp, fn = self.all_gather(tmp).sum(dim=0)
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        log_dict["test_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")
    
    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        preds = torch.cat(self.preds, dim=-1)
        target = torch.cat(self.targets, dim=-1)
        tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
        
        # Gather results
        # tmp = torch.tensor([tp, tn, fp, fn])
        # tp, tn, fp, fn = self.all_gather(tmp).sum(dim=0)
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        log_dict["valid_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)