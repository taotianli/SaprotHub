import torchmetrics
import torch
import torch.distributed as dist

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotPairClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels, **kwargs):
        """
        Args:
            num_labels: number of labels

            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        
        hidden_size = self.model.config.hidden_size * 2
        classifier = torch.nn.Sequential(
                        Linear(hidden_size, hidden_size),
                        ReLU(),
                        Linear(hidden_size, self.num_labels)
                    )
        
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs_1, inputs_2):
        if self.freeze_backbone:
            hidden_1 = torch.stack(self.get_hidden_states_from_dict(inputs_1, reduction="mean"))
            hidden_2 = torch.stack(self.get_hidden_states_from_dict(inputs_2, reduction="mean"))
        else:
            # If "esm" is not in the model, use "bert" as the backbone
            backbone = self.model.esm if hasattr(self.model, "esm") else self.model.bert
            
            # 检查输入的token IDs是否在有效范围内
            vocab_size = backbone.embeddings.word_embeddings.num_embeddings
            
            # 处理inputs_1
            input_ids_1 = inputs_1["input_ids"]
            if torch.max(input_ids_1) >= vocab_size:
                print(f"Warning: inputs_1 - Found token IDs exceeding vocabulary size. Max ID: {torch.max(input_ids_1).item()}, Vocab size: {vocab_size}")
                # 将超出范围的ID替换为UNK token ID
                unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                inputs_1["input_ids"] = torch.where(input_ids_1 < vocab_size, input_ids_1, torch.tensor(unk_id).to(input_ids_1.device))
            
            # 处理inputs_2
            input_ids_2 = inputs_2["input_ids"]
            if torch.max(input_ids_2) >= vocab_size:
                print(f"Warning: inputs_2 - Found token IDs exceeding vocabulary size. Max ID: {torch.max(input_ids_2).item()}, Vocab size: {vocab_size}")
                # 将超出范围的ID替换为UNK token ID
                unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                inputs_2["input_ids"] = torch.where(input_ids_2 < vocab_size, input_ids_2, torch.tensor(unk_id).to(input_ids_2.device))
                
            hidden_1 = backbone(**inputs_1)[0][:, 0, :]
            hidden_2 = backbone(**inputs_2)[0][:, 0, :]

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat)
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))


        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))


        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)