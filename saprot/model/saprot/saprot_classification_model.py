import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # If backbone is frozen, the embedding will be the average of all residues
        # print('freeze_backbone:',self.freeze_backbone)
        if self.freeze_backbone:
            # print(inputs)
            print(self.freeze_backbone)
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)

        # For ESM models
        elif hasattr(self.model, "esm"):
            vocab_size = self.model.esm.embeddings.word_embeddings.num_embeddings
            print(f"esm vocab_size: {vocab_size}")
            print('here esm',inputs)
            logits = self.model(**inputs).logits

        elif hasattr(self.model, "bert"):
            vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
            print(f"vocab_size: {vocab_size}")
            # input_ids = inputs["input_ids"]
            # print(f"input_ids: {input_ids}")
            # print(f"inputs类型: {type(inputs)}")
            # print(f"inputs字典键: {inputs.keys()}")
            # print(f"input_ids形状: {inputs['input_ids'].shape}")
            # print(f"input_ids内容: {inputs['input_ids']}")
            # if torch.max(input_ids) >= vocab_size:
            #     unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
            #     inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
                    
            logits = self.model(**inputs).logits
        
        return logits

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