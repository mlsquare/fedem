import torch
from transformers import Trainer


class MambaTrainer(Trainer):
    """
    Trainer subclass for training Mamba models.

    Inherits from transformers.Trainer.

    Args:
        Trainer: Parent class for training transformers models.

    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for Mamba model training.

        Args:
            model: Mamba model.
            inputs: Model inputs.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False.

        Returns:
            torch.Tensor: Computed language modeling loss.
            
        """
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids)[0]
        labels = input_ids.to(lm_logits.device)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss
