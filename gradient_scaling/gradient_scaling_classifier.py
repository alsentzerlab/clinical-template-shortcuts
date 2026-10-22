"""Multitask readmission classifier with a gradient-scaled template head (ShorT).

Follows Brown et al. 2023 (Nat. Commun. 14:4314, "Detecting shortcut learning for
fair medical AI using shortcut testing"): a shared feature extractor feeds a clinical
head (readmission) and an attribute head (template A/B). The attribute head is always
trained normally; only the gradient it sends back INTO the shared encoder is
multiplied by `grad_scale`:

    encoder update  ∝  ∇L_readmit + grad_scale * ∇L_template

Sign convention is the paper's: grad_scale > 0 encourages the encoder to represent
the template, grad_scale < 0 is gradient reversal (removes it), and grad_scale = 0
is the single-task baseline (the template head sees detached features).

The readmission path is built exactly as in finetuning/train.py: the same
AutoModelForSequenceClassification (2 labels, cross-entropy, its own `score` head)
wrapped by the same PEFT LoRA config, created in the same order so the seeded
initialisation matches. The template head sits on top, reading the same pooled
hidden state the `score` head reads. So at grad_scale = 0 the readmission model and
its updates are those of a normal finetuning run.
"""

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _GradScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.scale * grad_output, None


class GradientScalingClassifier(nn.Module):
    # Copies how the Trainer treats the finetuning model (transformers 4.57): Qwen3's
    # forward takes **kwargs, so the Trainer does NOT divide the loss by the gradient
    # accumulation steps, and Qwen3ForSequenceClassification ignores the
    # num_items_in_batch it is handed. Gradients are therefore summed (not averaged)
    # over the accumulated notes. Matching this keeps grad_scale = 0 identical to
    # finetuning, including how often gradient clipping kicks in.
    accepts_loss_kwargs = True

    def __init__(
        self,
        model_name: str,
        grad_scale: float,
        pad_token_id: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grad_scale = grad_scale
        self.pad_token_id = pad_token_id

        # --- Same calls, same order as finetuning/train.py ---
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification",
            pad_token_id=pad_token_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
        )
        self.model = get_peft_model(model, peft_config)

        # The score head's input is the final hidden state of every token; keep it so
        # the template head reads the same pooled features.
        self._hidden: Optional[torch.Tensor] = None
        self.model.base_model.model.score.register_forward_hook(self._capture_hidden)

        # Created after LoRA so it doesn't shift the seeded init above.
        self.template_head = nn.Linear(model.config.hidden_size, 1).to(model.device)
        self.bce = nn.BCEWithLogitsLoss()

    def _capture_hidden(self, module, inputs, output):
        self._hidden = inputs[0]

    # The Trainer calls this when TrainingArguments(gradient_checkpointing=True).
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()

    def _pool(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Last non-pad token, same rule as Qwen3ForSequenceClassification."""
        non_pad = (input_ids != self.pad_token_id).to(hidden.device, torch.int32)
        idx = torch.arange(input_ids.shape[-1], device=hidden.device, dtype=torch.int32)
        last = (idx * non_pad).argmax(-1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled features f(X) that both heads (and the probe) see."""
        self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(self._hidden, input_ids).float()
        self._hidden = None
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,  # num_items_in_batch from the Trainer; ignored, as in Qwen3 (see above)
    ):
        # labels: [B, 2] = [readmission, template]
        y = labels[:, 0].long() if labels is not None else None
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
        pooled = self._pool(self._hidden, input_ids).float()
        self._hidden = None
        template_logit = self.template_head(_GradScaleFn.apply(pooled, self.grad_scale)).squeeze(-1)
        # [B, 3] = the 2 readmission logits (as in finetuning) + the template logit
        logits = torch.cat([out.logits.float(), template_logit.unsqueeze(-1)], dim=-1)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = out.loss + self.bce(template_logit, labels[:, 1].float())
        return result
