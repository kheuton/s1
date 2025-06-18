import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch
import torch.nn.functional as F

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="Qwen2.5-1.5B-Instruct-s1-top128")
    wandb_entity: Optional[str] = field(default="wandb_kheuton")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    # Add custom loss configuration
    use_custom_loss: bool = field(default=False)
    loss_type: str = field(default="cross_entropy")  # Options: "cross_entropy", "focal", "label_smoothing", "topk_cross_entropy", etc.
    focal_alpha: float = field(default=1.0)
    focal_gamma: float = field(default=2.0)
    label_smoothing: float = field(default=0.1)
    # Top-k parameters
    topk_k: int = field(default=50)  # Number of top predictions to keep
    topk_temperature: float = field(default=1.0)  # Temperature for softmax before top-k filtering

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


class CustomSFTTrainer(trl.SFTTrainer):
    """Custom SFT Trainer with configurable loss functions"""
    
    def __init__(self, loss_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation with different loss functions
        """
        if hasattr(inputs, "pop"):
            labels = inputs.pop("labels")
        else:
            labels = inputs["labels"]
            
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if not self.loss_config.use_custom_loss:
            # Use default loss computation
            return super().compute_loss(model, {**inputs, "labels": labels}, return_outputs, num_items_in_batch)
        
        # Custom loss computation
        loss = self._compute_custom_loss(logits, labels)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _compute_custom_loss(self, logits, labels):
        """Compute custom loss based on configuration"""
        # Shift labels and logits for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Only compute loss on non-ignored tokens (labels != -100)
        valid_mask = shift_labels != -100
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask]
        
        if self.loss_config.loss_type == "focal":
            return self._focal_loss(valid_logits, valid_labels)
        elif self.loss_config.loss_type == "label_smoothing":
            return self._label_smoothing_loss(valid_logits, valid_labels)
        elif self.loss_config.loss_type == "topk_cross_entropy":
            return self._topk_cross_entropy_loss(valid_logits, valid_labels)
        else:
            # Default cross entropy
            return F.cross_entropy(valid_logits, valid_labels)
    
    def _focal_loss(self, logits, labels):
        """Focal Loss implementation for handling class imbalance"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.loss_config.focal_alpha * (1 - pt) ** self.loss_config.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def _label_smoothing_loss(self, logits, labels):
        """Label smoothing loss implementation"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction='none')
        
        # Apply label smoothing
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.loss_config.label_smoothing) * nll_loss + self.loss_config.label_smoothing * smooth_loss
        return loss.mean()
    
    def _topk_cross_entropy_loss(self, logits, labels):
        """
        Top-k cross-entropy loss: rescale probabilities so only top-k predictions have mass
        """
        # Apply temperature scaling if specified
        scaled_logits = logits / self.loss_config.topk_temperature
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(scaled_logits, k=self.loss_config.topk_k, dim=-1)
        
        # Create a mask for top-k elements
        topk_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_indices, True)
        
        # Set non-top-k logits to very negative values (effectively zero probability)
        masked_logits = scaled_logits.clone()
        masked_logits[~topk_mask] = float('-inf')
        
        # Compute cross-entropy with the masked logits
        # The softmax will automatically renormalize so top-k probabilities sum to 1
        return F.cross_entropy(masked_logits, labels)
    
    def _weighted_cross_entropy_loss(self, logits, labels):
        """Weighted cross-entropy loss implementation"""
        # Assuming weights are provided in the labels tensor for simplicity
        # In practice, you might want to pass a separate weights tensor
        weights = labels.new_ones(labels.size())
        return F.cross_entropy(logits, labels, weight=weights)

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    # Use custom trainer if custom loss is enabled
    if config.use_custom_loss:
        trainer = CustomSFTTrainer(
            loss_config=config,
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator
        )
    else:
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator
        )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
