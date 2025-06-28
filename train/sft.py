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


def create_custom_loss_function(config):
    """
    Factory function to create custom loss functions for use with SFTTrainer's compute_loss_func parameter
    """
    def custom_loss_func(logits, labels, num_items_in_batch=None):
        """
        Custom loss function that will be called by SFTTrainer's compute_loss method
        Note: logits and labels are already shifted and flattened by the trainer
        Args:
            logits: Model logits (already shifted and flattened)
            labels: Target labels (already shifted and flattened)
            num_items_in_batch: Number of items in the batch (for potential scaling)
        """
        if not config.use_custom_loss:
            # Return None to use default cross-entropy loss
            return None
        
        # Debug logging
        logging.info(f"Custom loss function called with logits shape: {logits.shape}, labels shape: {labels.shape}")
        
        # Check for valid tokens (labels != -100)
        valid_mask = labels != -100
        total_tokens = len(labels)
        valid_tokens = valid_mask.sum().item()
        logging.info(f"Total tokens: {total_tokens}, Valid tokens: {valid_tokens}, Valid ratio: {valid_tokens/total_tokens:.3f}")
        
        if not valid_mask.any():
            logging.warning("No valid tokens found! Returning zero loss.")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Filter to only valid tokens
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        if config.loss_type == "focal":
            return _focal_loss(valid_logits, valid_labels, config)
        elif config.loss_type == "label_smoothing":
            return _label_smoothing_loss(valid_logits, valid_labels, config)
        elif config.loss_type == "topk_cross_entropy":
            return _topk_cross_entropy_loss(valid_logits, valid_labels, config)
        else:
            # Default cross entropy
            return F.cross_entropy(valid_logits, valid_labels)
    
    return custom_loss_func

def _focal_loss(logits, labels, config):
    """Focal Loss implementation for handling class imbalance"""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = config.focal_alpha * (1 - pt) ** config.focal_gamma * ce_loss
    return focal_loss.mean()

def _label_smoothing_loss(logits, labels, config):
    """Label smoothing loss implementation"""
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = F.nll_loss(log_probs, labels, reduction='none')
    
    # Apply label smoothing
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1 - config.label_smoothing) * nll_loss + config.label_smoothing * smooth_loss
    return loss.mean()

def _topk_cross_entropy_loss(logits, labels, config):
    """
    Top-k cross-entropy loss: rescale probabilities so only top-k predictions have mass
    """
    # Apply temperature scaling if specified
    scaled_logits = logits / config.topk_temperature
    
    # Ensure k doesn't exceed vocabulary size
    vocab_size = scaled_logits.size(-1)
    k = min(config.topk_k, vocab_size)
    
    # Debug logging
    logging.info(f"Top-k loss: vocab_size={vocab_size}, k={k}, batch_size={logits.size(0)}")
    
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
    
    # Create a mask for top-k elements
    topk_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)
    
    # Set non-top-k logits to very negative values (effectively zero probability)
    masked_logits = scaled_logits.clone()
    masked_logits[~topk_mask] = float('-inf')
    
    # Compute cross-entropy with the masked logits
    # The softmax will automatically renormalize so top-k probabilities sum to 1
    loss = F.cross_entropy(masked_logits, labels)
    logging.info(f"Top-k loss value: {loss.item()}")
    return loss

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
    
    # Create custom loss function if enabled
    custom_loss_func = None
    if config.use_custom_loss:
        custom_loss_func = create_custom_loss_function(config)
    
    # Create trainer with optional custom loss function
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
        compute_loss_func=custom_loss_func  # This is the key parameter!
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
