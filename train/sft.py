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
    """Custom SFT Trainer with configurable loss functions, based on the original compute_loss method"""
    
    def __init__(self, loss_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        Based on the original SFTTrainer compute_loss method.
        """
        # Original logic: Handle labels and compute_loss_func
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        # Original logic: Handle model loss kwargs
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
            
        # Original logic: Get model outputs
        outputs = model(**inputs)
        
        # Original logic: Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            
            # Check for PEFT model (from original)
            try:
                from trl.trainer.sft_trainer import _is_peft_model
                is_peft = _is_peft_model(unwrapped_model)
            except ImportError:
                # Fallback for different TRL versions
                is_peft = hasattr(unwrapped_model, 'base_model')
                
            if is_peft:
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
                
            # CUSTOM MODIFICATION: Check if we should use custom loss
            if self.loss_config.use_custom_loss:
                loss = self._compute_custom_loss(outputs, labels, num_items_in_batch)
            # Original logic: User-defined compute_loss function
            elif self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            # Original logic: Use appropriate loss based on model type
            else:
                try:
                    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
                except ImportError:
                    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {}
                    
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
        else:
            # Original logic: Handle case where no labels
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Original logic: Handle token averaging across devices
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    def _compute_custom_loss(self, outputs, labels, num_items_in_batch=None):
        """
        Compute custom loss based on configuration.
        This mimics what the label_smoother would do but with custom loss functions.
        """
        try:
            # Extract logits from outputs
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            else:
                logits = outputs[0] if hasattr(outputs, '__getitem__') else outputs.logits
            
            if logits is None:
                raise ValueError("Could not extract logits from model outputs")
            
            # Shift labels and logits for causal language modeling (shift_labels=True behavior)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Only compute loss on non-ignored tokens (labels != -100)
            valid_mask = shift_labels != -100
            
            if not valid_mask.any():
                logging.warning("No valid tokens found! Returning zero loss.")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Filter to only valid tokens
            valid_logits = shift_logits[valid_mask]
            valid_labels = shift_labels[valid_mask]
            
            # Apply the specific loss function
            if self.loss_config.loss_type == "focal":
                return self._focal_loss(valid_logits, valid_labels)
            elif self.loss_config.loss_type == "label_smoothing":
                return self._label_smoothing_loss(valid_logits, valid_labels)
            elif self.loss_config.loss_type == "topk_cross_entropy":
                return self._topk_cross_entropy_loss(valid_logits, valid_labels)
            else:
                # Default cross entropy
                return F.cross_entropy(valid_logits, valid_labels)
                
        except Exception as e:
            logging.error(f"Error in custom loss computation: {e}")
            # Fallback to default loss computation using label_smoother
            return self.label_smoother(outputs, labels, shift_labels=True)
    
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
        try:
            # Apply temperature scaling if specified
            scaled_logits = logits / self.loss_config.topk_temperature
            
            # Ensure k doesn't exceed vocabulary size
            vocab_size = scaled_logits.size(-1)
            k = min(self.loss_config.topk_k, vocab_size)
            
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
            
            # Create a mask for top-k elements
            topk_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            
            # Set non-top-k logits to very negative values (effectively zero probability)
            masked_logits = scaled_logits.clone()
            masked_logits[~topk_mask] = -1e9  # Use -1e9 instead of -inf for numerical stability
            
            # Compute cross-entropy with the masked logits
            loss = F.cross_entropy(masked_logits, labels)
            
            return loss
            
        except Exception as e:
            logging.error(f"Error in top-k loss: {e}, falling back to regular cross-entropy")
            return F.cross_entropy(logits, labels)


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # Check TRL version for compatibility
    try:
        trl_version = trl.__version__
        logging.info(f"Using TRL version: {trl_version}")
    except:
        logging.warning("Could not determine TRL version")

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

    # Load dataset with error handling
    try:
        dataset = load_dataset(config.train_file_path)
        logging.info(f"Loaded dataset with keys: {dataset.keys()}")
        logging.info(f"Train dataset size: {len(dataset['train'])}")
        if 'test' in dataset:
            logging.info(f"Test dataset size: {len(dataset['test'])}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

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
    
    # Set training arguments (avoid duplicate assignments)
    if not hasattr(args, 'dataset_text_field') or args.dataset_text_field is None:
        args.dataset_text_field = 'text'
    if not hasattr(args, 'max_seq_length') or args.max_seq_length is None:
        args.max_seq_length = config.block_size
    
    # Create trainer - use CustomSFTTrainer if custom loss is enabled
    if config.use_custom_loss:
        trainer = CustomSFTTrainer(
            loss_config=config,
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset  else dataset['train'],
            args=args,
            data_collator=collator,
        )
    else:
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator,
        )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
