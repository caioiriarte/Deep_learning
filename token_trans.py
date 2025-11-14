import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
import rich
from typing import List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
import tokenizers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    AutoModelForMaskedLM,
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoConfig
)
from torch.utils.data import DataLoader
from script1 import glove_vectors, glove_vocabulary, glove_tokenizer

sns.set_theme()

# ----------------------------------------------------------------------
# 1. SETUP AND MODEL/TOKENIZER MACROS
# ----------------------------------------------------------------------

# --- 1. Choose the Model Architecture ---
# ARCHITECTURE = 1: Base (GloVe-compatible/Random Init)
# ARCHITECTURE = 2: BERT-config (Random Init) -> Defines and trains a fresh BERT
# ARCHITECTURE = 3: GPT-2-config (Random Init) -> Defines and trains a fresh GPT-2
ARCHITECTURE = 1

# --- 2. Choose the Tokenization Scheme ---
# TOKENIZER = 1: GloVe Tokenizer
# TOKENIZER = 2: BERT WordPiece Tokenizer
# TOKENIZER = 3: GPT-2 BPE Tokenizer
TOKENIZER = 1

# Configuration mapping for architectures
ARCH_CONFIGS = {
    # CHANGE: AutoModel (Base) does not output logits. Switch to AutoModelForMaskedLM.
    1: {"name": "Base_Random", 
        "hf_model_name": "bert-base-uncased", 
        "model_class": AutoModelForMaskedLM,
        "random_init": True},
    
    # 2 and 3 remain correct as they already use LM-specific classes
    2: {"name": "BERT_Encoder_Random", "hf_model_name": "bert-base-uncased", "model_class": AutoModelForMaskedLM, "random_init": True},
    3: {"name": "GPT2_Decoder_Random", "hf_model_name": "gpt2", "model_class": AutoModelForCausalLM, "random_init": True},
}

# Configuration mapping for tokenizers (Unchanged)
TOKEN_CONFIGS = {
    1: {"name": "GloVe", "hf_name": None}, # Custom GloVe setup
    2: {"name": "BERT_WP", "hf_name": "bert-base-uncased"},
    3: {"name": "GPT2_BPE", "hf_name": "gpt2"},
}

# Global variables
current_tokenizer = None
current_model = None
current_vocab_size = 0
current_embed_dim = 0
PAD_IDX = 0
# This flag is now effectively always False, but we keep it for consistency
USE_PRETRAINED_WEIGHTS = False 

# --- Conditional Initialization Block ---

arch_settings = ARCH_CONFIGS.get(ARCHITECTURE)
token_settings = TOKEN_CONFIGS.get(TOKENIZER)

if arch_settings is None or token_settings is None:
    raise ValueError("Invalid ARCHITECTURE or TOKENIZER value. Must be 1, 2, or 3.")

arch_name = arch_settings["hf_model_name"]
model_class = arch_settings["model_class"]
token_hf_name = token_settings["hf_name"]

rich.print(f"[bold red]ARCHITECTURE: {arch_settings['name']} ({ARCHITECTURE})[/bold red] | [bold red]TOKENIZER: {token_settings['name']} ({TOKENIZER})[/bold red]")

# 1. TOKENIZER SETUP
if TOKENIZER == 1:
    # GloVe Tokenizer (Custom Vocab)
    current_tokenizer = glove_tokenizer
    # GloVe's vocab size is fixed by the dataset
    glove_vocab_size = glove_vectors.shape[0]
    PAD_IDX = glove_vocabulary.index('<|pad|>') if '<|pad|>' in glove_vocabulary else 0
    rich.print(f"[bold yellow]Using fixed GloVe Vocab Size: {glove_vocab_size}[/bold yellow]")
    current_vocab_size = glove_vocab_size
    
else:
    # BERT or GPT-2 Tokenizer (HF Vocab)
    current_tokenizer = AutoTokenizer.from_pretrained(token_hf_name)
    current_vocab_size = current_tokenizer.vocab_size
    if TOKENIZER == 2: # BERT
        PAD_IDX = current_tokenizer.pad_token_id or current_tokenizer.unk_token_id
    elif TOKENIZER == 3: # GPT-2
        current_tokenizer.pad_token = current_tokenizer.eos_token 
        PAD_IDX = current_tokenizer.pad_token_id

# 2. MODEL SETUP (Weights) - SIMPLIFIED LOGIC
# Since all architectures are now random initialization, we always load from config.

# 1. Load the specific configuration object using AutoConfig
config = AutoConfig.from_pretrained(arch_name) 

# 2. Update the vocabulary size and output flags in the configuration
config.vocab_size = current_vocab_size 
# Ensure these flags are True in the config itself, which is the correct place for them.
config.output_hidden_states = True # This sets the configuration flag
config.output_attentions = True    # This sets the configuration flag

# 3. Instantiate the model from the configuration (random weights)
# REMOVE THE REDUNDANT KEYWORD ARGUMENTS
current_model = model_class.from_config(config) 
rich.print(f"[bold red]{arch_settings['name']}: Using RANDOM weights (vocab size adjusted).[/bold red]")


current_embed_dim = current_model.config.hidden_size
current_model.eval()

# ----------------------------------------------------------------------
# 2. CONSTANTS AND DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 3 
max_dataset_size = 1000
max_seq_size = 50 
rich.print(f"Device: [red]{DEVICE}[/red] | Model Class: [red]{current_model.__class__.__name__}[/red] | Vocab Size: [red]{current_vocab_size}[/red] | Embed Dim: [red]{current_embed_dim}[/red] | PAD IDX: [red]{PAD_IDX}[/red]")

# control verbosity
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------------------

def custom_collate_fn(batch):
    """
    Collate function to extract 'token_ids' from the dataset batch 
    and convert them into a single long tensor.
    """
    token_ids_list = [item['token_ids'] for item in batch]
    return torch.tensor(token_ids_list, dtype=torch.long) 


def batch_tokenize(
    batch: List[Dict[str, Any]], 
    max_length=max_seq_size, 
    tokenizer: Union[tokenizers.Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
    key:str="text"
) -> Dict[str, Any]:
    texts = batch[key]
    
    if isinstance(tokenizer, tokenizers.Tokenizer):
        # Handle original `tokenizers` library output (TOKENIZER=1)
        encodings = tokenizer.encode_batch(texts)
        
        # --- FIX: Manually apply truncation AND padding for GloVe ---
        token_ids_padded = []
        for x in encodings:
            # 1. Truncate
            ids = x.ids[:max_length]
            # 2. Pad
            padding_needed = max_length - len(ids)
            ids.extend([PAD_IDX] * padding_needed)
            token_ids_padded.append(ids)
            
        return {"token_ids": token_ids_padded}
    
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        # Handle Hugging Face `transformers` output (TOKENIZER=2 and 3)
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length", 
            truncation=True,
            return_attention_mask=False
        )
        return {"token_ids": encodings["input_ids"]}
        
    else:
        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

# ----------------------------------------------------------------------
# 4. DATASET PREPARATION
# ----------------------------------------------------------------------

# load AG News, take a subset of `max_dataset_size` rows and tokenize
dataset = datasets.load_dataset("ag_news")
dataset = datasets.DatasetDict({split: dset.select(range(max_dataset_size)) if len(dset) > max_dataset_size else dset for split, dset in dataset.items()})

# Use the global `current_tokenizer`
dataset = dataset.map(
    partial(batch_tokenize, tokenizer=current_tokenizer), 
    batched=True, 
    num_proc=2, 
    batch_size=10,
    load_from_cache_file=False # Crucial for fresh tokenization
)
rich.print(dataset)


# Create the DataLoader for the training split
train_dataloader = torch.utils.data.DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,         # Important for training stability
    collate_fn=custom_collate_fn,
    pin_memory=True,      # Speeds up data transfer to GPU
)

# ----------------------------------------------------------------------
# 5. MODEL INSTANTIATION, LOSS, AND OPTIMIZER
# ----------------------------------------------------------------------

# Send it to the device
current_model = current_model.to(DEVICE)

# 1. Define the Loss Criterion
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion = criterion.to(DEVICE)

# 2. Define the Optimizer
optimizer = torch.optim.Adam(current_model.parameters(), lr=1e-5) 

# ----------------------------------------------------------------------
# BLOCK 6: THE FINE-TUNING LOOP
# ----------------------------------------------------------------------
rich.print("[bold green]STARTING FINE-TUNING...[/bold green]")
# Set model to training mode
current_model.train()

for epoch in range(NUM_EPOCHS):
    rich.print(f"[bold blue]-- Epoch {epoch+1}/{NUM_EPOCHS} --[/bold blue]")
    
    total_loss = 0
    
    for step, token_ids_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        
        token_ids_batch = token_ids_batch.to(DEVICE) 
        
        # 1. Zero the gradients 
        optimizer.zero_grad() 

        # 2. Forward Pass
        outputs = current_model(token_ids_batch)
        logits = outputs.logits
        
        # 3. Prepare Logits (Causal/Next-Token Objective)
        # We apply the Causal Language Modeling (CLM) shift:
        # Input t predicts Target t+1
        logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = token_ids_batch[:, 1:].reshape(-1)
        
        # 4. Calculate Loss
        loss = criterion(logits_flat, targets)
        total_loss += loss.item()

        # 5. Backward Pass and Optimization
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_dataloader)
    rich.print(f"[bold green]Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}[/bold green]")

# Set model back to evaluation for visualization blocks
current_model.eval()

# ----------------------------------------------------------------------
# BLOCK 7: GRADIENT VISUALIZATION TEST
# ----------------------------------------------------------------------

rich.print("[bold yellow]STARTING GRADIENT VISUALIZATION TEST...[/bold yellow]")

# Reset gradients 
current_model.zero_grad()

# 1. Get DUMMY token ids
T_SIZE = 10
# **Do NOT set requires_grad here. Keep it as a discrete long tensor.**
token_ids_test = torch.randint(low=0, high=current_vocab_size, size=(T_SIZE, T_SIZE))
token_ids_test = token_ids_test.to(DEVICE)
# token_ids_test.requires_grad_(True) # <--- REMOVE THIS LINE

# Get the internal Embedding layer (Model-specific access)
wte = None
if hasattr(current_model, 'transformer') and hasattr(current_model.transformer, 'wte'):
    wte = current_model.transformer.wte # GPT2 access
elif hasattr(current_model, 'bert') and hasattr(current_model.bert.embeddings, 'word_embeddings'):
    wte = current_model.bert.embeddings.word_embeddings # BERT access
    
if wte is not None:
    # 2. Run forward pass and manually retrieve embeddings
    
    # The `wte` layer converts the long tensor (IDs) to a float tensor (Embeddings)
    input_embeddings = wte(token_ids_test) 
    
    # Now, set requires_grad and retain_grad on the continuous embedding tensor
    input_embeddings.requires_grad_(True) 
    input_embeddings.retain_grad()
    
    # Run the rest of the model from the embeddings
    outputs = current_model(inputs_embeds=input_embeddings)
    logits = outputs.logits
    
    # 3. Compute the specific loss for the visualization test
    loss_locations = torch.arange(0, T_SIZE)[:, None, None].expand(T_SIZE, 1, logits.shape[-1])
    loss_locations = loss_locations.to(DEVICE)
    loss_test = logits.gather(index=loss_locations, dim=1).mean()
    
    # 4. Backward pass to retrieve the gradients
    loss_test.backward()
    
    # The magnitude is the L2 norm of the gradient w.r.t. the input embeddings
    grad_magnitude = input_embeddings.grad.norm(dim=2)
    
    # 5. Visualize the gradient
    grad_magnitude[grad_magnitude==0] = -math.inf 
    grad_magnitude = grad_magnitude.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
    plt.colorbar()
    plt.grid(False)
    plt.xlabel("$t$ (input)")
    plt.ylabel("$t'$ (loss)")
    plt.title(f"Gradient Magnitude (Arch: {arch_settings['name']}, Token: {token_settings['name']})")
    plt.show()
else:
    rich.print("[bold red]ERROR:[/bold red] Could not reliably access the Word Token Embedding (WTE) layer for gradient test.")

# ----------------------------------------------------------------------
# BLOCK 8: ATTENTION MAP VISUALIZATION
# ----------------------------------------------------------------------
rich.print("[bold yellow]STARTING ATTENTION MAP VISUALIZATION (Transformer)...[/bold yellow]")

# Helper function to convert IDs back to tokens (handles all tokenizers)
def get_tokens_from_ids(token_ids, tokenizer):
    if isinstance(tokenizer, tokenizers.Tokenizer):
        if hasattr(tokenizer, 'get_vocabulary'):
            vocab = tokenizer.get_vocabulary()
            return [vocab[i] for i in token_ids]
        else:
            return [str(i) for i in token_ids] 
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)): 
        return tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return [str(i) for i in token_ids]

# get a more natural sentence
sentence = "Masked attention allows implementing dependency constraints between inputs and outputs."

# Tokenize the sentence 
token_ids = torch.tensor(batch_tokenize({"text": [sentence]}, max_length=50, tokenizer=current_tokenizer)['token_ids'][0])
tokens = get_tokens_from_ids(token_ids, current_tokenizer)

# Pass the input to the model to get attention weights
input_ids_tensor = token_ids.unsqueeze(0).to(DEVICE)

# Run the forward pass, ensuring attention is outputted
outputs = current_model(input_ids_tensor, output_attentions=True)

if hasattr(outputs, 'attentions') and outputs.attentions is not None:
    attention_layers = outputs.attentions
    
    # Extract the attention map from the last layer and average across all heads for a cleaner visualization
    # Shape: (1, num_heads, sequence_length, sequence_length)
    attention_map_all_heads = attention_layers[-1].squeeze(0).mean(dim=0).detach().cpu().numpy()
    
    # Clamp to the actual sequence length (ignore padding)
    actual_len = len(tokens)
    attention_map_plot = attention_map_all_heads[:actual_len, :actual_len]
    
    def plot_transformer_attention_map(attention_map, labels):
        fig, ax = plt.subplots(figsize = (10,6), dpi=300) 
        im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
        ax.grid(False)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8) 
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8) 
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Query (Q)", fontsize=10) 
        ax.set_xlabel("Key (K)", fontsize=10) 
        cbar = fig.colorbar(im, fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        return fig, ax

    fig, ax = plot_transformer_attention_map(attention_map_plot, tokens)
    ax.set_title(f"Attention Map (Arch: {arch_settings['name']}, Token: {token_settings['name']})")
    plt.show()
    
else:
    rich.print("[bold red]ERROR:[/bold red] Transformer model did not return attention weights. Cannot visualize attention map.")