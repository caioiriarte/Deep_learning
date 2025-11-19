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
import pickle
import time


GLOVE_PICKLE_FILE = "glove_data.pickle"

if not Path(GLOVE_PICKLE_FILE).exists():
    rich.print(f"[bold cyan]Saving GloVe data to {GLOVE_PICKLE_FILE} ...[/bold cyan]")

    #   This import automatically downloads the full dataset
    from script1 import glove_vectors, glove_vocabulary, glove_tokenizer
    
    glove_data_to_save = {
        "vectors": glove_vectors,
        "vocabulary": glove_vocabulary,
        "tokenizer": glove_tokenizer
    }
    
    #   Write the dictionary to the pickle file
    try:
        with open(GLOVE_PICKLE_FILE, 'wb') as f:
            pickle.dump(glove_data_to_save, f)
        rich.print(f"[bold green]Successfully saved GloVe data.[/bold green]")
        
    except Exception as e:
        rich.print(f"[bold red]ERROR saving GloVe data:[/bold red] {e}")

else:
    with open(GLOVE_PICKLE_FILE, 'rb') as f:
        loaded_data = pickle.load(f)
        glove_vectors = loaded_data['vectors']
        glove_vocabulary = loaded_data['vocabulary']
        glove_tokenizer = loaded_data['tokenizer']


sns.set_theme()

# ----------------------------------------------------------------------
#   SETUP AND MODEL/TOKENIZER MACROS
# ----------------------------------------------------------------------

# --- 1. Choose the Model Architecture ---
# ARCHITECTURE = 1: Base (GloVe-compatible/Random Init)
# ARCHITECTURE = 2: BERT-config (Random Init)
# ARCHITECTURE = 3: GPT-2-config (Random Init)
ARCHITECTURE = 3

# --- 2. Choose the Tokenization Scheme ---
# TOKENIZER = 1: GloVe Tokenizer
# TOKENIZER = 2: BERT WordPiece Tokenizer
# TOKENIZER = 3: GPT-2 BPE Tokenizer
TOKENIZER = 3

# Configuration mapping for architectures
ARCH_CONFIGS = {
    1: {"name": "Base_Random", "hf_model_name": "bert-base-uncased", "model_class": AutoModelForMaskedLM,"random_init": True},
    2: {"name": "BERT_Encoder_Random", "hf_model_name": "bert-base-uncased", "model_class": AutoModelForMaskedLM, "random_init": True},
    3: {"name": "GPT2_Decoder_Random", "hf_model_name": "gpt2", "model_class": AutoModelForCausalLM, "random_init": True},
}

# Configuration mapping for tokenizers
TOKEN_CONFIGS = {
    1: {"name": "GloVe", "hf_name": None},
    2: {"name": "BERT_WP", "hf_name": "bert-base-uncased"},
    3: {"name": "GPT2_BPE", "hf_name": "gpt2"},
}

# Global variables
current_tokenizer = None
current_model = None
current_vocab_size = 0
current_embed_dim = 0
PAD_IDX = 0

# --- Conditional Initialization Block ---

arch_settings = ARCH_CONFIGS.get(ARCHITECTURE)
token_settings = TOKEN_CONFIGS.get(TOKENIZER)

if arch_settings is None or token_settings is None:
    raise ValueError("Invalid ARCHITECTURE or TOKENIZER value. Must be 1, 2, or 3.")

arch_name = arch_settings["hf_model_name"]
model_class = arch_settings["model_class"]
token_hf_name = token_settings["hf_name"]

rich.print(f"[bold red]ARCHITECTURE: {arch_settings['name']} ({ARCHITECTURE})[/bold red] \
[bold red]TOKENIZER: {token_settings['name']} ({TOKENIZER})[/bold red]")

#   TOKENIZER SETUP
if TOKENIZER == 1:
    #   GloVe Tokenizer (Custom Vocab)
    current_tokenizer = glove_tokenizer

    #   GloVe's vocab size is fixed by the dataset
    glove_vocab_size = glove_vectors.shape[0]

    PAD_IDX = glove_vocabulary.index('<|pad|>') if '<|pad|>' in glove_vocabulary else 0
    rich.print(f"[bold yellow]Using fixed GloVe Vocab Size: {glove_vocab_size}[/bold yellow]")
    current_vocab_size = glove_vocab_size
    
else:
    # BERT or GPT-2 Tokenizer (HF Vocab)
    current_tokenizer = AutoTokenizer.from_pretrained(token_hf_name)
    current_vocab_size = current_tokenizer.vocab_size
    if TOKENIZER == 2:      # BERT
        PAD_IDX = current_tokenizer.pad_token_id or current_tokenizer.unk_token_id
    elif TOKENIZER == 3:    # GPT-2
        current_tokenizer.pad_token = current_tokenizer.eos_token 
        PAD_IDX = current_tokenizer.pad_token_id

#   MODEL SETUP (Weights)
config = AutoConfig.from_pretrained(arch_name) 
config.vocab_size = current_vocab_size 
config.hidden_size = 240                    #   Must be multiple of 12
config.num_hidden_layers = 6                #   Down from 12 (BERT-base)
config.intermediate_size = 1024             #   Must be adjusted proportionally

config.output_hidden_states = True
config.output_attentions = True

current_model = model_class.from_config(config) 
rich.print(f"[bold red]{arch_settings['name']}: Using RANDOM weights (vocab size adjusted).[/bold red]")


current_embed_dim = current_model.config.hidden_size
current_model.eval()

# ----------------------------------------------------------------------
#   CONSTANTS AND DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 10
max_dataset_size = 512
max_seq_size = 50 
rich.print(f"Device: [red]{DEVICE}[/red] | Model Class: [red]{current_model.__class__.__name__}[/red] \
Vocab Size: [red]{current_vocab_size}[/red] | Embed Dim: [red]{current_embed_dim}[/red] | PAD IDX: [red]{PAD_IDX}[/red]")

# control verbosity
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def custom_collate_fn(batch):
    """
    Collate function to extract 'token_ids' from the dataset batch 
    and convert them into a single long tensor.
    """
    token_ids_list = [item['token_ids'] for item in batch]
    return torch.tensor(token_ids_list, dtype=torch.long) 

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
        token_ids_padded = []

        for x in encodings:
            #   Truncate
            ids = x.ids[:max_length]
            #   Pad
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
#   DATASET PREPARATION
# ----------------------------------------------------------------------

# load AG News, take a subset of `max_dataset_size` rows and tokenize
dataset = datasets.load_dataset("ag_news")
dataset = datasets.DatasetDict({split: dset.select(range(max_dataset_size)) if \
                                len(dset) > max_dataset_size else dset for split, dset in dataset.items()})

# Use the global `current_tokenizer`
dataset = dataset.map(
    partial(batch_tokenize, tokenizer=current_tokenizer), 
    batched=True, 
    num_proc=2, 
    batch_size=10,
    load_from_cache_file=False
)
rich.print(dataset)


# Create the DataLoader for the training split
train_dataloader = torch.utils.data.DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate_fn,
    pin_memory=True,      # Speeds up data transfer to GPU
)

# ----------------------------------------------------------------------
#   MODEL INSTANTIATION, LOSS, AND OPTIMIZER
# ----------------------------------------------------------------------

#   Send it to the device
current_model = current_model.to(DEVICE)

#   Define the Loss Criterion
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion = criterion.to(DEVICE)

#   Define the Optimizer
optimizer = torch.optim.Adam(current_model.parameters(), lr=1e-5) 

# ----------------------------------------------------------------------
#   FINE-TUNING LOOP
# ----------------------------------------------------------------------
rich.print("[bold green]STARTING FINE-TUNING...[/bold green]")

start_time = time.time()

#   Set model to training mode
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
        #   We apply the Causal Language Modeling (CLM) shift:
        #   Input t predicts Target t+1 (0 to T-2)
        logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])

        #   Supposet targets to have been predicted (1 to T-1)
        targets = token_ids_batch[:, 1:].reshape(-1)
        
        # 4. Calculate Loss
        loss = criterion(logits_flat, targets)
        total_loss += loss.item()

        # 5. Backward Pass and Optimization
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_dataloader)
    rich.print(f"[bold green]Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}[/bold green]")

#   Set model back to evaluation for visualization blocks
current_model.eval()

total_time = time.time() - start_time
print(f"\nTotal time for training model: {total_time} s")


# Use the same sentence from the attention map visualization, as to evaluate the token's metrics
eval_sentence = "A dog is an amazing animal with a heart of a true lifemate of men, and with many other qualities"
rich.print(f"\nTokenizer evaluation on '{eval_sentence}'")

# ----------------------------------------------------------------------
#   NORMALIZED LENGTH SCORE (NLS) EVALUATION
# ----------------------------------------------------------------------

rich.print("\n[bold blue]STARTING NORMALIZED LENGTH SCORE (NLS) EVALUATION...[/bold blue]\n")
""" The higher the NLS value, the worse the option is, in principle. """

# Count words (W) and characters (C)
nls_word_count = len(eval_sentence.split())
nls_char_count = len(eval_sentence.replace(" ", "").replace(",","")) # Count characters excluding spaces and commas

# We use a known efficient tokenizer like Llama for the NLS (Reference) metric
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

# A. Tokenize with the CURRENT Tokenizer
current_token_ids = batch_tokenize({"text": [eval_sentence]}, max_length=max_seq_size, tokenizer=current_tokenizer)['token_ids'][0]
non_padded_token_ids = torch.tensor([id for id in current_token_ids if id != PAD_IDX])
actual_tokens = get_tokens_from_ids(non_padded_token_ids, current_tokenizer)

# B. Calculate NLS_W (Word-Normalized Length)
token_count = len(actual_tokens)
nls_w = token_count / nls_word_count
rich.print(f"NLS_w: [green]{nls_w:.4f}[/green] (Tokens per Word)")

# C. Calculate NLS_C (Character-Normalized Length)
nls_c = token_count / nls_char_count
rich.print(f"NLS_c: [green]{nls_c:.4f}[/green] (Tokens per Character)")

# D. Calculate NLS (Reference)
if t5_tokenizer is not None:
    # Tokenize with the REFERENCE Tokenizer
    t5_tokenization = t5_tokenizer(
        eval_sentence,
        max_length=100,
        padding=False,
        truncation=True,
        return_attention_mask=False
    )
    # Exclude special tokens like BOS/EOS/Pad if they were included
    t5_token_ids = t5_tokenization["input_ids"]
    t5_token_count = len(t5_token_ids)
    
    # Calculate NLS as the ratio of the tested tokenizer's token count to the reference tokenizer's count
    nls_ref = token_count / t5_token_count

    rich.print(f"NLS (Reference: T/R): [green]{nls_ref:.4f}[/green] (Efficiency vs. t5)")


rich.print("\n[bold blue]NLS Evaluation Complete.[/bold blue]")

# ----------------------------------------------------------------------
#   SUBWORD FERTILITY (SF) AND CONTINUED WORDS (CW) EVALUATION
# ----------------------------------------------------------------------

rich.print("\n[bold blue]STARTING SUBWORD FERTILITY (SF) AND CONTINUED WORDS (CW) EVALUATION...[/bold blue]\n")

"""
    The subword fertility evaluates the amount of tokens generated per word. A fertility of 1.0 is
    ideal. Normally, the SF value is >= 1.0

    Another metric that is employed is the proportion of continued words which indicates the percentage
    of words that are split into multiple tokens. 0 is an ideal value for the CW proportion.
"""

def calculate_subword_metrics(sentence: str, tokenizer: Any, tokens: List[str]) -> Tuple[float, float]:
    # Count words using simple space splitting (as per standard practice for these metrics)
    total_words = len(sentence.split())
    total_tokens = len(tokens)
    
    if total_words == 0:
        return 0.0, 0.0
        
    fertility = total_tokens / total_words
    
    #   Calculate Proportion of Continued Words (PCW)
    split_words_count = 0
    in_split_word = False
    
    # Check for GloVe/Word-based tokenizer
    if token_settings['name'] == 'GloVe':
        # GloVe is strictly word-based, so no words are split (Fertility = 1.0, PCW = 0.0)
        return 1.0, 0.0
    
    for i, token in enumerate(tokens):
        if token_settings['name'] == 'BERT_WP':
            if token.startswith('##'):
                # Token is a continuation of a previous word
                if not in_split_word:
                    # Found the first continuation token for a split word
                    split_words_count += 1
                    in_split_word = True
            else:
                # Token is a new word (or the start of a split word)
                in_split_word = False
                
        elif token_settings['name'] == 'GPT2_BPE':
            # Check for the space marker 'Ġ' (which signifies the start of a new word)
            if i > 0 and not token.startswith('Ġ'):
                # Token is a continuation of the previous word
                if not in_split_word:
                    # Found the first continuation token for a split word
                    split_words_count += 1
                    in_split_word = True
            else:
                # Token is a new word (or the first token of the sequence)
                in_split_word = False

    #   PCW = (Number of split words) / (Total number of words)
    proportion_continued_words = split_words_count / total_words if total_words > 0 else 0.0
    
    # Note: Split words count in this methodology starts counting *after* the initial word part
    # A word "mammal" split into ["ma", "##mmal"] is counted as 1 split word if '##mmal' is found.
    
    return fertility, proportion_continued_words


fertility, pcw = calculate_subword_metrics(eval_sentence, current_tokenizer, actual_tokens)

rich.print(f"Subword Fertility (Ideal: 1.0): [green]{fertility:.4f}[/green]")
rich.print(f"Proportion of Continued Words (Ideal: 0.0): [green]{pcw:.4f}[/green]")

rich.print("\n[bold blue]Subword Fertility Metrics Complete.[/bold blue]")


# ----------------------------------------------------------------------
#   GRADIENT VISUALIZATION TEST
# ----------------------------------------------------------------------

rich.print("\n[bold yellow]STARTING GRADIENT VISUALIZATION TEST...[/bold yellow]")

#   Reset gradients 
current_model.zero_grad()

T_SIZE = 10
token_ids_test = torch.randint(low=0, high=current_vocab_size, size=(T_SIZE, T_SIZE))
token_ids_test = token_ids_test.to(DEVICE)

#   Get the internal Embedding layer (Model-specific access)
wte = None
if hasattr(current_model, 'transformer') and hasattr(current_model.transformer, 'wte'):
    wte = current_model.transformer.wte                     #   GPT2 access
elif hasattr(current_model, 'bert') and hasattr(current_model.bert.embeddings, 'word_embeddings'):
    wte = current_model.bert.embeddings.word_embeddings     #   BERT access
    
if wte is not None:
    #   The `wte` layer converts the long tensor (IDs) to a float tensor (Embeddings)
    input_embeddings = wte(token_ids_test)
    
    #   Now, set requires_grad and retain_grad on the continuous embedding tensor
    input_embeddings.requires_grad_(True) 
    input_embeddings.retain_grad()
    
    #   Run the rest of the model from the embeddings
    outputs = current_model(inputs_embeds=input_embeddings)
    logits = outputs.logits
    
    #   Compute the specific loss for the visualization test
    loss_locations = torch.arange(0, T_SIZE)[:, None, None].expand(T_SIZE, 1, logits.shape[-1])
    loss_locations = loss_locations.to(DEVICE)
    loss_test = logits.gather(index=loss_locations, dim=1).mean()
    
    #   Backward pass to retrieve the gradients
    loss_test.backward()
    
    #   The magnitude is the L2 norm of the gradient w.r.t. the input embeddings
    grad_magnitude = input_embeddings.grad.norm(dim=2)
    
    #   Visualize the gradient
    grad_magnitude[grad_magnitude==0] = -math.inf 
    grad_magnitude = grad_magnitude.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
    cbar = plt.colorbar() 
    cbar.ax.tick_params(labelsize=10)
    plt.grid(False)
    plt.xlabel("$t$ (input)",fontsize=10)
    plt.ylabel("$t'$ (loss)",fontsize=10)
    plt.title(f"Gradient Magnitude (Arch: {arch_settings['name']}, Token: {token_settings['name']})",fontsize=16)
    plt.show()

    #   -----------------------------------------------------------------------------

    #   Get the size of the WTE for comparison (LUT table)
    wte_params = wte.weight.numel() 
    total_wte = wte_params * 4        #   Bytes/parameter (float32)

    #   Get the size related to the MODEL and WTE together (obtain parameters' amount & related B size)
    total_params = sum(p.numel() for p in current_model.parameters())
    total_bytes = total_params * 4
    total_model = total_bytes - total_wte

    def bytes_size(total_bytes):
        if total_bytes > 1024**3:   # Gigabytes
            readable_size = f"{total_bytes / 1024**3:.2f} GB"
        elif total_bytes > 1024**2: # Megabytes
            readable_size = f"{total_bytes / 1024**2:.2f} MB"
        else:                       # Kilobytes
            readable_size = f"{total_bytes / 1024:.2f} KB"

        return readable_size
    
    tot_bytes_size = bytes_size(total_bytes)
    wte_size = bytes_size(total_wte)
    model_size = bytes_size(total_model)
    print(f"\nTOTAL SIZE OF CHOSEN OPTION: MODEL (parameters) + TOKENIZER (WTE size): {tot_bytes_size}")
    print(f"TOTAL SIZE OF TRANSFORMER LAYERS (excluding WTE): {model_size}")
    print(f"TOTAL WTE SIZE (vector matrix size): {wte_size}\n")
else:
    rich.print("[bold red]ERROR:[/bold red] Could not reliably access the Word Token Embedding (WTE) layer for gradient test.")

# ----------------------------------------------------------------------
#   ATTENTION MAP VISUALIZATION
# ----------------------------------------------------------------------

rich.print("[bold yellow]STARTING ATTENTION MAP VISUALIZATION (Transformer)...[/bold yellow]")

#   get a more natural sentence
sentence = "A dog is a really stunning mammal from the animal kingdom and is also considered the best friend of men"

#   Tokenize the sentence 
full_token_ids = torch.tensor(batch_tokenize({"text": [sentence]}, max_length=25, tokenizer=current_tokenizer)['token_ids'][0])
token_ids_list = [id for id in full_token_ids if id != PAD_IDX]
token_ids = torch.tensor(token_ids_list) 

# Get token names only for the actual content
tokens = get_tokens_from_ids(token_ids, current_tokenizer)

#   Pass the input to the model to get attention weights
input_ids_tensor = token_ids.unsqueeze(0).to(DEVICE)

#   Run the forward pass, ensuring attention is outputted
outputs = current_model(input_ids_tensor, output_attentions=True)

if hasattr(outputs, 'attentions') and outputs.attentions is not None:
    attention_layers = outputs.attentions
    attention_map_all_heads = attention_layers[-1].squeeze(0).mean(dim=0).detach().cpu().numpy()
    
    #   Clamp to the actual sequence length (ignore padding)
    actual_len = len(tokens)
    attention_map_plot = attention_map_all_heads[:actual_len, :actual_len]
    
    def plot_transformer_attention_map(attention_map, labels):
        fig, ax = plt.subplots(figsize = (6, 3), dpi=300) 
        
        im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
        ax.grid(False)
        
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=3) 
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=3) 
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_ylabel("Query (Q)", fontsize=7) 
        ax.set_xlabel("Key (K)", fontsize=7) 
        
        cbar = fig.colorbar(im, 
                            fraction=0.046,     #   fraction and pad slightly adjusted for bottom position
                            pad=0.04, 
                            location='right',   #   Place color bar horizontally at the bottom
                            shrink=0.75)        #   Shrink the color bar size
        
        cbar.ax.tick_params(labelsize=4)
        fig.tight_layout()
        return fig, ax

    fig, ax = plot_transformer_attention_map(attention_map_plot, tokens)
    ax.set_title(f"Attention Map (Arch: {arch_settings['name']}, Token: {token_settings['name']})", fontsize=8)
    plt.show()
    
else:
    rich.print("[bold red]ERROR:[/bold red] Transformer model did not return attention weights. Cannot visualize attention map.")