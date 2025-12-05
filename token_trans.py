import os
from sklearn.pipeline import islice
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
import rich, sys
from typing import List, Tuple, Dict, Any, Union
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
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
import rich

from byte_level_tokenizer import ByteLevelTokenizer


GLOVE_PICKLE_FILE = "glove_data.pickle"

if not Path(GLOVE_PICKLE_FILE).exists():
    rich.print(f"[bold cyan]Saving GloVe data to {GLOVE_PICKLE_FILE} ...[/bold cyan]")

    #   This import automatically downloads the full dataset
    from token_utils import glove_vectors, glove_vocabulary, glove_tokenizer
    
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
ARCHITECTURE = int(sys.argv[1])

# --- 2. Choose the Tokenization Scheme ---
# TOKENIZER = 1: GloVe Tokenizer
# TOKENIZER = 2: BERT WordPiece Tokenizer
# TOKENIZER = 3: GPT-2 BPE Tokenizer
# TOKEN = 4: Byte-level encoding
TOKENIZER = int(sys.argv[2])

# Dataset options:

# 1: AG News dataset
# 2: Tiny Shakespeare dataset
# 3: FineWeb2 dataset
DATASET_OPTION = int(sys.argv[3])
max_dataset_size = int(sys.argv[4])


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
    4: {"name": "ByteLevel", "hf_name": None},
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
    
elif TOKENIZER == 4:
    rich.print("[bold red]Using TOKENIZER 4: Byte-level encoding[/bold red]")
    current_tokenizer = ByteLevelTokenizer()
    current_vocab_size = current_tokenizer.vocab_size   # 258 by default
    current_embed_dim = 300
    PAD_IDX = current_tokenizer.pad_token_id
else:
    # BERT or GPT-2 Tokenizer (HF Vocab)
    current_tokenizer = AutoTokenizer.from_pretrained(token_hf_name)
    current_vocab_size = current_tokenizer.vocab_size
    if TOKENIZER == 2:      # BERT
        rich.print("[bold red]Using TOKENIZER 2: HuggingFace WordPiece (BERT-base-uncased)[/bold red]")
        PAD_IDX = current_tokenizer.pad_token_id or current_tokenizer.unk_token_id
    elif TOKENIZER == 3:    # GPT-2
        rich.print("[bold red]Using TOKENIZER 3: HuggingFace BPE (GPT-2)[/bold red]")
        current_tokenizer.pad_token = current_tokenizer.eos_token 
        PAD_IDX = current_tokenizer.pad_token_id
    else:
        raise ValueError(f"Invalid TOKEN value: {TOKENIZER}. Must be 1, 2, 3, or 4.")


#   MODEL SETUP (Weights)
config = AutoConfig.from_pretrained(arch_name) 
config.vocab_size = current_vocab_size 
config.hidden_size = 240                        #   Must be multiple of 12 (number of attention heads) Same as RNN
config.num_hidden_layers = 11                   #   Down from 12 (BERT-base)
config.intermediate_size = 1024                 #   Must be adjusted proportionally

config.output_hidden_states = True
config.output_attentions = True

current_model = model_class.from_config(config) 
rich.print(f"[bold red]{arch_settings['name']}: Using RANDOM weights (vocab size adjusted).[/bold red]")


current_embed_dim = current_model.config.hidden_size

# ----------------------------------------------------------------------
#   CONSTANTS AND DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 1
max_seq_size = 10 
rich.print(f"Device: [red]{DEVICE}[/red] | Model Class: [red]{current_model.__class__.__name__}[/red] \
Vocab Size: [red]{current_vocab_size}[/red] | Embed Dim: [red]{current_embed_dim}[/red] | PAD IDX: [red]{PAD_IDX}[/red]")

# control verbosity
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

# ----------------------------------------------------------------------
#   HELPER FUNCTIONS
# ----------------------------------------------------------------------

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
    
    elif isinstance(tokenizer, ByteLevelTokenizer):
        enc = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_attention_mask=False,
        )
        token_ids = enc["input_ids"]
        return {"token_ids": token_ids}
        
    else:
        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")
    

def count_chars(sample, key):
    # Save text
    text = sample[key]
    # preprocess text by removing spaces and commas
    text = text.replace(" ", "").replace(",", "")
    return len(text)


def count_bytes(sample, key, encoding="utf-8"):
    """
    Counts the number of bytes in the given text sample when encoded with the specified encoding.
    """
    text = sample[key]
    text = text.replace(" ", "").replace(",", "")
    return len(text.encode(encoding))


# ----------------------------------------------------------------------
#  SIZE BREAKDOWN HELPERS 
# ----------------------------------------------------------------------

def bytes_to_readable(total_bytes: int) -> str:
    """Convert a number of bytes into a human-readable string."""
    if total_bytes >= 1024**3:   # Gigabytes
        return f"{total_bytes / 1024**3:.2f} GB"
    elif total_bytes >= 1024**2: # Megabytes
        return f"{total_bytes / 1024**2:.2f} MB"
    else:                        # Kilobytes
        return f"{total_bytes / 1024:.2f} KB"


def get_transformer_head_module(model: PreTrainedModel):
    """
    Try to locate the output head (LM head / decoder) of a transformers model.
    Returns the module or None if it cannot be found.
    """
    # 1) Official API: get_output_embeddings
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    
    # 2) GPT-like causal LMs usually expose `lm_head`
    if hasattr(model, "lm_head"):
        return model.lm_head
    
    # 3) BERT-style MaskedLM often uses `cls.predictions.decoder`
    if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
        preds = model.cls.predictions
        if hasattr(preds, "decoder"):
            return preds.decoder
    
    return None


def get_transformer_size_breakdown(model: PreTrainedModel) -> Dict[str, int]:
    """
    Return a dictionary with the parameter breakdown of a transformers model:
      - total_params : all parameters
      - embed_params : input embedding table
      - head_params  : output head / decoder (if not tied)
      - core_params  : the rest (encoder/decoder blocks only)
    Correctly handles the case where input embeddings and output head share weights.
    """
    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Input embeddings
    embed_params = 0
    embed_weight = None
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            embed_weight = emb.weight
            embed_params = emb.weight.numel()
    
    # Output head
    head_params = 0
    head_weight = None
    head_module = get_transformer_head_module(model)
    if head_module is not None and hasattr(head_module, "weight"):
        head_weight = head_module.weight
        
        # If head and embeddings share the same underlying storage, do not double-count
        if embed_weight is not None and head_weight.data_ptr() == embed_weight.data_ptr():
            head_params = 0
        else:
            head_params = head_weight.numel()
    
    core_params = total_params - embed_params - head_params
    
    return {
        "total_params": total_params,
        "embed_params": embed_params,
        "head_params": head_params,
        "core_params": core_params,
    }


def print_transformer_size_breakdown(model: PreTrainedModel):
    """
    Pretty-print the size breakdown (in parameters and bytes) for a transformers model.
    """
    breakdown = get_transformer_size_breakdown(model)
    
    total_bytes = breakdown["total_params"] * 4
    embed_bytes = breakdown["embed_params"] * 4
    head_bytes = breakdown["head_params"] * 4
    core_bytes = breakdown["core_params"] * 4
    
    rich.print("\n" + "="*50)
    rich.print(f"[bold red]TRANSFORMER MODEL SIZE BREAKDOWN[/bold red]\n")
    rich.print(f"  Total params      : {breakdown['total_params']:,} "
               f"([bold red]{bytes_to_readable(total_bytes)}[/bold red])")
    rich.print(f"  Embeddings (WTE)  : {breakdown['embed_params']:,} "
               f"([bold red]{bytes_to_readable(embed_bytes)}[/bold red])")
    rich.print(f"  Output head       : {breakdown['head_params']:,} "
               f"([bold red]{bytes_to_readable(head_bytes)}[/bold red])")
    rich.print(f"  Core (blocks only): {breakdown['core_params']:,} "
               f"([bold red]{bytes_to_readable(core_bytes)}[/bold red])")
    rich.print("="*50 + "\n")


# ----------------------------------------------------------------------
#   DATASET PREPARATION
# ----------------------------------------------------------------------

if DATASET_OPTION == 1:
    rich.print(f"[bold blue]Using AG News Dataset: {max_dataset_size} samples[/bold blue]")
    # load AG News, take a subset of `max_dataset_size` rows and tokenize
    dataset = datasets.load_dataset("ag_news")
    dataset = datasets.DatasetDict({split: dset.select(range(max_dataset_size)) if len(dset) > max_dataset_size else dset for split, dset in dataset.items()})
elif DATASET_OPTION == 2:
    rich.print(f"[bold blue]Using Tiny Shakespeare Dataset: {max_dataset_size} samples[/bold blue]")
    dataset = datasets.load_dataset("Trelis/tiny-shakespeare")
    dataset = datasets.DatasetDict({split: dset.select(range(max_dataset_size)) if len(dset) > max_dataset_size else dset for split, dset in dataset.items()})
elif DATASET_OPTION == 3:
    rich.print(f"[bold blue]Using FineWeb2 Dataset: {max_dataset_size} samples[/bold blue]")
    splits = ["train", "test"]
    filtered_dict = {}
    for split in splits:
        # Load FineWeb2 as IterableDataset for each split
        fineweb_list = list(islice(datasets.load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="spa_Latn",
            split=split,
            streaming=True
        ), max_dataset_size))
        
        # Filter only the 'text' field
        filtered = [{"text": row["text"]} for row in fineweb_list]
        
        # Build the Dataset for that split
        filtered_dict[split] = datasets.Dataset.from_list(filtered)

    # Build the final DatasetDict with both splits
    dataset = datasets.DatasetDict(filtered_dict)
else:
    raise ValueError(f"Invalid DATASET_OPTION: {DATASET_OPTION}. Must be 1, 2, or 3.")

# Use the global `current_tokenizer`
dataset = dataset.map(
    partial(batch_tokenize, tokenizer=current_tokenizer, key = "Text" if DATASET_OPTION == 2 else "text"), 
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


# Pre-calculate total characters in test set for BPC calculation
total_test_chars = sum(
    count_chars(sample, key="Text" if DATASET_OPTION == 2 else "text")
    for sample in dataset["test"]
)
total_test_bytes = sum(
    count_bytes(sample, key="Text" if DATASET_OPTION == 2 else "text")
    for sample in dataset["test"]
)
total_test_bits = total_test_bytes * 8

#   Prepare the full test text for compression baselines
test_text = "".join(
    sample["Text" if DATASET_OPTION == 2 else "text"].replace(" ", "").replace(",", "")
    for sample in dataset["test"]
)
test_text_bytes = test_text.encode("utf-8")

start_time = time.time()
accumulated_testing_duration = 0.0


train_loss_per_epoch = []
test_loss_per_epoch = []
final_total_training_tokens = 0

for epoch in range(NUM_EPOCHS):
    rich.print(f"[bold blue]-- Epoch {epoch+1}/{NUM_EPOCHS} --[/bold blue]")
    
    #   Set model to training mode
    current_model.train()
    total_loss = 0
    total_tokens = 0
    
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

        #   weight the loss by the number of non-PAD tokens
        batch_tokens = (targets != PAD_IDX).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        # 5. Backward Pass and Optimization
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / total_tokens
    train_loss_per_epoch.append(avg_loss)

    final_total_training_tokens += total_tokens


    # TESTING PHASE at the end of each epoch
    # Do not time testing phase
    start_testing_time = time.time()

    current_model.eval()
    total_test_loss = 0
    total_test_tokens = 0
    total_entropy_nats = 0.0 
    with torch.no_grad():
        for step, token_ids_batch in enumerate(tqdm(
            torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collate_fn,
                pin_memory=True,
            ),
            desc=f"Testing {epoch+1}"
        )):
            
            token_ids_batch = token_ids_batch.to(DEVICE) 

            # Forward Pass
            outputs = current_model(token_ids_batch)
            logits = outputs.logits
            
            # Prepare Logits and Targets
            logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            targets = token_ids_batch[:, 1:].reshape(-1)
            
            # --------------------------------------------------------------
            # 1) NLL (Cross-entropy) 
            loss = criterion(logits_flat, targets)

            # weight the loss by the number of non-PAD tokens
            batch_tokens = (targets != PAD_IDX).sum().item()
            total_test_loss += loss.item() * batch_tokens
            total_test_tokens += batch_tokens
            # --------------------------------------------------------------
            # 2) ENTROPY RATE: H(p(.|context)) in nats
            #    H(p) = - sum_v p_v log p_v
            log_probs = torch.log_softmax(logits_flat, dim=-1)        # (N, V)
            probs = log_probs.exp()                                   # (N, V)

            entropy_per_position = -(probs * log_probs).sum(dim=-1)   # (N,), in nats

            # Only consider non-PAD tokens for entropy calculation
            non_pad_mask = (targets != PAD_IDX)
            entropy_valid = entropy_per_position[non_pad_mask]

            total_entropy_nats += entropy_valid.sum().item()

    avg_test_loss = total_test_loss / total_test_tokens
    test_loss_per_epoch.append(avg_test_loss)

    # BITS PER CHARACTER (BPC) CALCULATION
    #---------------------------------------
    total_test_nll_nats = total_test_loss

    # Convert to bits
    total_test_nll_bits = total_test_nll_nats / math.log(2.0)

    # BPC = total bits / nº of characters in test set
    test_bpc = total_test_nll_bits / total_test_chars

    #---------------------------------------
    # BITS PER BYTE (BPB) CALCULATION
    bpb_model = total_test_nll_bits / total_test_bytes

    #---------------------------------------
    # COMPRESSION RATIO ESTIMATION
    # Assuming 8 bits per byte in original text
    compression_ratio_model = total_test_nll_bits / total_test_bits

    #---------------------------------------
    # NLL per char
    nll_per_char_nats = total_test_loss / total_test_chars

    #---------------------------------------
    # PERPLEXITY PER CHARACTER
    perplexity_per_char = math.exp(nll_per_char_nats)

    #---------------------------------------
    # ENTROPY RATE per char
    entropy_rate_nats_per_token = total_entropy_nats / total_test_tokens
    entropy_rate_bits_per_token = entropy_rate_nats_per_token / math.log(2.0)

    # Convert to "per character" using the total number of characters in the test corpus
    entropy_rate_bits_per_char = (total_entropy_nats / math.log(2.0)) / total_test_chars
    #---------------------------------------

    # SUMMARY OF EPOCH
    rich.print(
        f"[bold green]Epoch {epoch+1} Complete."
        # f" Avg Train Loss: {avg_loss:.4f}"
        # f" | Avg Test Loss (per token): {avg_test_loss:.4f}"
        f" | Perplexity/char: {perplexity_per_char:.4f}"
        # f" | Test BPC: {test_bpc:.4f} bits/char"
        # f" | Entropy Rate: {entropy_rate_bits_per_char:.4f} bits/char"
        f" | Test BPB: {bpb_model:.4f} bits/byte"
        f" | Compression Ratio: {compression_ratio_model:.4f}[/bold green]"
        # f" | Entropy Rate: {entropy_rate_bits_per_token:.4f} bits/token"
    )

    # CALCULATE TESTING DURATION
    end_testing_time = time.time()
    testing_duration = end_testing_time - start_testing_time
    accumulated_testing_duration += testing_duration

end_time = time.time()
total_time = end_time - start_time - accumulated_testing_duration # exclude testing time
rich.print(f"Total Training Time: [yellow]{total_time:.2f} seconds[/yellow]")
print(f"Total tokens: {final_total_training_tokens}\n")


# Use the same sentence from the attention map visualization, as to evaluate the token's metrics
eval_sentence = "A dog is an amazing animal with a heart of a true lifemate of men, and with many other qualities"
rich.print(f"\nTokenizer evaluation on '{eval_sentence}'")

# # ----------------------------------------------------------------------
# #   NORMALIZED LENGTH SCORE (NLS) EVALUATION
# # ----------------------------------------------------------------------

rich.print("\n[bold blue]STARTING NORMALIZED LENGTH SCORE (NLS) EVALUATION...[/bold blue]\n")
""" The higher the NLS value, the worse the option is, in principle. """
# Count words (W) and characters (C)
nls_word_count = len(eval_sentence.split())
nls_char_count = len(eval_sentence.replace(" ", "").replace(",","")) # Count characters excluding spaces and comma

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

# # ----------------------------------------------------------------------
# #   SUBWORD FERTILITY (SF) AND CONTINUED WORDS (CW) EVALUATION
# # ----------------------------------------------------------------------

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
  
    #   Note: Split words count in this methodology starts counting *after* the initial word part
    #   A word "mammal" split into ["ma", "##mmal"] is counted as 1 split word if '##mmal' is found.
    return fertility, proportion_continued_words

fertility, pcw = calculate_subword_metrics(eval_sentence, current_tokenizer, actual_tokens)
rich.print(f"Subword Fertility (Ideal: 1.0): [green]{fertility:.4f}[/green]")
rich.print(f"Proportion of Continued Words (Ideal: 0.0): [green]{pcw:.4f}[/green]")
rich.print("\n[bold blue]Subword Fertility Metrics Complete.[/bold blue]")


print_transformer_size_breakdown(current_model)


#   Uncomment section below for gradient visualization plots
"""
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

#   Get token names only for the actual content
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
"""