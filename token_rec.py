from itertools import islice
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
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
import tokenizers
import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import pickle
import time
import gzip
import bz2
import lzma

from byte_level_tokenizer import ByteLevelTokenizer

## LOAD OR SAVE GLOVE DATA
GLOVE_PICKLE_FILE = "glove_data.pickle"

# 1. Check if the pickle file already exists
if not Path(GLOVE_PICKLE_FILE).exists():
    rich.print(f"[bold cyan]Saving GloVe data to {GLOVE_PICKLE_FILE} ...[/bold cyan]")

    #   This import automatically downloads the full dataset
    from script1 import glove_vectors, glove_vocabulary, glove_tokenizer
    
    # 2. Create a dictionary containing all necessary GloVe components
    glove_data_to_save = {
        "vectors": glove_vectors,
        "vocabulary": glove_vocabulary,
        "tokenizer": glove_tokenizer
    }
    
    # 3. Write the dictionary to the pickle file
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
# 1. SETUP AND MACRO (TOKENIZER SWITCH)
# ----------------------------------------------------------------------

# --- Define the macro-like switch variable ---
# TOKEN = 1: Original GloVe Tokenizer
# TOKEN = 2: HuggingFace WordPiece (BERT-base-uncased)
# TOKEN = 3: HuggingFace BPE (GPT-2)
# TOKEN = 4: Byte-level encoding
TOKEN = 4

# Global variables for conditional model setup
current_tokenizer = None
current_vocab_size = 0
current_embed_dim = 0
vectors_for_init = None
PAD_IDX = 0

# --- Conditional Initialization Block ---

if TOKEN == 1:
    rich.print("[bold red]Using TOKENIZER 1: Original GloVe Tokenizer[/bold red]")
    current_tokenizer = glove_tokenizer
    current_vocab_size = glove_vectors.shape[0]
    current_embed_dim = glove_vectors.shape[1]
    vectors_for_init = glove_vectors # Use GloVe vectors for initialization
    # Get PAD_IDX from the known GloVe vocabulary
    PAD_IDX = glove_vocabulary.index('<|pad|>') if '<|pad|>' in glove_vocabulary else 0

elif TOKEN == 2:
    rich.print("[bold red]Using TOKENIZER 2: HuggingFace WordPiece (BERT-base-uncased)[/bold red]")
    tokenizer_name = "bert-base-uncased"
    current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    current_vocab_size = current_tokenizer.vocab_size
    current_embed_dim = 240 # Matching transformer 
    PAD_IDX = current_tokenizer.pad_token_id or current_tokenizer.unk_token_id

elif TOKEN == 3:
    rich.print("[bold red]Using TOKENIZER 3: HuggingFace BPE (GPT-2)[/bold red]")
    tokenizer_name = "gpt2"
    current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    current_vocab_size = current_tokenizer.vocab_size
    current_embed_dim = 240 # Matching transformer 
    # GPT-2 has no explicit pad token by default; using EOS as a common choice for LMs
    current_tokenizer.pad_token = current_tokenizer.eos_token 
    PAD_IDX = current_tokenizer.pad_token_id

elif TOKEN == 4:
    rich.print("[bold red]Using TOKENIZER 4: Byte-level encoding[/bold red]")
    current_tokenizer = ByteLevelTokenizer()
    current_vocab_size = current_tokenizer.vocab_size   # 258 by default
    current_embed_dim = 240 # Matching transformer
    PAD_IDX = current_tokenizer.pad_token_id


else:
    raise ValueError(f"Invalid TOKEN value: {TOKEN}. Must be 1, 2, 3, or 4.")

# ----------------------------------------------------------------------
# 2. CONSTANTS AND DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 1
max_dataset_size = 10#1000000
max_seq_size = 10
rich.print(f"Device: [red]{DEVICE}[/red] | Vocab Size: [red]{current_vocab_size}[/red] | Embed Dim: [red]{current_embed_dim}[/red] | PAD IDX: [red]{PAD_IDX}[/red]")

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
    # Ensure all tokens are padded to the max length in the batch if needed.
    # The dataset.map already truncates/pads based on `max_seq_size` if a transformer tokenizer is used.
    token_ids_list = [item['token_ids'] for item in batch]
    # Stacks the list of token lists into a single (Batch_Size, Max_Seq_Length) tensor
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
    

# Helper function for memory size
def bytes_to_readable(total_bytes: int) -> str:
    if total_bytes > 1024**3:   # Gigabytes
        return f"{total_bytes / 1024**3:.2f} GB"
    elif total_bytes > 1024**2: # Megabytes
        return f"{total_bytes / 1024**2:.2f} MB"
    else:                       # Kilobytes
        return f"{total_bytes / 1024:.2f} KB"
    

# Helper function for subword metrics (copied from previous response)
def calculate_subword_metrics(sentence: str, tokenizer: Any, tokens: List[str]) -> Tuple[float, float]:
    total_words = len(sentence.split())
    total_tokens = len(tokens)
    if total_words == 0:
        return 0.0, 0.0
    fertility = total_tokens / total_words
    split_words_count = 0
    in_split_word = False
    
    if TOKEN == 1:
        return 1.0, 0.0 # GloVe is strictly word-based
    
    for i, token in enumerate(tokens):
        if TOKEN == 2: # BERT_WP
            if token.startswith('##'):
                if not in_split_word:
                    split_words_count += 1
                    in_split_word = True
            else:
                in_split_word = False
                
        elif TOKEN == 3: # GPT2_BPE
            if i > 0 and not token.startswith('Ġ'):
                if not in_split_word:
                    split_words_count += 1
                    in_split_word = True
            else:
                in_split_word = False

    proportion_continued_words = split_words_count / total_words if total_words > 0 else 0.0
    return fertility, proportion_continued_words

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
# 4. RNNLM MODEL DEFINITION (Modified for Flexibility)
# ----------------------------------------------------------------------

class RNNLM(torch.nn.Module):
    """A simple implementation of a language model using RNNs."""
    def __init__(
            self, 
            vocab_size: int, 
            embed_dim: int, 
            vectors: torch.Tensor = None,
            num_parallel_layers: int = 1
        ):
        super().__init__()
        
        # 1. Embeddings Layer
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        
        # Initialize weights with GloVe ONLY if TOKEN=1 and size matches
        if vectors is not None and vectors.shape == (vocab_size, embed_dim):
            rich.print("[bold cyan]Initializing Embeddings and Proj with GloVe Vectors.[/bold cyan]")
            self.embeddings.weight.data = vectors
        else:
            rich.print("[bold cyan]Randomly Initializing Embeddings/Proj (GloVe not used).[/bold cyan]")

        # 2. LSTM Layers in parallel
        #    All layers share the same input/output dimensionality.
        self.num_parallel_layers = num_parallel_layers
        self.rnns = nn.ModuleList([
            nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                num_layers=1,
                batch_first=True,
            )
            for _ in range(self.num_parallel_layers)
        ])

        # 3. Projection Layer 
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Re-initialize projection weights with GloVe ONLY if TOKEN=1
        if vectors is not None and vectors.shape == (vocab_size, embed_dim):
            self.proj.weight.data = vectors

    def forward(self, token_ids: torch.Tensor, retain_ws:bool=False) -> torch.Tensor:
        # convert the tokens into vectors
        ws = self.embeddings(token_ids)

        # store the word vectors for debugging
        if retain_ws:
          ws.retain_grad()
          self.ws = ws

        #   shift the input `ws` right
        # Need to use the embedding weight's dtype, which is typically float
        w0 = torch.zeros((ws.shape[0], 1, self.embeddings.weight.shape[1]),
                         device=self.embeddings.weight.device, dtype=self.embeddings.weight.dtype)
        ws_shifted = torch.cat([w0, ws[:, :-1]], dim=1)

        # Call the parallel RNNs: w_{-1:T-1} -> h_{1:T}
        hidden_states_sum = None
        for rnn in self.rnns:
            hs, _ = rnn(ws_shifted)
            if hidden_states_sum is None:
                hidden_states_sum = hs
            else:
                hidden_states_sum = hidden_states_sum + hs

        # Average the outputs of all parallel layers
        hidden_states = hidden_states_sum / self.num_parallel_layers

        # project the hidden state to the vocabulary space
        logits = self.proj(hidden_states)
        return logits

    def sample(
            self,
            batch_size:int=1,
            num_steps:int=10,
            temperature: float=1.0,
            prevent_repetitions: bool=False
        ):
        token_ids = torch.empty((batch_size, 0), device=self.embeddings.weight.device, dtype=torch.long)
        for t in range(num_steps):
            logits = self.forward(token_ids)
            logits_t = logits[:, -1:] / temperature
            if prevent_repetitions and t > 0:
                # mask the last generated tokens to avoid repetitions
                logits_t.scatter_(index=token_ids[:,-1:, None], dim=2, value=-math.inf)
            p_wt = torch.distributions.Categorical(logits=logits_t)
            tokens_t = p_wt.sample()
            token_ids = torch.cat([token_ids, tokens_t], dim=1)
        return token_ids

# Calculate size of the model:

def get_rnn_size_breakdown(rnn: RNNLM) -> Dict[str, int]:
    """
    Outputs a breakdown of the RNN model size in terms of number of parameters:
      - total_params: all parameters
      - embed_params: embeddings only
      - head_params: output projection only
      - core_params: rest (parallel LSTMs)
    """
    total_params = sum(p.numel() for p in rnn.parameters())
    
    # Embedding (WTE)
    embed_params = rnn.embeddings.weight.numel()
    
    # Output projection (head)
    head_params = rnn.proj.weight.numel()
    
    # Core of the model (independent of vocab_size)
    core_params = total_params - embed_params - head_params
    
    return {
        "total_params": total_params,
        "embed_params": embed_params,
        "head_params": head_params,
        "core_params": core_params,
    }


def print_rnn_size_breakdown(rnn: RNNLM):
    breakdown = get_rnn_size_breakdown(rnn)
    
    total_bytes = breakdown["total_params"] * 4
    embed_bytes = breakdown["embed_params"] * 4
    head_bytes = breakdown["head_params"] * 4
    core_bytes = breakdown["core_params"] * 4
    
    rich.print("\n" + "="*50)
    rich.print("[bold red]RNN MODEL SIZE BREAKDOWN[/bold red]\n")
    rich.print(f"  Total params      : [{breakdown['total_params']:,} "
               f"([bold red]{bytes_to_readable(total_bytes)}[/bold red])")
    rich.print(f"  Embeddings (WTE)  : {breakdown['embed_params']:,} "
               f"([bold red]{bytes_to_readable(embed_bytes)}[/bold red])")
    rich.print(f"  Output head (proj): {breakdown['head_params']:,} "
               f"([bold red]{bytes_to_readable(head_bytes)}[/bold red])")
    rich.print(f"  Core (LSTMs only) : {breakdown['core_params']:,} "
               f"([bold red]{bytes_to_readable(core_bytes)}[/bold red])")
    rich.print("="*50 + "\n")


# ----------------------------------------------------------------------
# 5. DATASET PREPARATION
# ----------------------------------------------------------------------

# Dataset options:

# 1: AG News dataset
# 2: Tiny Shakespeare dataset
# 3: FineWeb2 dataset

DATASET_OPTION = 1

if DATASET_OPTION == 1:
    rich.print(f"[bold blue]Using AG News Dataset: {max_dataset_size} samples[/bold blue]")
    # load AG News, take a subset of `max_dataset_size` rows and tokenize
    dataset = datasets.load_dataset("ag_news")
    dataset = datasets.DatasetDict({split: dset.shuffle(seed=42).select(range(max_dataset_size)) if len(dset) > max_dataset_size else dset for split, dset in dataset.items()})
elif DATASET_OPTION == 2:
    rich.print(f"[bold blue]Using Tiny Shakespeare Dataset: {max_dataset_size} samples[/bold blue]")
    dataset = datasets.load_dataset("Trelis/tiny-shakespeare")
    dataset = datasets.DatasetDict({split: dset.shuffle(seed=42).select(range(max_dataset_size)) if len(dset) > max_dataset_size else dset for split, dset in dataset.items()})
elif DATASET_OPTION == 3:
    rich.print(f"[bold blue]Using FineWeb2 Dataset: {max_dataset_size} samples[/bold blue]")
    splits = ["train", "test"]
    filtered_dict = {}
    for split in splits:
        # Load FineWeb2 as IterableDataset for each split
        streaming_ds = datasets.load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="spa_Latn",
            split=split,
            streaming=True,
        )

        # Shuffle the stream (approximate shuffle using a buffer)
        streaming_ds = streaming_ds.shuffle(seed=42, buffer_size=10_000)

        # Now take max_dataset_size examples from the shuffled stream
        fineweb_list = list(islice(streaming_ds, max_dataset_size))
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
    shuffle=True,         # Important for training stability
    collate_fn=custom_collate_fn,
    pin_memory=True,      # Speeds up data transfer to GPU
)

# ----------------------------------------------------------------------
# 6. MODEL INSTANTIATION, LOSS, AND OPTIMIZER
# ----------------------------------------------------------------------

# init RNN
checkpoint_file = Path("rrn-lm.ckpt")
rnn = RNNLM(
    vocab_size=current_vocab_size,
    embed_dim=current_embed_dim,
    vectors=vectors_for_init,
    num_parallel_layers=17
)

if checkpoint_file.exists():
    # checkpoint_file.unlink() # delete the checkpoint by un-commenting this line
    rnn.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))

# Send it to the device
rnn = rnn.to(DEVICE)

# 1. Define the Loss Criterion
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion = criterion.to(DEVICE)

# 2. Define the Optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)


# ----------------------------------------------------------------------
# BLOCK 7: THE TRAINING LOOP
# ----------------------------------------------------------------------
rich.print("[bold green]STARTING TRAINING...[/bold green]")

# Pre-calculate total characters in test set for BPC calculation
total_test_chars = sum(
    count_chars(sample, key="Text" if DATASET_OPTION == 2 else "text")
    for sample in dataset["test"]
)
total_test_bytes = sum(
    count_bytes(sample, key="Text" if DATASET_OPTION == 2 else "text")
    for sample in dataset["test"]
)
total_test_bits = total_test_bytes * 8 # assuming 8 bits per byte

# Prepare the full test text for compression baselines
test_text = "".join(
    sample["Text" if DATASET_OPTION == 2 else "text"].replace(" ", "").replace(",", "")
    for sample in dataset["test"]
)
test_text_bytes = test_text.encode("utf-8")

start_time = time.time()
accumulated_testing_duration = 0.0 # to track total testing time


train_loss_per_epoch = []
test_loss_per_epoch = []

for epoch in range(NUM_EPOCHS):
    rich.print(f"[bold blue]-- Epoch {epoch+1}/{NUM_EPOCHS} --[/bold blue]")
    
    # Set model to training mode
    rnn.train() 
    total_loss = 0
    total_tokens = 0
    
    for step, token_ids_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        
        token_ids_batch = token_ids_batch.to(DEVICE) 

        # 1. Zero the gradients 
        optimizer.zero_grad() 

        # 2. Forward Pass: (B, T) -> (B, T, V). 
        logits = rnn(token_ids_batch)
        
        # 3. Prepare Logits (B * (T-1), V) and Targets (B * (T-1))
        logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = token_ids_batch[:, 1:].reshape(-1)
        
        # 4. Calculate Loss
        loss = criterion(logits_flat, targets)
        
        # weight the loss by the number of non-PAD tokens
        batch_tokens = (targets != PAD_IDX).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # 5. Backward Pass and Optimization
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / total_tokens
    train_loss_per_epoch.append(avg_loss)

    # TESTING PHASE at the end of each epoch
    # Do not time testing phase
    start_testing_time = time.time()

    rnn.eval()
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
            logits = rnn(token_ids_batch)
            
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


# ------------------------------------------------------------------
# Compression baselines on TEST corpus
# ------------------------------------------------------------------

# GZIP
gzip_compressed = gzip.compress(test_text_bytes)
gzip_bytes = len(gzip_compressed)
gzip_bits = gzip_bytes * 8
gzip_bpb = gzip_bits / total_test_bytes
gzip_ratio = gzip_bits / total_test_bits  # equivalente: gzip_bytes / total_test_bytes

# BZ2
bz2_compressed = bz2.compress(test_text_bytes)
bz2_bytes = len(bz2_compressed)
bz2_bits = bz2_bytes * 8
bz2_bpb = bz2_bits / total_test_bytes
bz2_ratio = bz2_bits / total_test_bits

# LZMA
lzma_compressed = lzma.compress(test_text_bytes)
lzma_bytes = len(lzma_compressed)
lzma_bits = lzma_bytes * 8
lzma_bpb = lzma_bits / total_test_bytes
lzma_ratio = lzma_bits / total_test_bits

rich.print("\n[bold cyan]Compression baselines on TEST corpus (preprocessed text):[/bold cyan]")
rich.print(f"  Raw size: {total_test_bytes} bytes")
rich.print(f"  GZIP: {gzip_bytes} bytes | {gzip_bpb:.4f} bits/byte | ratio: {gzip_ratio:.4f}")
rich.print(f"  BZ2 : {bz2_bytes} bytes | {bz2_bpb:.4f} bits/byte | ratio: {bz2_ratio:.4f}")
rich.print(f"  LZMA: {lzma_bytes} bytes | {lzma_bpb:.4f} bits/byte | ratio: {lzma_ratio:.4f}\n")


# ----------------------------------------------------------------------
# TRAINING SUMMARY REPORT
# ----------------------------------------------------------------------
rich.print("\n" + "="*50)
rich.print("[bold magenta]TRAINING SUMMARY REPORT (RNN)[/bold magenta]")
rich.print(f"  Total Training Time: [yellow]{total_time:.2f} seconds[/yellow]")
rich.print(f"  Final Avg Training Loss: [yellow]{avg_loss:.4f}[/yellow]")
rich.print("="*50 + "\n")

# ----------------------------------------------------------------------
# TRAINING/TESTING ERROR VS EPOCHS PLOTTING
# ----------------------------------------------------------------------

fig,ax = plt.subplots(figsize=(8,6), dpi=300)
ax.plot(range(1, NUM_EPOCHS+1), train_loss_per_epoch, marker='.', linestyle='-')
ax.plot(range(1, NUM_EPOCHS+1), test_loss_per_epoch, marker='.', linestyle='-')
ax.legend(['Training Loss', 'Testing Loss'], fontsize=10)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Average Loss per Token", fontsize=12)
ax.grid(True)
# plt.show()



# ----------------------------------------------------------------------
# BLOCK 8: TESTING AND OBTAINING METRICS
# ----------------------------------------------------------------------
rich.print("[bold green]STARTING TESTING...[/bold green]")

# Set model to evaluation mode
rnn.eval()
total_test_loss = 0
total_test_tokens = 0

# ... (Testing Loop body remains unchanged) ...
with torch.no_grad():
    for step, token_ids_batch in enumerate(tqdm(
        torch.utils.data.DataLoader(
            dataset["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        ),
        desc="Testing"
    )):
        
        token_ids_batch = token_ids_batch.to(DEVICE) 

        # Forward Pass
        logits = rnn(token_ids_batch)
        
        # Prepare Logits and Targets
        logits_flat = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = token_ids_batch[:, 1:].reshape(-1)
        
        # Calculate Loss
        loss = criterion(logits_flat, targets)

        # weight the loss by the number of non-PAD tokens
        batch_tokens = (targets != PAD_IDX).sum().item()
        total_test_loss += loss.item() * batch_tokens
        total_test_tokens += batch_tokens

avg_test_loss = total_test_loss / total_test_tokens
rich.print(f"[bold green]Testing Complete. Average Test Loss: {avg_test_loss:.4f}[/bold green]")

eval_sentence = "A dog is an amazing animal with a heart of a true lifemate of men, and with many other qualities"


# ----------------------------------------------------------------------
# NLS AND SUBWORD FERTILITY METRICS
# ----------------------------------------------------------------------
rich.print("\n" + "="*50)
rich.print("[bold blue]STARTING TOKENIZER EVALUATION (NLS + SUBWORD METRICS)[/bold blue]")

# Count words (W) and characters (C)
nls_word_count = len(eval_sentence.split())
nls_char_count = len(eval_sentence.replace(" ", "").replace(",","")) 

# Reference Tokenizer (for NLS_ref)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

# A. Tokenize with the CURRENT Tokenizer
current_token_ids = batch_tokenize({"text": [eval_sentence]}, max_length=max_seq_size, tokenizer=current_tokenizer)['token_ids'][0]
non_padded_token_ids = torch.tensor([id for id in current_token_ids if id != PAD_IDX])
actual_tokens = get_tokens_from_ids(non_padded_token_ids, current_tokenizer)
token_count = len(actual_tokens)

rich.print(f"Tokenizer evaluation on '{eval_sentence}'")
rich.print(f"Word Count: [green]{nls_word_count}[/green] | Char Count: [green]{nls_char_count}[/green]")

# NLS_W (Subword Fertility)
nls_w = token_count / nls_word_count
rich.print(f"  NLS_w (Tokens per Word): [green]{nls_w:.4f}[/green]")

# NLS_C
nls_c = token_count / nls_char_count
rich.print(f"  NLS_c (Tokens per Character): [green]{nls_c:.4f}[/green]")

# NLS (Reference)
t5_tokenization = t5_tokenizer(
    eval_sentence,
    max_length=100,
    padding=False,
    truncation=True,
    return_attention_mask=False
)
t5_token_ids = t5_tokenization["input_ids"]
t5_token_count = len(t5_token_ids)
nls_ref = token_count / t5_token_count
rich.print(f"  NLS (Reference T/R): [green]{nls_ref:.4f}[/green] (Efficiency vs. t5)")


# Subword Fertility and PCW
fertility, pcw = calculate_subword_metrics(eval_sentence, current_tokenizer, actual_tokens)

rich.print("\n[bold cyan]Subword Metrics:[/bold cyan]")
rich.print(f"  Subword Fertility (Ideal: 1.0): [green]{fertility:.4f}[/green]")
rich.print(f"  Proportion of Continued Words (Ideal: 0.0): [green]{pcw:.4f}[/green]")

rich.print("[bold blue]Tokenizer Evaluation Complete.[/bold blue]")
rich.print("="*50 + "\n")


# ----------------------------------------------------------------------
# MEMORY SIZE BREAKDOWN
# ----------------------------------------------------------------------
print_rnn_size_breakdown(rnn)

# # ----------------------------------------------------------------------
# # BLOCK 8: GRADIENT VISUALIZATION TEST (Diagnostic Check)
# # ----------------------------------------------------------------------
# rich.print("[bold yellow]STARTING GRADIENT VISUALIZATION TEST...[/bold yellow]")

# # Reset gradients 
# rnn.zero_grad()
# rnn.train()

# # 1. Get DUMMY token ids for the specific visualization size (10x10)
# T_SIZE = 10
# # Ensure token IDs are within the current vocabulary range
# token_ids_test = torch.randint(low=0, high=current_vocab_size, size=(T_SIZE, T_SIZE))
# token_ids_test = token_ids_test.to(DEVICE)

# # 2. Run forward pass with retain_ws=True to save embeddings for gradient extraction
# logits = rnn(token_ids_test, retain_ws=True)

# # 3. Compute the specific loss for the visualization test:
# loss_locations = torch.arange(0, T_SIZE)[:, None, None].expand(T_SIZE, 1, logits.shape[-1])
# loss_locations = loss_locations.to(DEVICE)
# loss_test = logits.gather(index=loss_locations, dim=1).mean()

# # 4. Backward pass to retrieve the gradients
# loss_test.backward()
# grad_magnitude = rnn.ws.grad.norm(dim=2)
# rnn.ws = None # Clean up

# # 5. Visualize the gradient
# grad_magnitude[grad_magnitude==0] = -math.inf 
# grad_magnitude = grad_magnitude.detach().cpu().numpy()
# plt.figure(figsize=(8, 6))
# plt.imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
# plt.colorbar()
# plt.grid(False)
# plt.xlabel("$t$ (input)")
# plt.ylabel("$t'$ (loss)")
# plt.title("Magnitude of the gradient w.r.t. $\mathbf{w}_{1:T}$")
# plt.show()

# # ----------------------------------------------------------------------
# # BLOCK 9: ATTENTION MAP VISUALIZATION (Conceptual Check)
# # ----------------------------------------------------------------------
# # Since the `RNNLM` does not use attention, this section is for conceptual comparison.

# # Helper function to convert IDs back to tokens (handles all tokenizers)
# def get_tokens_from_ids(token_ids, tokenizer):
#     if isinstance(tokenizer, tokenizers.Tokenizer):
#         # Assuming original tokenizers lib object has a proper vocabulary object
#         if hasattr(tokenizer, 'get_vocabulary'):
#             vocab = tokenizer.get_vocabulary()
#             return [vocab[i] for i in token_ids]
#         else:
#             return [str(i) for i in token_ids] # Fallback
#     elif isinstance(tokenizer, PreTrainedTokenizer):
#         # Use transformers decoder
#         return tokenizer.convert_ids_to_tokens(token_ids.tolist())
#     return [str(i) for i in token_ids]

# # Instantiate new embeddings for visualization (not part of the model)
# embeddings = torch.nn.Embedding(current_vocab_size, current_embed_dim)
# if TOKEN == 1:
#     embeddings.weight.data = glove_vectors
# embeddings.weight.requires_grad = False

# # get a more natural sentence
# sentence = "Masked attention allows implementing dependency constrains between inputs and outputs"
# # Tokenize the sentence using the current tokenizer
# # Use the batch_tokenize logic for consistency and get the first result
# token_ids = torch.tensor(batch_tokenize({"text": [sentence]}, max_length=15, tokenizer=current_tokenizer)['token_ids'][0])

# tokens = get_tokens_from_ids(token_ids, current_tokenizer)
# vectors = embeddings(token_ids)


# """
#     The plot_attention_map function is purely a visualization utility used to create and format a heatmap
#     that represents attention weights. It takes the numerical attention scores and transforms them into
#     an interpretable diagram.
# """
# def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
#     if ax is None:
#         fig, ax = plt.subplots(figsize = (4,2), dpi=300) 
#     else:
#         fig = plt.gcf()
#     im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
#     ax.grid(False)
    
#     # Set font size for the Y-axis labels
#     ax.set_yticks(np.arange(len(queries_labels)))
#     ax.set_yticklabels(queries_labels, fontsize=4) 
    
#     # Set font size for the X-axis labels
#     ax.set_xticks(np.arange(len(keys_labels)))
#     ax.set_xticklabels(keys_labels, fontsize=4) 
    
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     if print_values:
#         for i in range(len(queries_labels)):
#             for j in range(len(keys_labels)):
#                 text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
#                             ha="center", va="center", color="w", fontsize=4)

#     if color_bar:
#       cbar = fig.colorbar(im, fraction=0.02, pad=0.04, shrink=0.7)
#       cbar.ax.tick_params(labelsize=4)
      
#     fig.tight_layout()


# """
#     The masked_attention function implements a simplified version of the scaled dot-product attention mechanism
#     used in Transformers, with the crucial addition of an optional masking operation
# """
# def masked_attention(Q, K, V, tau=None, mask=None):
#     """A simple masked attention layer"""
#     if tau is None:
#         tau = math.sqrt(float(Q.shape[-1]))
#     assert Q.shape[-1] == K.shape[-1]
#     assert K.shape[0] == V.shape[0]
#     attention_map = Q @ K.T / tau
#     if mask is not None:
#         attention_map = mask + attention_map
#     attention_weights = attention_map.softmax(dim=1)
#     return torch.einsum("qk, kh -> qh", attention_weights, V), attention_weights


# # EXERCISE: Implement the masks corresponding to each factorization
# T = len(token_ids)
# masks = {
#     "left-to-right": torch.triu(torch.ones(T, T), diagonal=0),
#     "bidirectional": torch.zeros(T, T),
#     "right-to-left": torch.tril(torch.ones(T, T), diagonal=0)
# }
# for key in masks.keys():
#     if masks[key] is not None:
#         # Convert 1s to -inf and 0s to 0 to mask logits before softmax
#         masks[key] = torch.where(masks[key] == 0, 0.0, -math.inf)

# # visualized the log of the masked attention map
# fig, axes = plt.subplots(ncols=1+len(masks), figsize = (8,6), sharex=False, sharey=False, dpi=300)

# # plot the gradient map from the RNN LM
# axes.flat[0].imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
# axes.flat[0].set_xlabel("$t$ (input)",fontsize=4)
# axes.flat[0].set_ylabel("$t'$ (output)",fontsize=4)
# axes.flat[0].grid(False)
# axes.flat[0].set_title("Gradient map (RNN LM)",fontsize=6)
# axes.flat[0].tick_params(axis='both', which='major', labelsize=4)

# # plot the attention map
# for ax, (mask_name, mask) in zip(axes.flat[1:], masks.items()):
#     if mask is not None:
#         # Use zero matrix for Q, K, V as we only care about the mask effect on attention map
#         H, attention_map_masked = masked_attention(vectors, vectors, vectors, mask=mask)

#         # Use log() to better visualize the zero/masked areas
#         plot_attention_map(attention_map_masked.log(), tokens, tokens, ax=ax, color_bar=False)
#     ax.set_title(f"Attention map {mask_name}",fontsize=6)
# plt.tight_layout()
# plt.show()