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
TOKEN = 3

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
    current_embed_dim = 300 
    PAD_IDX = current_tokenizer.pad_token_id or current_tokenizer.unk_token_id

elif TOKEN == 3:
    rich.print("[bold red]Using TOKENIZER 3: HuggingFace BPE (GPT-2)[/bold red]")
    tokenizer_name = "gpt2"
    current_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    current_vocab_size = current_tokenizer.vocab_size
    current_embed_dim = 300 
    # GPT-2 has no explicit pad token by default; using EOS as a common choice for LMs
    current_tokenizer.pad_token = current_tokenizer.eos_token 
    PAD_IDX = current_tokenizer.pad_token_id

else:
    raise ValueError(f"Invalid TOKEN value: {TOKEN}. Must be 1, 2, or 3.")

# ----------------------------------------------------------------------
# 2. CONSTANTS AND DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 8
max_dataset_size = 100000
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


def batch_tokenize(
    batch: List[Dict[str, Any]], 
    max_length=max_seq_size, 
    # Update type hint to include both base classes
    tokenizer: Union[tokenizers.Tokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
    key:str="text"
) -> Dict[str, Any]:
    texts = batch[key]
    
    if isinstance(tokenizer, tokenizers.Tokenizer):
        # ... (logic for TOKEN=1)
        encodings = tokenizer.encode_batch(texts)
        return {"token_ids": [x.ids[:max_length] for x in encodings]}
    
    # Use a check that covers both PreTrainedTokenizer and PreTrainedTokenizerFast
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        # Handle Hugging Face `transformers` output (TOKEN=2 and 3)
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length", 
            truncation=True,
            return_attention_mask=False
        )
        return {"token_ids": encodings["input_ids"]}
        
    else:
        # This branch should now only catch actual unknown types
        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

# ----------------------------------------------------------------------
# 4. RNNLM MODEL DEFINITION (Modified for Flexibility)
# ----------------------------------------------------------------------

class RNNLM(torch.nn.Module):
    """A simple implementation of a language model using RNNs."""
    def __init__(self, vocab_size: int, embed_dim: int, vectors: torch.Tensor = None):
        super().__init__()
        
        # 1. Embeddings Layer
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        
        # # Initialize weights with GloVe ONLY if TOKEN=1 and size matches
        # if vectors is not None and vectors.shape == (vocab_size, embed_dim):
        #     rich.print("[bold cyan]Initializing Embeddings and Proj with GloVe Vectors.[/bold cyan]")
        #     self.embeddings.weight.data = vectors
        # else:
        #     rich.print("[bold cyan]Randomly Initializing Embeddings/Proj (GloVe not used).[/bold cyan]")

        # 2. LSTM Layer 
        self.rnn = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
        )

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

        # call the RNN: w_{-1:T-1} -> h{1:T}
        hidden_states, _ = self.rnn(ws_shifted)

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
    num_proc=2,           # To parallelize 
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
    vectors=vectors_for_init
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

    # Testing phase at the end of each epoch
    rnn.eval()
    total_test_loss = 0
    total_test_tokens = 0
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
            
            # Calculate Loss
            loss = criterion(logits_flat, targets)

            # weight the loss by the number of non-PAD tokens
            batch_tokens = (targets != PAD_IDX).sum().item()
            total_test_loss += loss.item() * batch_tokens
            total_test_tokens += batch_tokens

    avg_test_loss = total_test_loss / total_test_tokens
    test_loss_per_epoch.append(avg_test_loss)

    rich.print(f"[bold green]Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f} | Average Test Loss: {avg_test_loss:.4f}[/bold green]")

# ----------------------------------------------------------------------
# BLOCK 8: TRAINING ERROR VS BATCHES PLOTTING
# ----------------------------------------------------------------------

fig,ax = plt.subplots(figsize=(8,6), dpi=300)
ax.plot(range(1, NUM_EPOCHS+1), train_loss_per_epoch, marker='.', linestyle='-')
ax.plot(range(1, NUM_EPOCHS+1), test_loss_per_epoch, marker='.', linestyle='-')
ax.legend(['Training Loss', 'Testing Loss'], fontsize=10)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Average Loss per Token", fontsize=12)
ax.grid(True)
plt.show()


# ----------------------------------------------------------------------
# BLOCK 9: GRADIENT VISUALIZATION TEST (Diagnostic Check)
# ----------------------------------------------------------------------
rich.print("[bold yellow]STARTING GRADIENT VISUALIZATION TEST...[/bold yellow]")

# Reset gradients 
rnn.zero_grad()
rnn.train()

# 1. Get DUMMY token ids for the specific visualization size (10x10)
T_SIZE = 10
# Ensure token IDs are within the current vocabulary range
token_ids_test = torch.randint(low=0, high=current_vocab_size, size=(T_SIZE, T_SIZE))
token_ids_test = token_ids_test.to(DEVICE)

# 2. Run forward pass with retain_ws=True to save embeddings for gradient extraction
logits = rnn(token_ids_test, retain_ws=True)

# 3. Compute the specific loss for the visualization test:
loss_locations = torch.arange(0, T_SIZE)[:, None, None].expand(T_SIZE, 1, logits.shape[-1])
loss_locations = loss_locations.to(DEVICE)
loss_test = logits.gather(index=loss_locations, dim=1).mean()

# 4. Backward pass to retrieve the gradients
loss_test.backward()
grad_magnitude = rnn.ws.grad.norm(dim=2)
rnn.ws = None # Clean up

# 5. Visualize the gradient
grad_magnitude[grad_magnitude==0] = -math.inf 
grad_magnitude = grad_magnitude.detach().cpu().numpy()
plt.figure(figsize=(8, 6))
plt.imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
plt.colorbar()
plt.grid(False)
plt.xlabel("$t$ (input)")
plt.ylabel("$t'$ (loss)")
plt.title("Magnitude of the gradient w.r.t. $\mathbf{w}_{1:T}$")
plt.show()

# ----------------------------------------------------------------------
# BLOCK 10: ATTENTION MAP VISUALIZATION (Conceptual Check)
# ----------------------------------------------------------------------
# Since the `RNNLM` does not use attention, this section is for conceptual comparison.

# Helper function to convert IDs back to tokens (handles all tokenizers)
def get_tokens_from_ids(token_ids, tokenizer):
    if isinstance(tokenizer, tokenizers.Tokenizer):
        # Assuming original tokenizers lib object has a proper vocabulary object
        if hasattr(tokenizer, 'get_vocabulary'):
            vocab = tokenizer.get_vocabulary()
            return [vocab[i] for i in token_ids]
        else:
            return [str(i) for i in token_ids] # Fallback
    elif isinstance(tokenizer, PreTrainedTokenizer):
        # Use transformers decoder
        return tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return [str(i) for i in token_ids]

# Instantiate new embeddings for visualization (not part of the model)
embeddings = torch.nn.Embedding(current_vocab_size, current_embed_dim)
if TOKEN == 1:
    embeddings.weight.data = glove_vectors
embeddings.weight.requires_grad = False

# get a more natural sentence
sentence = "Masked attention allows implementing dependency constrains between inputs and outputs."
# Tokenize the sentence using the current tokenizer
# Use the batch_tokenize logic for consistency and get the first result
token_ids = torch.tensor(batch_tokenize({"text": [sentence]}, max_length=50, tokenizer=current_tokenizer)['token_ids'][0])

tokens = get_tokens_from_ids(token_ids, current_tokenizer)
vectors = embeddings(token_ids)


"""
    The plot_attention_map function is purely a visualization utility used to create and format a heatmap
    that represents attention weights. It takes the numerical attention scores and transforms them into
    an interpretable diagram.
"""
def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6), dpi=300) 
    else:
        fig = plt.gcf()
    im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
    ax.grid(False)
    
    # Set font size for the Y-axis labels
    ax.set_yticks(np.arange(len(queries_labels)))
    ax.set_yticklabels(queries_labels, fontsize=4) 
    
    # Set font size for the X-axis labels
    ax.set_xticks(np.arange(len(keys_labels)))
    ax.set_xticklabels(keys_labels, fontsize=4) 
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_ylabel("$\mathbf{Q}$", fontsize=7) 
    ax.set_xlabel("$\mathbf{K}$", fontsize=7) 

    if print_values:
        for i in range(len(queries_labels)):
            for j in range(len(keys_labels)):
                text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                            ha="center", va="center", color="w", fontsize=4)

    if color_bar:
      cbar = fig.colorbar(im, fraction=0.02, pad=0.04)
      cbar.ax.tick_params(labelsize=4)
      
    fig.tight_layout()


"""
    The masked_attention function implements a simplified version of the scaled dot-product attention mechanism
    used in Transformers, with the crucial addition of an optional masking operation
"""
def masked_attention(Q, K, V, tau=None, mask=None):
    """A simple masked attention layer"""
    if tau is None:
        tau = math.sqrt(float(Q.shape[-1]))
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[0] == V.shape[0]
    attention_map = Q @ K.T / tau
    if mask is not None:
        attention_map = mask + attention_map
    attention_weights = attention_map.softmax(dim=1)
    return torch.einsum("qk, kh -> qh", attention_weights, V), attention_weights


# EXERCISE: Implement the masks corresponding to each factorization
T = len(token_ids)
masks = {
    "left-to-right": torch.triu(torch.ones(T, T), diagonal=0),
    "bidirectional": torch.zeros(T, T),
    "right-to-left": torch.tril(torch.ones(T, T), diagonal=0)
}
for key in masks.keys():
    if masks[key] is not None:
        # Convert 1s to -inf and 0s to 0 to mask logits before softmax
        masks[key] = torch.where(masks[key] == 0, 0.0, -math.inf)

# visualized the log of the masked attention map
fig, axes = plt.subplots(ncols=1+len(masks), figsize = (16,6), sharex=False, sharey=False, dpi=300)

# plot the gradient map from the RNN LM
axes.flat[0].imshow(grad_magnitude, sns.color_palette("viridis", as_cmap=True))
axes.flat[0].set_xlabel("$t$ (input)",fontsize=4)
axes.flat[0].set_ylabel("$t'$ (output)",fontsize=4)
axes.flat[0].grid(False)
axes.flat[0].set_title("Gradient map (RNN LM)",fontsize=6)
axes.flat[0].tick_params(axis='both', which='major', labelsize=4)

# plot the attention map
for ax, (mask_name, mask) in zip(axes.flat[1:], masks.items()):
    if mask is not None:
        # Use zero matrix for Q, K, V as we only care about the mask effect on attention map
        H, attention_map_masked = masked_attention(vectors, vectors, vectors, mask=mask)

        # Use log() to better visualize the zero/masked areas
        plot_attention_map(attention_map_masked.log(), tokens, tokens, ax=ax, color_bar=False)
    ax.set_title(f"Attention map {mask_name}",fontsize=6)
plt.tight_layout()
plt.show()