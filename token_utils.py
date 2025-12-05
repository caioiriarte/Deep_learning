import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import math
from pathlib import Path
from tqdm import tqdm
import rich
from typing import List, Tuple, Optional
import seaborn as sns
import transformers
import tokenizers
import datasets
import zipfile
from huggingface_hub import hf_hub_download

sns.set_theme()

# define the device to use
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rich.print(f"Device: [red]{DEVICE}")

# control verbosity
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

# define support functions
def load_glove_vectors(filename = "glove.6B.50d.txt", max_lines: int = 250000) -> Tuple[List[str], torch.Tensor]:
    """
    Load the GloVe vectors, limiting the total number of words loaded.
    See: `https://github.com/stanfordnlp/GloVe`
    """
    path = Path(hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.6B.zip"))
    target_file = path.parent / filename
    
    # --- File Extraction Logic (Unchanged) ---
    if not target_file.exists():
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(path.parent)

        if not target_file.exists():
            print(f"Available files:")
            for p in path.parent.iterdir():
                print(p)
            raise ValueError(f"Target file `{target_file.name}` can't be found. Check if `{filename}` was properly downloaded.")

    # --- Parsing with Line Limit (Modified) ---
    vocabulary = []
    vectors = []
    
    # We first count the lines for accurate tqdm progress
    total_lines = sum(1 for line in open(target_file, "r", encoding="utf8"))
    
    with open(target_file, "r", encoding="utf8") as f:
        # Use min() to ensure tqdm's total is the maximum of the actual lines or the requested max_lines
        for i, l in enumerate(tqdm(f.readlines(), 
                                   desc=f"Parsing {target_file.name} (Max: {max_lines})...", 
                                   total=min(total_lines, max_lines))):
            
            # Stop condition: if we have read max_lines, break the loop
            if i >= max_lines:
                break
                
            word, *vector = l.split()
            vocabulary.append(word)
            vectors.append(torch.tensor([float(v) for v in vector]))
            
    vectors = torch.stack(vectors)
    return vocabulary, vectors


# prepare data for the later cells
glove_vocabulary, glove_vectors = load_glove_vectors()
rich.print(f"glove_vocabulary: type={type(glove_vocabulary)}, length={len(glove_vocabulary)}")
rich.print(f"glove_vectors: type={type(glove_vectors)}, shape={glove_vectors.shape}, dtype={glove_vectors.dtype}")

# add special tokens
special_tokens = ['<|start|>', '<|unknown|>', '<|pad|>']
glove_vocabulary = special_tokens + glove_vocabulary
glove_vectors = torch.cat([torch.randn_like(glove_vectors[:len(special_tokens)]), glove_vectors])

# tokenizer for GloVe
glove_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab={v:i for i,v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>"))
glove_tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
glove_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()


 # Example sentence with rare English words and non-english words
sentence = "It is jubilating to see how élégant my horse has became"
rich.print(f"Input sentence: [bold blue]`{sentence}`")

# Define multiple tokenizers
tokenizer_ids = {
    "Word-level": glove_tokenizer,
    "WordPiece": "bert-base-cased",
    "BPE": "distilgpt2",
    "Character-level":  "google/byt5-small",
    }

# iterate through the tokenizers and decode the input sentences
for tokenizer_name, tokenizer in tokenizer_ids.items():
    # intialize the tokenizer (either)
    if isinstance(tokenizer, str):
        # init a `transformers.PreTrainedTokenizerFast`
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        vocab_size = tokenizer.vocab_size
    else:
        # use the provided `tokenizers.Tokenizer``
        vocab_size = tokenizer.get_vocab_size()

    # Tokenize
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if isinstance(token_ids, tokenizers.Encoding):
        token_ids = token_ids.ids

    # Report
    rich.print(f"[red]{tokenizer_name}[/red]: sentence converted into {len(token_ids)} tokens (vocabulary: {vocab_size} tokens)")
    rich.print(f"Tokens:\n{[tokenizer.decode([t]) for t in token_ids]}")
    rich.print(f"Token ids:\n{[t for t in token_ids]}")


hdim = 5 # embedding dimension
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased") # tokenizer
sentence = "Hello World!" # input text
embeddings = torch.randn((tokenizer.vocab_size, hdim)) # embedding matrix
rich.print(f"Embeddings (shape): {embeddings.shape}")
token_ids = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0]
rich.print(f"Tokens ids (shape): {token_ids.shape}")
vectors =  torch.nn.functional.one_hot(token_ids, tokenizer.vocab_size).float() @ embeddings # equivalent to a `nn.Linear` layer
rich.print(f"Vectors (shape): {vectors.shape}")
rich.print(f"List of tokens and their corresponding vectors:")
for t,v in zip(token_ids, vectors):
    token_info = f"[blue]{tokenizer.decode(t):5}[/blue] (token id: {t:4})"
    rich.print(f" * {token_info} -> {v}")

# NB: in practice, we use the simpler interface `torch.nn.Embedding``
# embeddings = torch.nn.Embedding(tokenizer.vocab_size, hdim)
# vectors = embeddings(token_ids)


def word2vec(
        word: str,
        vocabulary:List[str],
        vectors: torch.Tensor
    ) -> Optional[torch.Tensor]:
    """Convert a word into a vector"""
    word = word.lower()
    if word in vocabulary:
        word_idx = vocabulary.index(word)
        return vectors[word_idx]
    else:
        return None

def vec2words(
        vec: torch.Tensor,
        k=5,
        *,
        vocabulary:List[str],
        vectors: torch.Tensor,
        exclude_vecs: List[torch.Tensor] = None,
    ) -> Tuple[List[str], torch.Tensor]:
    """Retrieve the nearest word neighbours for an input vector"""

    # compute the similarity between `vec`and all the vectors in `glove_vectors`
    similarity = vectors @ vec

    # potentially filter out some vocabulary entries
    if exclude_vecs is not None and len(exclude_vecs):
        mask = None
        for e in exclude_vecs:
            mask_ = (vectors == e[None, :]).all(dim=1)
            if mask is None:
                mask = mask_
            else:
                mask |= mask_
        similarity.masked_fill_(mask=mask, value=-math.inf)

    # return the ids of the nearesrt neighbours given the similarity
    nearest_neighbour_ids = torch.argsort(-similarity)[:k]

    # retrieve the corresponding words in the `vocabulary``
    return [vocabulary[idx] for idx in nearest_neighbour_ids], similarity[nearest_neighbour_ids]

# register the vocab and vectors args
glove_args = {'vocabulary':glove_vocabulary, 'vectors':glove_vectors}

# Nearest neighbours
rich.print("[red]Nearest neighbour search:")
for word in ["king", "queen", "dog", "France"]:
    rich.print(f'Nearest neighbours of the word "{word}":')
    word_vec = word2vec(word, **glove_args)
    words, similarities = vec2words(word_vec, k=5, **glove_args, exclude_vecs=[word_vec])
    rich.print(f"Words: {words}")
    rich.print(f"Similarities: {similarities}")

# Word analogies
rich.print("\n[red]Vector arithmetic:")
cases = [
    [("+", "king"), ("-", "man"), ("+", "woman")],
    [("+", "denmark"), ("-", "france"), ("+", "paris")],
    [("+", "pakistan"), ("-", "belgium"), ("+", "brussels")],
]
for operations in cases:
    # current location in the vector space
    location = 0
    rich.print(f"Vector Translation: [blue]0 {' '.join(f'{d} {v}' for d,v in operations)} = ")
    for sign, word in operations:
        # retrieve the `vec(word)``
        vec = word2vec(word, **glove_args)
        if vec is None:
            raise ValueError(f"Unknown word `{word}`")

        # parse the direction (+/-)
        direction = {"+": 1, "-": -1}[sign]

        # apply the vector transform to the current location
        location  +=  direction * vec

    # return the nearest neighbours of the end location
    exclude_list = [word2vec(w, **glove_args) for _, w in operations]
    words, similarities = vec2words(location, k=5, exclude_vecs=exclude_list, **glove_args)
    rich.print(f"Words: {words}")
    rich.print(f"Similarities: {similarities}")