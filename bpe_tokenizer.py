"""
Example: Train and use a Byte Pair Encoding (BPE) tokenizer
using the Hugging Face `tokenizers` library.
"""

from pathlib import Path

from bpe_tokenizer import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    input_files,
    vocab_size=8000,
    save_path="bpe_tokenizers/bpe_tokenizer.json",
):
    """
    Train a BPE tokenizer on one or more text files. Create vocabulary.

    Parameters
    ----------
    input_files : list[str]
        List of paths to text files. Each file should contain plain text,
        usually one sentence or document per line.
    vocab_size : int
        Target vocabulary size for the BPE tokenizer.
    save_path : str
        Where to save the trained tokenizer (JSON format).
    """
    # Create a tokenizer with a BPE model; define the unknown token
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Define how text is split into initial tokens before BPE merges
    # Here we use simple whitespace tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Configure the BPE trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Train the tokenizer on the provided files
    tokenizer.train(files=input_files, trainer=trainer)

    # Save tokenizer to disk in JSON format
    tokenizer.save(save_path)
    print(f"Saved BPE tokenizer to: {save_path}")


def load_and_tokenize(tokenizer_path: str, text: str):
    """
    Load a trained tokenizer from disk and tokenize a given string.

    Parameters
    ----------
    tokenizer_path : str
        Path to the JSON file containing the trained tokenizer.
    text : str
        Input sentence or text to tokenize.

    Returns
    -------
    tokens : list[str]
        List of subword tokens produced by the BPE tokenizer.
    ids : list[int]
        List of token IDs corresponding to the tokens.
    """
    # Load tokenizer from JSON file
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Encode the text
    encoded = tokenizer.encode(text)

    # Return tokens and their integer IDs
    return encoded.tokens, encoded.ids


if __name__ == "__main__":
    # Path to training text file
    # NOTE: Each line in this file should be plain text.
    train_file = Path("Dataset/fineweb2_subset/fineweb2_100.jsonl")

    # 1. Train the BPE tokenizer
    # NOTE: Change file name
    if train_file.exists():
        train_bpe_tokenizer(
            input_files=[train_file.as_posix()],
            vocab_size=8000,
            save_path="bpe_tokenizers/bpe_ita_tokenizer100.json", # Make reference to file size
        )
    else:
        print(f"Training file not found: {train_file}. Skipping training step.")

    # 2. Use the trained tokenizer (assuming it was already trained)
    tokenizer_path = "bpe_ita_tokenizer.json"
    if Path(tokenizer_path).exists():
        sample_text = "ciao mondo, come stai?"
        tokens, ids = load_and_tokenize(tokenizer_path, sample_text)

        print("\nInput text:")
        print(sample_text)

        print("\nBPE tokens:")
        print(tokens)

        print("\nToken IDs:")
        print(ids)
    else:
        print(f"Tokenizer file not found: {tokenizer_path}. "
              f"Train it first using the function above.")
