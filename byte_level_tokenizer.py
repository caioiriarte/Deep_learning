from typing import List


def byte_level_tokenize(text: str) -> List[int]:
    """
    Perform byte-level tokenization on a Unicode string.

    This converts the input text into a list of integers in the range [0, 255],
    where each integer is a raw byte from the UTF-8 encoded form of the text.

    Parameters
    ----------
    text : str
        Input text (Unicode string).

    Returns
    -------
    tokens : List[int]
        List of byte IDs (one per byte), each in the range [0, 255].
    """
    # Encode the string to bytes using UTF-8
    byte_sequence = text.encode("utf-8")

    # Convert each byte (0–255) to an integer token ID
    tokens = list(byte_sequence)

    return tokens


def byte_level_detokenize(token_ids: List[int]) -> str:
    """
    Convert a list of byte-level token IDs back into a Unicode string.

    Parameters
    ----------
    token_ids : List[int]
        List of integers, each representing a byte in [0, 255].

    Returns
    -------
    text : str
        Decoded Unicode string.
    """
    # Convert the list of integers back to a bytes object
    byte_sequence = bytes(token_ids)

    # Decode the bytes into a string using UTF-8
    text = byte_sequence.decode("utf-8")

    return text


if __name__ == "__main__":
    # Example usage
    original_text = "Ciao mondo! 😄"

    # Tokenize at byte level
    tokens = byte_level_tokenize(original_text)
    print("Byte-level tokens:", tokens)

    # Detokenize back to string
    recovered_text = byte_level_detokenize(tokens)
    print("Recovered text:", recovered_text)
