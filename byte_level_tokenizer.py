class ByteLevelTokenizer:
    """
    Simple byte-level tokenizer, compatible with the HuggingFace interface
    that (tokenizer(...)-> dict with 'input_ids', 'attention_mask').

    - Vocab: 0..255 = bytes
             256    = <|pad|>
             257    = <|eos|>
    """
    def __init__(self):
        self.byte_vocab_size = 256
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.pad_token_id = 256
        self.eos_token_id = 257
        self.vocab_size = self.byte_vocab_size + 2  # 0..257

    def _encode_single(self, text, max_length=None, add_eos=True):
        # bytes in UTF-8
        b = text.encode("utf-8")
        ids = list(b)  # each byte is already an int in [0,255]

        if add_eos:
            ids.append(self.eos_token_id)

        # truncation
        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def __call__(
        self,
        texts,
        max_length=None,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    ):
        """
        Mimics HuggingFace tokenizers:
        - texts: str or list of str
        - returns dict with 'input_ids' and 'attention_mask'
        """
        if isinstance(texts, str):
            texts = [texts]

        all_ids = []
        for t in texts:
            ids = self._encode_single(t, max_length=max_length)
            all_ids.append(ids)

        # padding
        if padding:
            # if max_length is not given, use the longest sequence
            if max_length is None:
                max_length = max(len(x) for x in all_ids)

            padded_ids = []
            attn_mask = []
            for ids in all_ids:
                if truncation:
                    ids = ids[:max_length]
                pad_len = max_length - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                # 1 for real tokens, 0 for pad
                attn_mask.append([1] * len(ids) + [0] * pad_len)
        else:
            padded_ids = all_ids
            attn_mask = [[1] * len(ids) for ids in all_ids]

        return {
            "input_ids": padded_ids,
            "attention_mask": attn_mask if return_attention_mask else None,
        }

    # so that your function get_tokens_from_ids can work
    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            if i == self.pad_token_id:
                tokens.append(self.pad_token)
            elif i == self.eos_token_id:
                tokens.append(self.eos_token)
            elif 0 <= i < 256:
                # convert the byte to a readable pseudo-token
                tokens.append(f"<b{int(i):03d}>")
            else:
                tokens.append("<unk>")
        return tokens
