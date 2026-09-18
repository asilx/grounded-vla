"""Local PaliGemma tokenizer, using the official π0 prompt convention."""

import numpy as np
import sentencepiece as spm


class Pi0Tokenizer:
    def __init__(self, path, max_length=48):
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        self.max_length = max_length

    def __call__(self, prompt):
        cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
        tokens = self.processor.encode(cleaned, add_bos=True) + self.processor.encode("\n")
        if len(tokens) > self.max_length:
            raise ValueError(f"Prompt exceeds π0 token budget ({len(tokens)} > {self.max_length})")
        mask = np.arange(self.max_length) < len(tokens)
        return np.asarray(tokens + [0] * (self.max_length - len(tokens)), np.int64), mask
