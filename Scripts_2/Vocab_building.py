import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from transformers import XLNetTokenizer


class TextProcessor:
    def __init__(self, pretrained_model="xlnet-base-cased"):
        # Load XLNet tokenizer
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_model)

        # Define special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # Add to tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.pad_token, self.bos_token, self.eos_token, self.unk_token
            ]
        })

        # Placeholder for vocab
        self.vocab = None

    def build_vocab(self, texts):
        """
        Build vocabulary from list of text strings
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            counter.update(tokens)

        # Sort tokens by frequency
        sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # Special tokens
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        # Create OrderedDict with specials first
        od = OrderedDict()
        for tok in specials:
            od[tok] = float("inf")  # big freq so they come first

        for tok, freq in sorted_by_freq:
            if tok not in od:  # avoid duplicates
                od[tok] = freq

        # Build vocab with torchtext
        self.vocab = vocab(od)

        # Set default index for unknown tokens
        self.vocab.set_default_index(self.vocab[self.unk_token])

        # 🔥 Check indices
        for sp in specials:
            print(f"{sp} → {self.vocab[sp]}")

    def numericalize(self, text):
        tokens = [self.bos_token] + self.tokenizer.tokenize(text) + [self.eos_token]
        return torch.tensor([self.vocab[token] for token in tokens])

    def collate_fn(self, batch):
        numericalized = [self.numericalize(text) for text in batch]
        return pad_sequence(numericalized, batch_first=True, padding_value=self.vocab[self.pad_token])


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world!",
        "This is an example using XLNet tokenizer.",
        "PyTorch and TorchText are powerful."
    ]

    processor = TextProcessor()
    processor.build_vocab(texts)

    # Numericalize single sentence
    # print("Numericalized:", processor.numericalize("Hello world!"))

    # Batch with padding
    batch = processor.collate_fn(texts)
    print("Batch shape:", batch.shape)
    print(batch)
