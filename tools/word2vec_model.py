import torch
import torch.nn as nn
import re

class Word2VecEmbedder(nn.Module):
    def __init__(self, weights: torch.Tensor, freeze: bool = True):
        super().__init__()
        vocab_size, dim = weights.shape
        self.dim        = dim

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = dim,
            padding_idx    = 0
        )
        self.embedding.weight = nn.Parameter(weights, requires_grad=not freeze)

    def forward(self, token_ids: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        embedded = self.embedding(token_ids)          # (batch, seq_len, dim)

        if weights is not None:
            weights  = weights.unsqueeze(-1)
            embedded = embedded * weights
            summed   = embedded.sum(dim=1)
            norm     = weights.sum(dim=1).clamp(min=1e-9)
            return summed / norm
        else:
            return embedded.mean(dim=1)

def preprocess_financial(text: str) -> str:
    text = text.lower()
    text = re.sub(r'₹\s?',         'inr ',      text)
    text = re.sub(r'(\d+)\s?cr\b', r'\1 crore', text)
    text = re.sub(r'(\d+)\s?l\b',  r'\1 lakh',  text)
    text = re.sub(r'[^a-z0-9\s]',  ' ',         text)
    text = re.sub(r'\s+',           ' ',         text).strip()
    return text

