import torch
from gpt import GPTLanguageModel, GPTConfig, n_vocab, encode, decode
from typing import Dict

config = GPTConfig(
    n_vocab=n_vocab,
    batch_size=64,
    block_size=256,
    n_embed=384,
    n_head=6,
    n_layer=6,
    lr=3e-4
)

model = GPTLanguageModel(config)

print(f"{model._get_model_size():.1f} MB model")
print(f"{(model._get_n_params() / 1e6):.2f} million parameters")
model.load_state_dict(torch.load("model.pt", map_location='cpu', weights_only=True))


