from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
import math

@dataclass
class ChessBertConfig:
    vocab_size: int = 290
    block_size: int = 72
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 512
    n_labels: int = 1972
    ffn_size: int = 2048

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.ffn_size, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class ChessBertModel(nn.Module):
    def __init__(self, config: ChessBertConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.encoder = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, config.n_labels)
        self.value_head = nn.Linear(config.n_embd, 1)

        self.init_weights() #Initiliaze weights according to BERT


    def forward(self, x, policy_targets=None, value_targets = None, return_losses = False):
        x = self.token_embeddings(x) + self.position_embeddings(torch.arange(x.size(1), device=x.device).unsqueeze(0)) # (B, T, n_embd)
        for block in self.encoder:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.policy_head(x[:, :1, :])
        value = self.value_head(x[:, 0, :])
        if policy_targets is not None and value_targets is not None:
            criterion_1 = nn.CrossEntropyLoss()
            criterion_2 = nn.MSELoss()
            policy_loss = criterion_1(logits.view(-1, self.config.n_labels), policy_targets.view(-1))
            value_loss = criterion_2(value.view(-1), value_targets.view(-1))
            loss = policy_loss + value_loss
            if return_losses:
                return loss, logits, value, policy_loss, value_loss
            return loss, logits, value
        else:
            return logits, value

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """

        for param in self.parameters():

            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=1.0)

        nn.init.normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=math.pow(self.config.n_embd, -0.5)
        )
        nn.init.normal_(
            self.position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.config.n_embd, -0.5)
        )