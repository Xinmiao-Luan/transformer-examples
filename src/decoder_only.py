import math
import torch
import torch.nn as nn
from common import PositionalEncoding, generate_causal_mask


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.ln1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class DecoderOnlyLM(nn.Module):
    """
    GPT-style decoder-only language model.
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_layers=6,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len] (1=keep, 0=pad)
        """
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        x = self.dropout(x)

        seq_len = x.size(1)
        causal_mask = generate_causal_mask(seq_len, device=x.device)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        logits = self.lm_head(x)     # [batch, seq_len, vocab_size]
        return logits


if __name__ == "__main__":
    vocab_size = 1000
    model = DecoderOnlyLM(vocab_size)

    batch, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits = model(input_ids)
    print("Decoder-only logits:", logits.shape)  # [2, 10, vocab_size]
