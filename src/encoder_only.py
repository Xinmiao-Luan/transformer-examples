# src/encoder_only.py
import torch
import torch.nn as nn
from common import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention (no causal mask in encoder)
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,  # True where PAD
        )
        x = self.ln1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class EncoderOnlyClassifier(nn.Module):
    """
    Minimal BERT-like encoder with a CLS token for classification.
    """
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_layers=4,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Classification head (use embedding at position 0 as [CLS])
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len], 1 = keep, 0 = pad
        """
        x = self.embed(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # Bool: True where pad

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        cls_repr = x[:, 0, :]               # [batch, d_model]
        logits = self.cls_head(cls_repr)    # [batch, num_classes]
        return logits


if __name__ == "__main__":
    # tiny demo
    vocab_size = 1000
    num_classes = 3
    model = EncoderOnlyClassifier(vocab_size, num_classes)

    batch, seq_len = 2, 8
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len)

    logits = model(input_ids, attention_mask)
    print("Encoder-only logits:", logits.shape)  # [2, 3]
