import math
import torch
import torch.nn as nn
from common import PositionalEncoding, generate_causal_mask


class EncoderLayer(nn.Module):
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

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Masked self-attention
        self_attn_out, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.ln1(tgt + self.dropout(self_attn_out))

        # Cross-attention
        cross_attn_out, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.ln2(tgt + self.dropout(cross_attn_out))

        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.ln3(tgt + self.dropout(ffn_out))
        return tgt


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.src_pos = PositionalEncoding(d_model, max_len)
        self.tgt_pos = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self, 
        src_ids, 
        src_key_padding_mask=None
    ):
        x = self.src_embed(src_ids) * math.sqrt(self.d_model)
        x = self.src_pos(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(
        self,
        tgt_ids,
        memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        y = self.tgt_embed(tgt_ids) * math.sqrt(self.d_model)
        y = self.tgt_pos(y)
        y = self.dropout(y)

        seq_len = y.size(1)
        tgt_mask = generate_causal_mask(seq_len, device=y.device)

        for layer in self.decoder_layers:
            y = layer(
                y,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        logits = self.out_proj(y)
        return logits

    def forward(
        self,
        src_ids,
        tgt_ids,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        memory = self.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(
            tgt_ids,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits


if __name__ == "__main__":
    src_vocab, tgt_vocab = 800, 900
    model = Seq2SeqTransformer(src_vocab, tgt_vocab)

    batch, src_len, tgt_len = 2, 7, 6
    src_ids = torch.randint(0, src_vocab, (batch, src_len))
    tgt_ids = torch.randint(0, tgt_vocab, (batch, tgt_len))

    logits = model(src_ids, tgt_ids)
    print("Encoder–decoder logits:", logits.shape)  # [2, 6, tgt_vocab]
