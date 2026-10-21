"""Treina o Transformer mínimo da aula (inverter sequências de dígitos).

Uso:  python train_mini_transformer.py            # 6000 passos (~7 min em CPU de 2 núcleos)
      python train_mini_transformer.py --steps 2000   # já chega perto de 100%

Gera mini_transformer.pt, que o notebook 01 carrega para fazer só inferência.
O código das classes é o MESMO do notebook (copiado das mesmas células).
"""


import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# Vocabulário da tarefa de brinquedo: inverter sequências de dígitos
PAD, BOS, EOS = 0, 1, 2          # tokens especiais
DIGIT0 = 3                       # o dígito d vira o token DIGIT0 + d
VOCAB = DIGIT0 + 10              # 13 tokens no total


class PositionalEncoding(nn.Module):
    """Codificação posicional senoidal (Vaswani et al., 2017, seção 3.5).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)                                        # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))   # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                                     # (1, max_len, d_model)

    def forward(self, x):                                                               # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    q: (..., Tq, d_k)   k: (..., Tk, d_k)   v: (..., Tk, d_v)
    mask: booleano, True = pode olhar, False = bloqueado; broadcast para (..., Tq, Tk)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)                 # (..., Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)                               # cada linha soma 1
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Atenção multi-cabeça (seção 3.2.2): h cabeças em paralelo, cada uma com d_k = d_model / h."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model precisa ser divisível por n_heads"
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.last_attn = None                                         # guardamos os pesos para visualizar

    def split_heads(self, x):                                         # (B, T, D) -> (B, h, T, d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        q = self.split_heads(self.w_q(x_q))                           # Q vem de x_q
        k = self.split_heads(self.w_k(x_kv))                          # K e V vêm de x_kv
        v = self.split_heads(self.w_v(x_kv))                          # (self-attention: x_kv = x_q)
        out, weights = scaled_dot_product_attention(q, k, v, mask)    # (B, h, Tq, d_k)
        self.last_attn = weights.detach()
        B, _, Tq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.h * self.d_k)   # concatena as cabeças
        return self.w_o(out)


class EncoderBlock(nn.Module):
    """Self-attention + FFN, cada subcamada com residual e LayerNorm (pós-norm, como no paper)."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, src_mask)))      # LayerNorm(x + Sublayer(x))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderBlock(nn.Module):
    """Masked self-attention + cross-attention (olha para o encoder) + FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, src_mask, tgt_mask):
        y = self.norm1(y + self.drop(self.self_attn(y, y, tgt_mask)))            # Q, K, V = decoder (com máscara causal)
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, src_mask)))      # Q = decoder; K, V = saída do encoder
        y = self.norm3(y + self.drop(self.ffn(y)))
        return y


def make_pad_mask(seq):
    """(B, T) -> (B, 1, 1, T); True onde NÃO é padding."""
    return (seq != PAD).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T, device=None):
    """(1, 1, T, T); True na diagonal e abaixo dela (cada posição só vê o passado e ela mesma)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab)                          # projeção final para o vocabulário
        for p in self.parameters():                                   # inicialização Xavier, como no paper
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))     # embedding * sqrt(d_model) + PE
        for block in self.encoder:
            x = block(x, src_mask)
        return x                                                      # "memory": (B, Ts, d_model)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        y = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for block in self.decoder:
            y = block(y, memory, src_mask, tgt_mask)
        return y                                                      # (B, Tt, d_model)

    def forward(self, src, tgt):
        src_mask = make_pad_mask(src)                                                    # (B, 1, 1, Ts)
        tgt_mask = make_pad_mask(tgt) & make_causal_mask(tgt.size(1), tgt.device)        # (B, 1, Tt, Tt)
        memory = self.encode(src, src_mask)
        return self.out(self.decode(tgt, memory, src_mask, tgt_mask))                    # logits: (B, Tt, vocab)


def make_batch(batch_size, min_len=3, max_len=8):
    """Gera pares (src, tgt_in, tgt_out) para a tarefa 'inverter a sequência'.

    src     : d1 d2 d3 PAD PAD            (dígitos, com padding à direita)
    tgt_in  : BOS d3 d2 d1 PAD PAD        (entrada do decoder: deslocada uma posição)
    tgt_out : d3 d2 d1 EOS PAD PAD        (o que o decoder deve prever)
    """
    src = torch.full((batch_size, max_len), PAD, dtype=torch.long)
    tgt_in = torch.full((batch_size, max_len + 1), PAD, dtype=torch.long)
    tgt_out = torch.full((batch_size, max_len + 1), PAD, dtype=torch.long)
    for i in range(batch_size):
        n = random.randint(min_len, max_len)
        digits = torch.randint(0, 10, (n,)) + DIGIT0
        src[i, :n] = digits
        rev = digits.flip(0)
        tgt_in[i, 0] = BOS
        tgt_in[i, 1 : n + 1] = rev
        tgt_out[i, :n] = rev
        tgt_out[i, n] = EOS
    return src, tgt_in, tgt_out


def to_tokens(digits):
    return [DIGIT0 + int(d) for d in digits]


def to_text(tokens):
    """Converte tokens em string de dígitos, parando no EOS."""
    out = []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t >= DIGIT0:
            out.append(str(t - DIGIT0))
    return "".join(out)


@torch.no_grad()
def greedy_decode(model, src, max_new_tokens=16):
    """Decodificação gulosa autorregressiva: a cada passo, escolhe o token mais provável."""
    model.eval()
    src_mask = make_pad_mask(src)
    memory = model.encode(src, src_mask)                              # o encoder roda UMA vez
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_new_tokens):
        tgt_mask = make_causal_mask(ys.size(1), src.device)
        h = model.decode(ys, memory, src_mask, tgt_mask)              # decoder roda a cada passo
        next_tok = model.out(h[:, -1]).argmax(-1, keepdim=True)       # só a última posição importa
        ys = torch.cat([ys, next_tok], dim=1)
        finished |= next_tok.squeeze(1) == EOS
        if finished.all():
            break
    return ys[:, 1:]                                                  # remove o BOS


def exact_match_accuracy(model, n=500, min_len=3, max_len=8):
    """Fração de sequências em que a saída gerada é exatamente a sequência invertida."""
    src, _, tgt_out = make_batch(n, min_len, max_len)
    pred = greedy_decode(model, src, max_new_tokens=max_len + 2)
    ok = 0
    for p, t in zip(pred, tgt_out):
        ok += int(to_text(p) == to_text(t))
    return ok / n


def train(model, steps=6000, batch_size=128, lr=1e-3, warmup=300, log_every=500):
    """Loop de treino com teacher forcing. Em aula, NÃO rode: carregue os pesos prontos."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        for g in opt.param_groups:                                    # warmup linear, depois decaimento linear
            g["lr"] = lr * min(step / warmup, max(0.05, 1 - (step - warmup) / (steps - warmup + 1)))
        src, tgt_in, tgt_out = make_batch(batch_size)
        logits = model(src, tgt_in)                                   # (B, Tt, vocab)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            acc = exact_match_accuracy(model, n=300)
            print(f"passo {step:5d} | loss {loss.item():.4f} | acerto exato {acc:.1%} | {time.time() - t0:.0f}s")
    return model


CONFIG = dict(vocab=VOCAB, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, max_len=64)


def save_checkpoint(model, path="mini_transformer.pt"):
    torch.save({"config": CONFIG, "state_dict": model.state_dict()}, path)


def load_checkpoint(path="mini_transformer.pt"):
    ckpt = torch.load(path, map_location="cpu")
    model = Transformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treina o Transformer mínimo (inverter sequências de dígitos).")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--out", default="mini_transformer.pt")
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = Transformer(**CONFIG)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    train(model, steps=args.steps)
    model.eval()
    print(f"Acerto exato (comprimentos 3-8, exemplos novos): {exact_match_accuracy(model, n=1000):.1%}")
    save_checkpoint(model, args.out)
    print(f"Pesos salvos em {args.out}")
