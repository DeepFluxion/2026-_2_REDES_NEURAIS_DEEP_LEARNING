"""Verifica se o ambiente da aula está pronto. Uso: python verificar_ambiente.py"""
import importlib
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # mesmo ajuste dos notebooks (Windows/conda)
import shutil
import sys

print(f"Python {sys.version.split()[0]}  ({sys.executable})")
falhas = 0

for nome in ["torch", "transformers", "numpy", "scipy", "matplotlib", "soundfile", "jupyterlab"]:
    try:
        m = importlib.import_module(nome)
        print(f"  OK   {nome:13s} {getattr(m, '__version__', '')}")
    except Exception as e:
        falhas += 1
        print(f"  FALTA {nome:12s} ({type(e).__name__})")

try:
    import torch

    x = torch.randn(2, 3) @ torch.randn(3, 2)
    print(f"  OK   operação com tensores | GPU disponível: {torch.cuda.is_available()}")
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (mesma ordem de imports dos notebooks)
    rnn = torch.nn.RNN(8, 8, batch_first=True)
    inp = torch.randn(1, 20, 8, requires_grad=True)
    rnn(inp)[0][0, -1].sum().backward()
    print("  OK   RNN com backward (a conta que derrubava o kernel)")
except Exception as e:
    falhas += 1
    print("  ERRO PyTorch:", e)

print("  OK   ffmpeg encontrado" if shutil.which("ffmpeg") else "  AVISO ffmpeg não encontrado (só afeta .mp3/.m4a; .wav funciona)")

for arq in ["mini_transformer.pt", "01_transformer_simplificado.ipynb", "02_whisper_huggingface.ipynb"]:
    print(f"  {'OK  ' if os.path.exists(arq) else 'FALTA'} {arq}")
    falhas += not os.path.exists(arq)

print("\nTudo certo!" if falhas == 0 else f"\n{falhas} problema(s) encontrado(s): veja o LEIA-ME.")
