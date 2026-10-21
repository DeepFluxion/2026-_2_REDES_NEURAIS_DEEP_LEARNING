# Ambiente Anaconda para os dois notebooks (Windows)

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `01_transformer_simplificado.ipynb` | Notebook 1: Transformer simplificado em PyTorch |
| `02_whisper_huggingface.ipynb` | Notebook 2: Whisper do Hugging Face, camada a camada, em português |
| `train_mini_transformer.py` | Treina o modelo do notebook 1 (já foi feito: ver `mini_transformer.pt`) |
| `mini_transformer.pt` | Pesos treinados do notebook 1 (usados na aula; só inferência) |
| `environment.yml` | Receita do ambiente conda |
| `verificar_ambiente.py` | Checa se tudo está instalado |

Coloque todos na **mesma pasta**, com caminho curto e sem acentos (por exemplo `C:\aula_transformers`).

## 1. Criar o ambiente (uma vez, com internet)

Abra o **Anaconda Prompt** (menu Iniciar) e rode:

```bat
cd C:\aula_transformers
conda env create -f environment.yml
conda activate transformers-aula
```

Isso instala Python 3.11, PyTorch (versão **CPU**), Transformers, JupyterLab, `soundfile`, `ffmpeg` e `pyttsx3`. Leva alguns minutos.

## 2. Verificar

```bat
python verificar_ambiente.py
```

Todos os itens devem aparecer como `OK`.

## 3. Baixar o modelo Whisper antes da aula (uma vez, com internet)

```bat
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration as W; WhisperProcessor.from_pretrained('openai/whisper-tiny'); W.from_pretrained('openai/whisper-tiny'); print('modelo em cache')"
```

Repita com `openai/whisper-base` se quiser usá-lo. Os arquivos ficam em `C:\Users\<você>\.cache\huggingface` e a aula passa a funcionar sem rede.

## 4. Abrir os notebooks

```bat
jupyter lab
```

Ao abrir cada notebook, confira se o *kernel* é **Python 3 (transformers-aula)**; se não, use *Kernel → Change Kernel*. Se o kernel não aparecer na lista:

```bat
python -m ipykernel install --user --name transformers-aula --display-name "Python 3 (transformers-aula)"
```

## 5. Áudio para o notebook 2

Salve um áudio em português como `amostra_pt.wav` na pasta (5 a 25 s, fala clara). Opções:

- **Gravador de Voz do Windows**: o arquivo sai em `.m4a`; ajuste `AUDIO = "gravacao.m4a"` na primeira célula (o `ffmpeg` do ambiente lê o formato).
- **Sem áudio:** se `amostra_pt.wav` não existir, o notebook tenta gerar um com a voz do Windows (`pyttsx3`; exige uma voz em português instalada em *Configurações → Hora e idioma → Fala*).

## Se você tem GPU NVIDIA

Não é necessária. Se quiser usar, troque, no `environment.yml`, a linha do índice por uma das versões CUDA listadas em <https://pytorch.org/get-started/locally/> e recrie o ambiente. O notebook 2 detecta a GPU sozinho (`DEVICE`).

## Recriar ou remover o ambiente

```bat
conda env remove -n transformers-aula
conda env create -f environment.yml
```

## Problemas comuns

| Sintoma | Solução |
|---|---|
| **"Kernel Restarting": o kernel caiu ao rodar uma célula com PyTorch** | causa comum no Windows: duas cópias da biblioteca OpenMP (PyTorch + numpy do conda). A primeira célula dos notebooks já define `KMP_DUPLICATE_LIB_OK=TRUE`; rode essa célula de novo após *Kernel → Restart*. Se ainda cair, acrescente `torch.set_num_threads(1)` logo depois do `import torch`; se persistir, abra o *Anaconda Prompt*, rode `jupyter lab` e copie a mensagem de erro que aparece no terminal na hora da queda |
| `conda` não é reconhecido | use o **Anaconda Prompt**, não o PowerShell comum |
| Erro de `torch` ao instalar | rode `conda activate transformers-aula` e depois `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `mini_transformer.pt` não encontrado | confirme que o `.pt` está na mesma pasta do notebook; ou rode `python train_mini_transformer.py` (~7 min em CPU) |
| Download do Whisper falha | veja o passo 3 e confira o acesso a `huggingface.co` |
| Aviso sobre *symlinks* | inofensivo |
