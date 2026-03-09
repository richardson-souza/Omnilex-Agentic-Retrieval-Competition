import os
import re
import pandas as pd
from tqdm.auto import tqdm
import gc

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class MagicMockPlaceholder:
        def __getattr__(self, name):
            return MagicMockPlaceholder()

        def __call__(self, *args, **kwargs):
            return MagicMockPlaceholder()

    AutoModelForSeq2SeqLM = AutoTokenizer = DataLoader = BitsAndBytesConfig = torch = (
        MagicMockPlaceholder()
    )


def load_translation_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """
    Etapa 2.1: Inicialização Dinâmica (CUDA/CPU).
    Usa CUDA se disponível para evitar gargalo de tempo no Kaggle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando carregamento do modelo {model_name} em: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.type == "cuda":
        # Ativa compressão INT8 para rodar com folga em GPUs de 2GB (Local) a 16GB (Kaggle)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True
        )
    else:
        # Fallback para CPU em ambientes sem GPU
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

    model.eval()
    return tokenizer, model, device


def batch_translate(
    queries: list[str],
    tokenizer,
    model,
    device,
    target_lang: str = "deu_Latn",
    batch_size: int = 16,
) -> list[str]:
    # Ajuste dinâmico de batch_size se for CPU para evitar lentidão extrema
    if device.type == "cpu":
        print("  Aviso: Tradução em CPU detectada. Reduzindo carga.")
        batch_size = max(1, batch_size // 4)

    translated_queries = []
    dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False)
    inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()

    try:
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    except (AttributeError, KeyError):
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    with inference_ctx:
        for batch in tqdm(dataloader, desc=f"Translating to {target_lang}"):
            # Garante que os inputs estejam no dispositivo correto da primeira camada do modelo
            model_device = next(model.parameters()).device
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(model_device)

            generated_tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_length=128, num_beams=1
            )

            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_queries.extend(decoded_batch)

            if model_device.type == "cuda":
                del inputs, generated_tokens
                torch.cuda.empty_cache()
                gc.collect()

    return translated_queries


def sanitize_legal_terms(text: str) -> str:
    text = re.sub(r"Art\s+\.\s+(\d+)", r"Art. \1", text)
    text = re.sub(r"Abs\s+\.\s+(\d+)", r"Abs. \1", text)
    return text


def apply_translation_pipeline(
    df: pd.DataFrame,
    text_column: str = "query",
    target_lang: str = "deu_Latn",
    batch_size: int = 16,
) -> pd.DataFrame:
    df = df.copy()
    tokenizer, model, device = load_translation_model()
    queries_en = df[text_column].tolist()

    queries_translated = batch_translate(
        queries_en, tokenizer, model, device, target_lang=target_lang, batch_size=batch_size
    )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    df[f"{text_column}_de"] = [sanitize_legal_terms(q) for q in queries_translated]
    return df
