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

    AutoModelForSeq2SeqLM = MagicMockPlaceholder()
    AutoTokenizer = MagicMockPlaceholder()
    DataLoader = MagicMockPlaceholder()
    BitsAndBytesConfig = MagicMockPlaceholder()
    torch = MagicMockPlaceholder()


def load_translation_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """
    Etapa 2.1: Inicialização com Fallback Seguro para CPU.
    MX150 (2GB) is too small for loading NLLB weights in GPU via transformers.
    """
    # Forçar CPU se detectado hardware restrito para evitar OOMs repetidos
    device = torch.device("cpu")
    print(f"Loading {model_name} on CPU to ensure stability (GPU 2GB limit).")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load on CPU (FP32)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

    model.eval()
    return tokenizer, model, device


def batch_translate(
    queries: list[str],
    tokenizer,
    model,
    device,
    target_lang: str = "deu_Latn",
    batch_size: int = 16,  # Can be larger on CPU
) -> list[str]:
    translated_queries = []

    dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False)
    inference_ctx = torch.no_grad()

    try:
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    except (AttributeError, KeyError):
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    with inference_ctx:
        for batch in tqdm(dataloader, desc=f"Translating to {target_lang} (CPU)"):
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)

            generated_tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_length=128, num_beams=1
            )

            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_queries.extend(decoded_batch)

            # GC nonetheless
            del inputs, generated_tokens
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

    print(f"Iniciando Tradução para {target_lang}...")
    queries_translated = batch_translate(
        queries_en, tokenizer, model, device, target_lang=target_lang, batch_size=batch_size
    )

    del model
    del tokenizer
    gc.collect()

    df[f"{text_column}_de"] = [sanitize_legal_terms(q) for q in queries_translated]
    return df
