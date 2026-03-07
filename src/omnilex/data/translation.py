import os
import re
import pandas as pd
from tqdm.auto import tqdm

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Define placeholder classes for type hints or simple mocking in tests
    class MagicMockPlaceholder:
        def __getattr__(self, name):
            return MagicMockPlaceholder()

        def __call__(self, *args, **kwargs):
            return MagicMockPlaceholder()

    # These will be mocked in tests
    AutoModelForSeq2SeqLM = MagicMockPlaceholder()
    AutoTokenizer = MagicMockPlaceholder()
    DataLoader = MagicMockPlaceholder()
    torch = MagicMockPlaceholder()


def load_translation_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """
    Etapa 2.1: Inicialização Otimizada do Modelo de Tradução.
    Loads the NLLB model in FP16 for optimized inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregamento do Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Carregamento Otimizado do Modelo em FP16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()  # Congela pesos, desliga Dropout

    return tokenizer, model, device


def batch_translate(
    queries: list[str],
    tokenizer,
    model,
    device,
    target_lang: str = "deu_Latn",
    batch_size: int = 32,
) -> list[str]:
    """
    Etapa 2.2: Inferência em Lotes (Batch Processing) e Geração.
    Translates a list of queries using batching and inference_mode.
    """
    # NLLB exige o ID do idioma alvo para forçar o output do Decoder
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    translated_queries = []

    # Criação de um DataLoader simples sobre a lista para iterar em lotes
    dataloader = DataLoader(queries, batch_size=batch_size, shuffle=False)

    # Use inference_mode if available (torch >= 1.9)
    inference_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()

    with inference_ctx:
        for batch in tqdm(dataloader, desc=f"Translating to {target_lang}"):
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)

            # Geração com num_beams=1 (Greedy Search) para máxima velocidade
            generated_tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_length=128, num_beams=1
            )

            # Decodificação em CPU
            decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_queries.extend(decoded_batch)

    return translated_queries


def sanitize_legal_terms(text: str) -> str:
    """
    Etapa 2.3: Pós-processamento de Segurança (Sanitization).
    Corrects common translation hallucinations in legal terms.
    """
    # Corrige espaçamentos alucinados perto de numerais legais
    # Ex: "Art . 1" -> "Art. 1"
    text = re.sub(r"Art\s+\.\s+(\d+)", r"Art. \1", text)
    text = re.sub(r"Abs\s+\.\s+(\d+)", r"Abs. \1", text)
    return text


def apply_translation_pipeline(
    df: pd.DataFrame,
    text_column: str = "query",
    target_lang: str = "deu_Latn",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Orchestrates the translation of a DataFrame's text column.
    """
    df = df.copy()

    # Etapa 2.1
    tokenizer, model, device = load_translation_model()
    queries_en = df[text_column].tolist()

    # Etapa 2.2: Tradução
    print(f"Iniciando Tradução para {target_lang}...")
    queries_translated = batch_translate(
        queries_en, tokenizer, model, device, target_lang=target_lang, batch_size=batch_size
    )

    # Otimização de RAM: Libertar a VRAM imediatamente após o uso da tradução
    # Isso ocorre antes da sanitização para garantir que os estágios seguintes tenham VRAM livre
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Etapa 2.3: Sanitização
    df[f"{text_column}_de"] = [sanitize_legal_terms(q) for q in queries_translated]

    return df
