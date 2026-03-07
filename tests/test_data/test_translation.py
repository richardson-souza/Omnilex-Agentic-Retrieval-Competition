import sys
from unittest.mock import MagicMock

# Mocking torch and transformers before any other imports
mock_torch = MagicMock()
mock_transformers = MagicMock()
mock_dataloader = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["torch.utils.data"].DataLoader = mock_dataloader
sys.modules["transformers"] = mock_transformers

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import os

from omnilex.data.translation import (
    load_translation_model,
    batch_translate,
    sanitize_legal_terms,
    apply_translation_pipeline,
)


class TestTranslation:
    def test_sanitize_legal_terms_basic(self):
        """Test that legal terms are correctly sanitized."""
        text = "Check Art . 1 and Abs . 2 of the law."
        expected = "Check Art. 1 and Abs. 2 of the law."
        assert sanitize_legal_terms(text) == expected

    def test_sanitize_legal_terms_no_change(self):
        """Test that already correct terms are not changed."""
        text = "Art. 1 ZGB is clear."
        assert sanitize_legal_terms(text) == text

    def test_load_translation_model(self):
        """Test that the model and tokenizer are loaded with correct parameters."""
        mock_torch.device.return_value.type = "cpu"

        # We need to handle the chain .from_pretrained().to(device)
        mock_model = MagicMock()
        mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model  # .to(device) returns the model itself

        tokenizer, model, device = load_translation_model("some-model")

        # Check if model was put in eval mode
        mock_model.eval.assert_called_once()
        assert mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.called
        assert mock_transformers.AutoTokenizer.from_pretrained.called

    def test_batch_translate(self):
        """Test batch translation with mocked model/tokenizer."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_device = MagicMock()
        mock_device.type = "cpu"

        # Mock tokenizer to return an object with .to() method (simulating BatchEncoding)
        mock_inputs = MagicMock()
        mock_tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_tokenizer.lang_code_to_id = {"deu_Latn": 123}

        # Mock model.generate to return some token IDs
        mock_model.generate.return_value = [[10, 11, 12]]

        # Mock tokenizer.batch_decode
        mock_tokenizer.batch_decode.return_value = ["translated text"]

        # Mock DataLoader to return a list containing the batch
        mock_dataloader.return_value = [["english text"]]

        queries = ["english text"]
        result = batch_translate(
            queries, mock_tokenizer, mock_model, mock_device, target_lang="deu_Latn", batch_size=1
        )

        assert result == ["translated text"]
        mock_model.generate.assert_called_once()
        mock_tokenizer.batch_decode.assert_called_once()

    @patch("omnilex.data.translation.load_translation_model")
    @patch("omnilex.data.translation.batch_translate")
    def test_apply_translation_pipeline(self, mock_batch_translate, mock_load_model):
        """Test the full pipeline on a DataFrame."""
        df = pd.DataFrame({"query": ["hello", "world"]})

        mock_load_model.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_batch_translate.return_value = ["hallo", "welt"]

        # Ensure torch.cuda.is_available returns False to avoid errors during test
        mock_torch.cuda.is_available.return_value = False

        result_df = apply_translation_pipeline(df, text_column="query")

        assert "query_de" in result_df.columns
        assert result_df["query_de"].tolist() == ["hallo", "welt"]
