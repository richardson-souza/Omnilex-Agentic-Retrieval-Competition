import pytest
import sqlite3
import pandas as pd
from omnilex.retrieval.hybrid import SQLiteTextLookup


class TestSQLiteIntegrity:
    def test_duplicate_citation_handled_gracefully(self, tmp_path):
        """Verify that duplicate citations don't crash and are indexed correctly."""
        db_path = str(tmp_path / "integrity_fixed.db")
        lookup = SQLiteTextLookup(db_path)

        # Insert first doc
        df1 = pd.DataFrame({"citation": ["cit1"], "text": ["text1"]})
        lookup.insert_chunk(df1)

        # Insert second doc with same citation - Should NOT crash now
        df2 = pd.DataFrame({"citation": ["cit1"], "text": ["text2"]})
        lookup.insert_chunk(df2)

        lookup.create_index()

        # Should return the first one found (standard SQLite behavior with min rowid cleanup)
        assert lookup.get("cit1") == "text1"
        lookup.close()
