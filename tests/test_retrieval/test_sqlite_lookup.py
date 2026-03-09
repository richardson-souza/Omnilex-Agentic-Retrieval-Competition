import pytest
import os
import pandas as pd
from omnilex.retrieval.hybrid import SQLiteTextLookup, build_text_lookup

class TestSQLiteLookup:
    def test_sqlite_insert_and_get(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        lookup = SQLiteTextLookup(db_path)
        
        # Setup data
        df = pd.DataFrame({
            "citation": ["cit1", "cit2"],
            "text": ["text1", "text2"]
        })
        
        # Act
        lookup.insert_chunk(df)
        lookup.create_index()
        
        # Assert
        assert lookup.get("cit1") == "text1"
        assert lookup.get("cit2") == "text2"
        assert lookup.get("non-existent") is None
        
        lookup.close()

    def test_build_text_lookup_flow(self, tmp_path):
        db_path = str(tmp_path / "flow.db")
        laws_csv = tmp_path / "laws.csv"
        pd.DataFrame({"citation": ["L1"], "text": ["Law 1"]}).to_csv(laws_csv, index=False)
        
        courts_csv = tmp_path / "courts.csv"
        pd.DataFrame({"citation": ["C1"], "text": ["Court 1"]}).to_csv(courts_csv, index=False)
        
        # Act
        lookup = build_text_lookup(str(laws_csv), str(courts_csv), db_path=db_path)
        
        # Assert
        assert lookup.get("L1") == "Law 1"
        assert lookup.get("C1") == "Court 1"
        lookup.close()
