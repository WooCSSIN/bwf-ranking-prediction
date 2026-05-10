"""Tests for data loading module."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.loader import DataLoader


class TestDataLoader:
    """Unit tests for DataLoader."""

    def test_load_returns_dataframe(self, tmp_path):
        """DataLoader.load() should return a pandas DataFrame."""
        # Create a minimal CSV
        csv_content = "player_id,player_name,country_code,draw,rank,points,date\n"
        csv_content += "1,Kento Momota,JPN,MS,1,95000,2024-01-01\n"
        csv_content += "2,Viktor Axelsen,DEN,MS,2,90000,2024-01-01\n"

        csv_file = tmp_path / "bwf_ranking.csv"
        csv_file.write_text(csv_content)

        loader = DataLoader(raw_dir=tmp_path)
        df = loader.load("bwf_ranking.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_coerces_types(self, tmp_path):
        """DataLoader should coerce date, rank, points to correct types."""
        csv_content = "player_id,player_name,country_code,draw,rank,points,date\n"
        csv_content += "1,Player A,JPN,MS,1,95000,2024-01-01\n"

        csv_file = tmp_path / "bwf_ranking.csv"
        csv_file.write_text(csv_content)

        loader = DataLoader(raw_dir=tmp_path)
        df = loader.load("bwf_ranking.csv")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert pd.api.types.is_numeric_dtype(df["rank"])
        assert pd.api.types.is_numeric_dtype(df["points"])

    def test_load_file_not_found(self, tmp_path):
        """DataLoader should raise FileNotFoundError for missing files."""
        loader = DataLoader(raw_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.csv")

    def test_draw_normalized_to_uppercase(self, tmp_path):
        """Draw column should be uppercased after loading."""
        csv_content = "player_id,player_name,country_code,draw,rank,points,date\n"
        csv_content += "1,Player A,JPN,ms,1,95000,2024-01-01\n"

        csv_file = tmp_path / "bwf_ranking.csv"
        csv_file.write_text(csv_content)

        loader = DataLoader(raw_dir=tmp_path)
        df = loader.load("bwf_ranking.csv")

        assert df["draw"].iloc[0] == "MS"
