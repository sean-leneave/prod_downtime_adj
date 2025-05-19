import pytest
import numpy as np
import pandas as pd
from utils import (
    standardize_downtime_percentage,
    interpolate_values,
    get_raw_data_csv,
    add_one_month,
    standardize_to_end_of_month,
    format_numbers,
)

def test_standardize_downtime_percentage():
    assert standardize_downtime_percentage(50) == 0.5
    assert standardize_downtime_percentage(0.5) == 0.5
    assert standardize_downtime_percentage(0) == 0.0
    assert standardize_downtime_percentage(np.nan) == 0.0
    with pytest.raises(ValueError):
        standardize_downtime_percentage(150)

def test_interpolate_values():
    xp = np.array([0, 1, 2])
    fp = np.array([0, 10, 20])
    assert interpolate_values(1.5, xp, fp) == pytest.approx(15.0)

def test_get_raw_data_csv():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv = get_raw_data_csv(df, ['a', 'b'], ['A', 'B'])
    assert "A,B" in csv

def test_add_one_month():
    d = pd.Timestamp('2024-01-31')
    assert add_one_month(d) == pd.Timestamp('2024-02-29')

def test_standardize_to_end_of_month():
    d = pd.Timestamp('2024-01-15')
    assert standardize_to_end_of_month(d) == pd.Timestamp('2024-01-31')

def test_format_numbers():
    df = pd.DataFrame({'x': [1000, 2000]})
    formatted = format_numbers(df)
    assert formatted['x'][0] == '1,000' 