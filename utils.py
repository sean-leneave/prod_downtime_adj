import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def standardize_downtime_percentage(value: float) -> float:
    """Convert downtime percentage to decimal format if needed."""
    if pd.isna(value):
        return 0.0
    if value < 0 or value > 100:
        raise ValueError("Downtime percentage must be between 0 and 100")
    # Convert to decimal if in percentage format (0-100)
    if value > 1:
        return value / 100
    return value


def interpolate_values(x, xp, fp):
    f = interp1d(xp, fp, fill_value='extrapolate')
    return f(x)


def get_raw_data_csv(df, columns, column_names):
    """Prepare raw data for clipboard"""
    temp_df = df[columns].copy()
    temp_df.columns = column_names
    return temp_df.to_csv(date_format='%Y-%m-%d')


def add_one_month(current_date):
    new_month = (current_date.month % 12) + 1
    new_year = current_date.year + (current_date.month // 12)
    next_month_days = pd.Timestamp(new_year, new_month, 1).days_in_month
    new_day = min(current_date.day, next_month_days)
    return pd.Timestamp(new_year, new_month, new_day)


def standardize_to_end_of_month(date):
    """Convert any date to end of month format.
    All dates are converted to the end of their respective months.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    # Always convert to end of month
    return date.replace(day=date.days_in_month)


def format_numbers(df, is_rate=True):
    """Format numbers with thousand separators and proper decimal places."""
    formatted_df = df.copy()
    for col in formatted_df.columns:
        # Only format numeric columns
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            if is_rate:
                # Format rate columns with no decimal places
                formatted_df[col] = formatted_df[col].apply(lambda x: '{:,.0f}'.format(x) if pd.notnull(x) else '')
            else:
                # Format volume columns with 2 decimal places
                formatted_df[col] = formatted_df[col].apply(lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else '')
    return formatted_df 