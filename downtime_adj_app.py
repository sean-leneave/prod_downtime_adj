import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import io
from openpyxl.styles import Font, PatternFill
import plotly.io as pio
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Constants
COLORS = {
    'background': '#F8F9FA',
    'text': '#2C3E50',
    'oil': '#2ECC71',
    'gas': '#E74C3C',
    'water': '#3498DB',
    'liquid': '#34495E'
}

PLOT_LAYOUT = {
    'font': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': COLORS['text']
    },
    'plot_bgcolor': COLORS['background'],
    'hovermode': 'x unified'
}

# Set page config
st.set_page_config(
    page_title="Production Downtime Adjustment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MAX_YEARS = 30
MAX_MONTHS = MAX_YEARS * 12

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = ""
if 'downtime_data' not in st.session_state:
    st.session_state.downtime_data = ""
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'figure' not in st.session_state:
    st.session_state.figure = None

def standardize_downtime_percentage(value: float) -> float:
    """Convert downtime percentage to decimal format if needed."""
    if value < 0 or value > 100:
        raise ValueError("Downtime percentage must be between 0 and 100")
    # Convert to decimal if in percentage format (0-100)
    if value > 1:
        return value / 100
    return value

def process_input_data(forecast_data, downtime_data):
    """Process input data into proper format for calculations."""
    if not forecast_data or not downtime_data:
        st.error("Please provide both forecast and downtime data.")
        return None, None

    try:
        # Convert strings to dataframes
        try:
            forecast_df = pd.read_csv(io.StringIO(forecast_data), sep='\t')
            if len(forecast_df.columns) == 1:
                # Try comma separator if tab didn't work
                forecast_df = pd.read_csv(io.StringIO(forecast_data), sep=',')
                st.info("Detected comma-separated values for forecast data instead of tabs. Processing anyway.")
        except Exception as e:
            st.error(f"Error parsing forecast data: {str(e)}\nPlease ensure data is tab-separated with proper headers.")
            return None, None

        try:
            downtime_df = pd.read_csv(io.StringIO(downtime_data), sep='\t')
            if len(downtime_df.columns) == 1:
                # Try comma separator if tab didn't work
                downtime_df = pd.read_csv(io.StringIO(downtime_data), sep=',')
                st.info("Detected comma-separated values for downtime data instead of tabs. Processing anyway.")
        except Exception as e:
            st.error(f"Error parsing downtime data: {str(e)}\nPlease ensure data is tab-separated with proper headers.")
            return None, None

        # Check for required columns
        if 'date' not in forecast_df.columns:
            st.error("Forecast data must include a 'date' column.")
            return None, None
            
        if 'date' not in downtime_df.columns:
            st.error("Downtime data must include a 'date' column.")
            return None, None

        # Convert dates to datetime
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        downtime_df['date'] = pd.to_datetime(downtime_df['date'])

        # Standardize dates to end of month
        forecast_df['date'] = forecast_df['date'].apply(standardize_to_end_of_month)
        downtime_df['date'] = downtime_df['date'].apply(standardize_to_end_of_month)

        # Standardize downtime percentages to decimal format
        if 'downtime_pct' in downtime_df.columns:
            try:
                downtime_df['downtime_pct'] = downtime_df['downtime_pct'].apply(standardize_downtime_percentage)
            except ValueError as e:
                st.error(f"Error in downtime data: {str(e)}")
                return None, None

        # Set index to date
        forecast_df.set_index('date', inplace=True)
        downtime_df.set_index('date', inplace=True)

        # Sort by date
        forecast_df.sort_index(inplace=True)
        downtime_df.sort_index(inplace=True)

        return forecast_df, downtime_df

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None

def validate_input_data(forecast_df, downtime_df):
    """Validate input dataframes."""
    errors = []
    warnings = []

    # Check forecast data
    possible_fluid_cols = ['oil_rate', 'gas_rate', 'water_rate']
    available_fluid_cols = [col for col in possible_fluid_cols if col in forecast_df.columns]
    
    # Check if at least one fluid type is present
    if not available_fluid_cols:
        errors.append("At least one fluid type (oil_rate, gas_rate, or water_rate) is required in forecast data")
    else:
        # Check if any available fluid has data (non-zero sum)
        has_fluid_data = False
        for col in available_fluid_cols:
            if forecast_df[col].sum() > 0:
                has_fluid_data = True
                break
        
        if not has_fluid_data:
            errors.append("At least one fluid type must have non-zero values")
        
        # Add warnings for missing fluid types
        missing_fluids = [col.replace('_rate', '') for col in possible_fluid_cols if col not in available_fluid_cols]
        if missing_fluids:
            warnings.append(f"Note: No {', '.join(missing_fluids)} data provided. This is acceptable but may affect some calculations.")

    # Check downtime data
    if 'downtime_pct' not in downtime_df.columns:
        errors.append("Downtime data must include a 'downtime_pct' column")
    else:
        # Check if downtime percentages are within valid range (0-100)
        invalid_downtimes = downtime_df[downtime_df['downtime_pct'] > 1.0]
        if not invalid_downtimes.empty:
            errors.append("Downtime percentages must be between 0 and 100 (or 0 and 1 if in decimal format)")

    # Check date alignment between forecast and downtime data
    forecast_dates = set(forecast_df.index)
    downtime_dates = set(downtime_df.index)
    
    if not forecast_dates.issubset(downtime_dates):
        missing_dates = forecast_dates - downtime_dates
        errors.append(f"Missing downtime data for dates: {sorted(missing_dates)}")
    
    if not downtime_dates.issubset(forecast_dates):
        extra_dates = downtime_dates - forecast_dates
        warnings.append(f"Downtime data includes dates not in forecast: {sorted(extra_dates)}")

    return errors, warnings

# helper functions
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
        if is_rate:
            # Format rate columns with no decimal places
            formatted_df[col] = formatted_df[col].apply(lambda x: '{:,.0f}'.format(x))
        else:
            # Format volume columns with 2 decimal places
            formatted_df[col] = formatted_df[col].apply(lambda x: '{:,.2f}'.format(x))

    return formatted_df


def process_production_data(df):
    dtype_dict = {
        'oil_rate': np.float64,
        'gas_rate': np.float64,
        'water_rate': np.float64,
        'downtime_pct': np.float64,
        'liquid_rate': np.float64,
        'daysInMonth': np.int64,
        'oil_volume': np.float64,
        'gas_volume': np.float64,
        'water_volume': np.float64,
        'liquid_volume': np.float64,
        'Np': np.float64,
        'Gp': np.float64,
        'Wp': np.float64,
        'Lp': np.float64,
        'adj_oil_rate': np.float64,
        'adj_gas_rate': np.float64,
        'adj_water_rate': np.float64,
        'adj_liquid_rate': np.float64,
        'Np_out': np.float64,
        'Gp_out': np.float64,
        'Wp_out': np.float64,
        'Lp_out': np.float64
    }

    # Create additional columns
    df['liquid_rate'] = df['oil_rate'] + df['water_rate']
    df['GOR'] = np.where(df['gas_rate'] / df['oil_rate']>0, df['gas_rate'] / df['oil_rate'], 0)
    df['WOR'] = np.where(df['water_rate'] / df['oil_rate']>0, df['water_rate'] / df['oil_rate'], 0)
    
    # Calculate days in month based on the date format
    def get_days_in_month(date):
        if date.day == 1:  # Beginning of month
            return pd.Timestamp(date.year, date.month, 1).days_in_month
        else:  # End of month
            return date.days_in_month
    
    df['daysInMonth'] = df.index.map(get_days_in_month)

    df['oil_volume'] = (df['oil_rate'] * df['daysInMonth'])/ 1000
    df['gas_volume'] = (df['gas_rate'] * df['daysInMonth']) / 1000
    df['water_volume'] = (df['water_rate'] * df['daysInMonth']) / 1000
    df['liquid_volume'] = (df['liquid_rate'] * df['daysInMonth']) / 1000

    df['Np'] = (df['oil_volume']).cumsum()
    df['Gp'] = (df['gas_volume']).cumsum()
    df['Wp'] = (df['water_volume']).cumsum()
    df['Lp'] = df['Np'] + df['Wp']

    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')

    # Identify decline start
    if (df['oil_rate'] > 0).any():
        oil_rate_diff = np.diff(df['oil_rate'].values)
        decline_start_index = np.where(oil_rate_diff <= 0)[0][0] + 1 if len(np.where(oil_rate_diff <= 0)[0]) > 0 else 0
    elif (df['gas_rate'] > 0).any():
        gas_rate_diff = np.diff(df['gas_rate'].values)
        decline_start_index = np.where(gas_rate_diff <= 0)[0][0] + 1 if len(np.where(gas_rate_diff <= 0)[0]) > 0 else 0
    else:
        decline_start_index = 0

    df_interpolate = df.iloc[decline_start_index:][['Np', 'oil_rate', 'GOR', 'WOR']]

    # Create output DataFrame with correct dtypes
    df_out = df.copy()

    # Initialize new columns with correct dtypes
    new_cols = ['adj_oil_rate', 'adj_gas_rate', 'adj_water_rate', 'adj_liquid_rate',
                'adj_oil_volume', 'adj_gas_volume', 'adj_water_volume', 'adj_liquid_volume',
                'Np_out', 'Gp_out', 'Wp_out', 'Lp_out']
    for col in new_cols:
        df_out[col] = pd.Series(0.0, index=df_out.index, dtype='float64')

    df_out = df_out.astype(dtype_dict)

    # Perform the initial row calculations and assign them to df_out
    df_out.loc[df_out.index[0], 'adj_oil_rate'] = df_out.loc[df_out.index[0], 'oil_rate'] * (
                1 - df_out.loc[df_out.index[0], 'downtime_pct'])
    df_out.loc[df_out.index[0], 'adj_gas_rate'] = df_out.loc[df_out.index[0], 'GOR'] * df_out.loc[
        df_out.index[0], 'adj_oil_rate']
    df_out.loc[df_out.index[0], 'adj_water_rate'] = df_out.loc[df_out.index[0], 'WOR'] * df_out.loc[
        df_out.index[0], 'adj_oil_rate']
    df_out.loc[df_out.index[0], 'adj_liquid_rate'] = df_out.loc[df_out.index[0], 'adj_oil_rate'] + df_out.loc[
        df_out.index[0], 'adj_water_rate']
    df_out.loc[df_out.index[0], 'adj_oil_volume'] = df_out.loc[df_out.index[0], 'adj_oil_rate'] * df_out.loc[
        df_out.index[0], 'daysInMonth'] / 1000
    df_out.loc[df_out.index[0], 'Np_out'] = (df_out.loc[df_out.index[0], 'adj_oil_volume']).cumsum()
    df_out.loc[df_out.index[0], 'Gp_out'] = (df_out.loc[df_out.index[0], 'adj_gas_volume']).cumsum()
    df_out.loc[df_out.index[0], 'Wp_out'] = (df_out.loc[df_out.index[0], 'adj_water_volume']).cumsum()
    df_out.loc[df_out.index[0], 'Lp_out'] = df_out.loc[df_out.index[0], 'Np_out'] + df_out.loc[df_out.index[0], 'Wp_out']

    # Initial conditions for the loop
    max_np = df['Np'].max()
    max_gp = df['Gp'].max()
    max_wp = df['Wp'].max()
    df['Lp'].max()

    last_np_out = 0
    last_gp_out = 0
    last_wp_out = 0
    current_date = df_out.index[0]

    # Main processing loop
    while True:
        if current_date not in df_out.index:
            new_row = {'oil_rate': 0,
                       'gas_rate': 0,
                       'water_rate': 0,
                       'downtime_pct': df_out['downtime_pct'].iloc[-1],
                       # Assume downtime_pct remains constant in extended years
                       'liquid_rate': 0,
                       'daysInMonth': get_days_in_month(current_date),
                        'oil_volume': 0,
                        'gas_volume': 0,
                        'water_volume': 0,
                        'liquid_volume': 0,
                       'Np': df_out['Np'].iloc[-1],
                       'Gp': df_out['Gp'].iloc[-1],
                       'Wp': df_out['Wp'].iloc[-1],
                       'Lp': df_out['Lp'].iloc[-1],
                       'adj_oil_rate': 0,
                       'adj_gas_rate': 0,
                       'adj_water_rate': 0,
                       'adj_liquid_rate': 0,
                          'adj_oil_volume': 0,
                        'adj_gas_volume': 0,
                        'adj_water_volume': 0,
                        'adj_liquid_volume': 0,
                       'Np_out': 0,
                       'Gp_out': 0,
                       'Wp_out': 0,
                       'Lp_out': 0
                       }

            # Date column is added to the dataframe and it should be the index
            df_out = pd.concat([df_out, pd.DataFrame([new_row], index=[current_date])])
            df_out.reset_index(drop=True)

        # Check if we're still within the input range for non-declining oil rates
        if current_date <= df.index[decline_start_index]:
            adj_oil_rate = df_out.loc[current_date, 'oil_rate'] * (1 - df_out.loc[current_date, 'downtime_pct'])
        else:
            # Use interpolated values for oil rate and adjust for downtime
            adj_oil_rate = interpolate_values(last_np_out, df_interpolate['Np'], df_interpolate['oil_rate'])
            downtime_pct = df_out.loc[current_date, 'downtime_pct']
            adj_oil_rate *= (1 - downtime_pct)

        adj_gas_rate = interpolate_values(last_np_out, df_interpolate['Np'], df_interpolate['GOR']) * adj_oil_rate
        adj_water_rate = interpolate_values(last_np_out, df_interpolate['Np'], df_interpolate['WOR']) * adj_oil_rate
        adj_liquid_rate = adj_oil_rate + adj_water_rate

        # Update cumulative productions
        days_in_month = get_days_in_month(current_date)
        adj_oil_volume = adj_oil_rate * days_in_month / 1000
        adj_gas_volume = adj_gas_rate * days_in_month / 1000
        adj_water_volume = adj_water_rate * days_in_month / 1000
        adj_liquid_volume = adj_liquid_rate * days_in_month / 1000
        np_out = last_np_out + adj_oil_volume
        gp_out = last_gp_out + adj_gas_volume
        wp_out = last_wp_out + adj_water_volume
        lp_out = np_out + wp_out

        # Update existing row
        df_out.loc[current_date, 'adj_oil_rate'] = adj_oil_rate
        df_out.loc[current_date, 'adj_gas_rate'] = adj_gas_rate
        df_out.loc[current_date, 'adj_water_rate'] = adj_water_rate
        df_out.loc[current_date, 'adj_liquid_rate'] = adj_liquid_rate
        if adj_oil_rate != 0 and not np.isnan(adj_oil_rate):
            df_out.loc[current_date, 'WOR'] = adj_water_rate / adj_oil_rate if adj_water_rate != 0 else 0
            df_out.loc[current_date, 'GOR'] = adj_gas_rate / adj_oil_rate if adj_gas_rate != 0 else 0
        else:
            df_out.loc[current_date, 'WOR'] = 0
            df_out.loc[current_date, 'GOR'] = 0
        df_out.loc[current_date, 'adj_oil_volume'] = adj_oil_volume
        df_out.loc[current_date, 'adj_gas_volume'] = adj_gas_volume
        df_out.loc[current_date, 'adj_water_volume'] = adj_water_volume
        df_out.loc[current_date, 'adj_liquid_volume'] = adj_liquid_volume
        df_out.loc[current_date, 'Np_out'] = np_out
        df_out.loc[current_date, 'Gp_out'] = gp_out
        df_out.loc[current_date, 'Wp_out'] = wp_out
        df_out.loc[current_date, 'Lp_out'] = lp_out

        np_out = df_out.loc[current_date, 'Np_out']
        last_gp_out = df_out.loc[current_date, 'Gp_out']
        last_wp_out = df_out.loc[current_date, 'Wp_out']
        last_np_out = np_out
        
        # Move to next month based on current date format
        if current_date.day == 1:  # Beginning of month
            current_date = pd.Timestamp(current_date.year, current_date.month, 1) + pd.offsets.MonthEnd(1)
        else:  # End of month
            current_date += pd.offsets.MonthEnd(1)

        # Check terminal conditions
        if last_np_out > max_np or last_gp_out > max_gp or last_wp_out > max_wp or len(df_out) >= 361:
            last_date = df_out.index[-2]
            current_date = df_out.index[-1]
            days_in_month = get_days_in_month(current_date)

            if last_np_out > max_np and last_gp_out > max_gp:
                # Both oil and gas exceeded - calculate rates to hit both targets
                rem_np = last_np_out - max_np
                rem_gp = last_gp_out - max_gp

                # Calculate required rates to hit targets
                adj_oil_rate = rem_np * 1000 / days_in_month
                adj_gas_rate = rem_gp * 1000 / days_in_month
                last_wor = df_out.loc[last_date, 'WOR']
                adj_water_rate = adj_oil_rate * last_wor

            elif last_np_out > max_np:
                # Only oil exceeded
                rem_np = last_np_out - max_np
                adj_oil_rate = rem_np * 1000 / days_in_month
                last_gor = df_out.loc[last_date, 'GOR']
                adj_gas_rate = adj_oil_rate * last_gor
                last_wor = df_out.loc[last_date, 'WOR']
                adj_water_rate = adj_oil_rate * last_wor

            elif last_wp_out > max_wp:
                # Only water exceeded
                rem_wp = last_wp_out - max_wp
                adj_water_rate = rem_wp * 1000 / days_in_month
                adj_oil_rate = df_out.loc[last_date, 'adj_oil_rate']
                last_wor = df_out.loc[last_date, 'WOR']
                adj_gas_rate = adj_oil_rate * last_wor

            else:
                # Only gas exceeded
                rem_gp = last_gp_out - max_gp
                adj_gas_rate = rem_gp * 1000 / days_in_month
                adj_oil_rate = df_out.loc[last_date, 'adj_oil_rate']
                last_wor = df_out.loc[last_date, 'WOR']
                adj_water_rate = adj_oil_rate * last_wor

            # Update the rates in df_out
            df_out.loc[current_date, 'adj_oil_rate'] = adj_oil_rate
            df_out.loc[current_date, 'adj_gas_rate'] = adj_gas_rate
            df_out.loc[current_date, 'adj_water_rate'] = adj_water_rate
            df_out.loc[current_date, 'adj_liquid_rate'] = adj_oil_rate + adj_water_rate
            df_out.loc[current_date, 'WOR'] = adj_water_rate / adj_oil_rate if adj_oil_rate > 0 else 0
            df_out.loc[current_date, 'GOR'] = adj_gas_rate / adj_oil_rate if adj_oil_rate > 0 else 0

            # Calculate volumes
            adj_oil_volume = adj_oil_rate * days_in_month / 1000
            adj_gas_volume = adj_gas_rate * days_in_month / 1000
            adj_water_volume = adj_water_rate * days_in_month / 1000

            df_out.loc[current_date, 'adj_oil_volume'] = adj_oil_volume
            df_out.loc[current_date, 'adj_gas_volume'] = adj_gas_volume
            df_out.loc[current_date, 'adj_water_volume'] = adj_water_volume
            df_out.loc[current_date, 'adj_liquid_volume'] = (adj_oil_rate + adj_water_rate) * days_in_month / 1000

            # Set final cumulative volumes to exactly match targets
            df_out.loc[current_date, 'Np_out'] = max_np if last_np_out > max_np else last_np_out + adj_oil_volume
            df_out.loc[current_date, 'Gp_out'] = max_gp if last_gp_out > max_gp else last_gp_out + adj_gas_volume
            df_out.loc[current_date, 'Wp_out'] = last_wp_out + adj_water_volume
            df_out.loc[current_date, 'Lp_out'] = df_out.loc[current_date, 'Np_out'] + df_out.loc[current_date, 'Wp_out']

            # Add final timestep with zero rates but maintaining cumulative volumes
            new_row = df_out.loc[current_date].copy()
            new_row[['adj_oil_rate', 'adj_gas_rate', 'adj_water_rate', 'adj_liquid_rate']] = 0
            new_row[['Np_out', 'Gp_out', 'Wp_out', 'Lp_out']] = [max_np, max_gp, new_row['Wp_out'], max_np + new_row['Wp_out']]
            new_df = pd.DataFrame(new_row).T.set_index(pd.Index([add_one_month(current_date)]))

            df_out = pd.concat([df_out, new_df])
            df_out.set_index(pd.Index(df_out.index), inplace=True)

        # Break condition
        if last_np_out > df_out['Np'].max() or last_gp_out > df_out['Gp'].max() or len(df_out) >= 361:
            break

        df_out['GOR_out'] = df_out['adj_gas_rate'] / df_out['adj_oil_rate'].replace(0, np.nan)
        df_out['WOR_out'] = df_out['adj_water_rate'] / df_out['adj_oil_rate'].replace(0, np.nan)

    return df_out, df_interpolate


def create_plotly_figure(df_out):
    """Create complete plotly figure with all traces."""
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Oil Rate vs Time', 'Cumulative Oil Production',
            'Gas Rate vs Time', 'Cumulative Gas Production',
            'Water Rate vs Time', 'Cumulative Water Production',
            'Liquid Rate vs Time', 'Cumulative Liquid Production'
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.1
    )

    # Common trace settings
    line_width = 3
    opacity = 0.8
    hover_tmpl = "%{y:,.0f}"

    # Calculate max values for axis scaling
    max_rate = max(
        df_out['oil_rate'].max(),
        df_out['adj_oil_rate'].max(),
        df_out['gas_rate'].max(),
        df_out['adj_gas_rate'].max(),
        df_out['water_rate'].max(),
        df_out['adj_water_rate'].max(),
        df_out['liquid_rate'].max(),
        df_out['adj_liquid_rate'].max()
    )

    max_cumulative = max(
        df_out['Np'].max(),
        df_out['Np_out'].max(),
        df_out['Gp'].max(),
        df_out['Gp_out'].max(),
        df_out['Wp'].max(),
        df_out['Wp_out'].max(),
        df_out['Lp'].max(),
        df_out['Lp_out'].max()
    )

    max_rate_rounded = max_rate * 1.1
    max_cumulative_rounded = max_cumulative * 1.1

    # Oil Rate and Cumulative
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['oil_rate'],
                  name='Oil Rate',
                  line=dict(color=COLORS['oil'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['adj_oil_rate'],
                  name='Oil Rate DT',
                  line=dict(color=COLORS['oil'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Np'],
                  name='Np',
                  line=dict(color=COLORS['oil'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Np_out'],
                  name='Np DT',
                  line=dict(color=COLORS['oil'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=1, col=2
    )

    # Gas Rate and Cumulative
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['gas_rate'],
                  name='Gas Rate',
                  line=dict(color=COLORS['gas'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['adj_gas_rate'],
                  name='Gas Rate DT',
                  line=dict(color=COLORS['gas'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Gp'],
                  name='Gp',
                  line=dict(color=COLORS['gas'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Gp_out'],
                  name='Gp DT',
                  line=dict(color=COLORS['gas'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=2, col=2
    )

    # Water Rate and Cumulative
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['water_rate'],
                  name='Water Rate',
                  line=dict(color=COLORS['water'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['adj_water_rate'],
                  name='Water Rate DT',
                  line=dict(color=COLORS['water'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Wp'],
                  name='Wp',
                  line=dict(color=COLORS['water'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Wp_out'],
                  name='Wp DT',
                  line=dict(color=COLORS['water'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=3, col=2
    )

    # Liquid Rate and Cumulative
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['liquid_rate'],
                  name='Liquid Rate',
                  line=dict(color=COLORS['liquid'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['adj_liquid_rate'],
                  name='Liquid Rate DT',
                  line=dict(color=COLORS['liquid'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Lp'],
                  name='Lp',
                  line=dict(color=COLORS['liquid'], width=line_width),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_out.index, y=df_out['Lp_out'],
                  name='Lp DT',
                  line=dict(color=COLORS['liquid'], width=line_width, dash='dot'),
                  opacity=opacity,
                  hovertemplate=hover_tmpl),
        row=4, col=2
    )

    # Update layout and axes
    fig.update_layout(
        height=1400,
        width=1000,
        showlegend=False,
        title={
            'text': "Forecast Comparison",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        **PLOT_LAYOUT
    )

    # Update axes with consistent ranges
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(
                row=i, col=j,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.5)'
            )

            fig.update_yaxes(
                row=i, col=j,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.5)',
                tickformat=",",
                title_standoff=15,
                range=[0, max_rate_rounded if j == 1 else max_cumulative_rounded]
            )

    # Add WOR vs Np plot
    fig = go.Figure()
    
    # Add interpolation points
    fig.add_trace(go.Scatter(
        x=np_values,
        y=wor_values,
        mode='markers',
        name='Interpolation Points',
        marker=dict(size=10, color='blue')
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=np_interp,
        y=wor_interp,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add vertical dashed line at 12M
    fig.add_trace(go.Scatter(
        x=[12_000_000, 12_000_000],
        y=[0.1, 0.81],
        mode='lines',
        line=dict(color='black', dash='dash', width=1),
        showlegend=False
    ))
    
    # Add horizontal dashed line at WOR=0.81
    fig.add_trace(go.Scatter(
        x=[0, 12_000_000],
        y=[0.81, 0.81],
        mode='lines',
        line=dict(color='black', dash='dash', width=1),
        showlegend=False
    ))
    
    # Add WOR lookup text
    fig.add_annotation(
        x=12_000_000,
        y=0.1,
        text="WOR = 0.81",
        showarrow=False,
        yshift=-20,
        xshift=0,
        font=dict(size=12)
    )

    return fig


def copy_to_clipboard(data_type='rates'):
    if 'processed_data' not in st.session_state:
        return

    df = st.session_state['processed_data']

    if data_type == 'rates':
        temp_df = df[['adj_oil_rate', 'adj_gas_rate', 'adj_water_rate', 'adj_liquid_rate']].copy()
        temp_df.columns = ['Oil Rate', 'Gas Rate', 'Water Rate', 'Liquid Rate']
    else:
        temp_df = df[['adj_oil_volume', 'adj_gas_volume', 'adj_water_volume', 'adj_liquid_volume']].copy()
        temp_df.columns = ['Oil Volume', 'Gas Volume', 'Water Volume', 'Liquid Volume']

    # Create CSV for download
    csv = temp_df.to_csv(index=True)
    
    # Create a download button
    st.download_button(
        label=f"Download {data_type.capitalize()} as CSV",
        data=csv,
        file_name=f"{data_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Display data in a text area for manual copying
    st.write(f"**{data_type.capitalize()} data (select all and copy):**")
    
    # Convert to tab-separated format for better clipboard pasting
    tsv_data = temp_df.to_csv(sep='\t', index=True)
    st.text_area("", tsv_data, height=150)
    
    st.info("👆 Select all text in the box above (Ctrl+A or Cmd+A), then copy (Ctrl+C or Cmd+C) to clipboard.")


def create_mosaic_template(df_out: pd.DataFrame, entity_name: str, reserve_category: str) -> pd.DataFrame:
    """Create Mosaic loader template from production data."""
    if df_out is None or df_out.empty:
        raise ValueError("No data to process.")
    # Get volume data (multiply by 1000 to convert back to bbl/mcf)
    vol_data = []

    # Process Oil volumes
    oil_data = df_out['adj_oil_volume'].copy() * 1000
    for date, value in oil_data.items():
        vol_data.append({
            'Entity Name': entity_name,
            'Reserve Category': reserve_category,
            'Use': 'Produced',
            'Product': 'Oil',
            'Detail Date (y-m-d)': date,
            'Cum For Period (Mcf or bbl)': value
        })

    # Process Gas volumes
    gas_data = df_out['adj_gas_volume'].copy() * 1000
    for date, value in gas_data.items():
        vol_data.append({
            'Entity Name': entity_name,
            'Reserve Category': reserve_category,
            'Use': 'Produced',
            'Product': 'Gas',
            'Detail Date (y-m-d)': date,
            'Cum For Period (Mcf or bbl)': value
        })

    # Process Water volumes
    water_data = df_out['adj_water_volume'].copy() * 1000
    for date, value in water_data.items():
        vol_data.append({
            'Entity Name': entity_name,
            'Reserve Category': reserve_category,
            'Use': 'Produced',
            'Product': 'Water',
            'Detail Date (y-m-d)': date,
            'Cum For Period (Mcf or bbl)': value
        })

    # Create DataFrame with exact column order
    mosaic_df = pd.DataFrame(vol_data)
    mosaic_df = mosaic_df[[
        'Entity Name',
        'Reserve Category',
        'Use',
        'Product',
        'Detail Date (y-m-d)',
        'Cum For Period (Mcf or bbl)'
    ]]

    product_order = pd.Categorical(mosaic_df['Product'], categories=['Oil', 'Gas', 'Water'], ordered=True)
    mosaic_df['Product'] = product_order

    # Sort by Date and Product
    mosaic_df = mosaic_df.sort_values(by=['Product', 'Reserve Category','Detail Date (y-m-d)'], ascending=[True, True, True])

    mosaic_df['Detail Date (y-m-d)'] = mosaic_df['Detail Date (y-m-d)'].dt.date
    
    return mosaic_df

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .section-break {
        margin: 0.3rem 0;
    }
    .table-title {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 0.3rem;
    }
    .input-title {
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 8px;
    }
    .plot-title {
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] div:has(> table) {
        width: 100%;
        text-align: center;
    }
    [data-testid="stDataFrame"] table {
        width: 100%;
        text-align: center;
    }
    [data-testid="stDataFrame"] th {
        text-align: center !important;
    }
    [data-testid="stDataFrame"] td {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Paste monthly forecast into the input areas
    2. Click 'Process Data' to run calculations
    3. Compare input and adjusted forecasts
    4. Copy or download processed forecasts

    ### Data Format
    - Production Forecast: 
        - Date, monthly
        - Oil Rate (bopd)
        - Gas Rate (mcfd)
        - Water Rate (bwpd)
    - Downtime Forecast: 
        - Date, monthly
        - Downtime Percentage (0-1 or 0-100)
    - Common date formats are accepted
    - Use tab-separated values when pasting data (e.g. from Excel)
    """)
    
    # Add Documentation button
    if st.button("📚 View Documentation", use_container_width=True):
        st.session_state.show_docs = True

# Documentation Modal
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = False

if st.session_state.show_docs:
    with st.expander("Documentation", expanded=True):
        st.markdown("""
        ## Process Flow & Interpolation Example
        """)
        
        # Create two columns for the flowchart and plot
        flow_col, plot_col = st.columns([1, 1])
        
        with flow_col:
            # Process Flow diagram using Graphviz
            process_flow = """
            digraph {
                rankdir=TB;
                node [shape=box, style=filled, fillcolor=lightgray];
                
                Start [label="Start"];
                Input [label="Input Data"];
                Calc [label="Calculate Initial Rates"];
                Interp [label="Interpolate GOR/WOR vs Np"];
                Adjust [label="Adjust Rates for Downtime"];
                Check [label="Check Np"];
                End [label="End"];
                
                Start -> Input;
                Input -> Calc;
                Calc -> Interp;
                Interp -> Adjust;
                Adjust -> Check;
                Check -> Interp [label="Output Np < Input Np"];
                Check -> End [label="Target Reached"];
            }
            """
            st.graphviz_chart(process_flow)
        
        with plot_col:
            # Create data
            np_values = np.linspace(0, 20_000_000, 6)
            slope = 3.0 / 20_000_000
            intercept = -2.0
            log_wor_values = slope * np_values + intercept
            wor_values = np.exp(log_wor_values)
            
            np_interp = np.linspace(0, 20_000_000, 100)
            log_wor_interp = slope * np_interp + intercept
            wor_interp = np.exp(log_wor_interp)
            
            # Create figure with smaller size
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot data
            ax.plot(np_interp/1e6, wor_interp, 'r-', label='Forecast', linewidth=2)
            ax.plot(np_values/1e6, wor_values, 'bo', label='Interpolation Points', markersize=6)
            
            # Calculate the exact WOR value at Np = 12M
            wor_at_12m = np.exp(slope * 12_000_000 + intercept)
            
            # Add reference lines only up to the intersection point
            ax.plot([12, 12], [0.1, wor_at_12m], 'k--', linewidth=1, alpha=0.5)  # Vertical line
            ax.plot([0, 12], [wor_at_12m, wor_at_12m], 'k--', linewidth=1, alpha=0.5)  # Horizontal line
            
            # Add text
            ax.text(12.2, 0.15, 'WOR = 0.81', fontsize=10)
            
            # Set scales and limits
            ax.set_yscale('log')
            ax.set_ylim(0.1, 10)
            ax.set_xlim(0, 20)
            
            # Set grid
            ax.grid(True, which='both', linestyle='-', alpha=0.2)
            ax.grid(True, which='minor', linestyle=':', alpha=0.2)
            
            # Set labels
            ax.set_xlabel('Cumulative Oil Production (MMBO)')
            ax.set_ylabel('Water-Oil Ratio (WOR)')
            
            # Format y-axis ticks
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
            
            # Add legend
            ax.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to image and display
            from io import BytesIO
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            plt.close()
        
        # Close button for documentation
        if st.button("Close Documentation"):
            st.session_state.show_docs = False
            st.experimental_rerun()

# Main application
st.title("Forecast-Downtime Processing Tool")

# Input Section
st.markdown("### Input Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Production Forecast")
    # Create a sample dataframe for the grid
    forecast_columns = ['date', 'oil_rate', 'gas_rate', 'water_rate']
    forecast_df = pd.DataFrame(columns=forecast_columns)
    
    # Convert stored data back to dataframe if it exists
    if st.session_state.forecast_data:
        try:
            # First try tab separator
            forecast_df = pd.read_csv(io.StringIO(st.session_state.forecast_data), sep='\t')
            if len(forecast_df.columns) == 1:
                # If tab separator didn't work, try comma
                forecast_df = pd.read_csv(io.StringIO(st.session_state.forecast_data), sep=',')
            # Ensure all required columns exist
            for col in forecast_columns:
                if col not in forecast_df.columns:
                    forecast_df[col] = None
            # Ensure date column is in string format
            if 'date' in forecast_df.columns:
                forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.strftime('%Y-%m-%d')
        except Exception as e:
            st.error(f"Error parsing forecast data: {str(e)}")
            forecast_df = pd.DataFrame(columns=forecast_columns)
    else:
        # Initialize with 5 empty rows
        forecast_df = pd.DataFrame({
            'date': [''] * 5,
            'oil_rate': [None] * 5,
            'gas_rate': [None] * 5,
            'water_rate': [None] * 5
        })
    
    # Display editable dataframe
    edited_forecast_df = st.data_editor(
        forecast_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn(
                "Date",
                help="Enter date in YYYY-MM-DD format",
                width="fixed",
                default="",
            ),
            "oil_rate": st.column_config.NumberColumn(
                "Oil Rate (bopd)",
                help="Enter oil rate in barrels per day",
                min_value=0,
                format="%d",
                width="fixed",
                default=None,
            ),
            "gas_rate": st.column_config.NumberColumn(
                "Gas Rate (mcfd)",
                help="Enter gas rate in thousand cubic feet per day",
                min_value=0,
                format="%d",
                width="fixed",
                default=None,
            ),
            "water_rate": st.column_config.NumberColumn(
                "Water Rate (bwpd)",
                help="Enter water rate in barrels per day",
                min_value=0,
                format="%d",
                width="fixed",
                default=None,
            ),
        }
    )
    
    # Only update session state if the dataframe has changed
    if not edited_forecast_df.equals(forecast_df):
        st.session_state.forecast_data = edited_forecast_df.to_csv(sep='\t', index=False)

with col2:
    st.subheader("Downtime Forecast")
    # Create a sample dataframe for the grid
    downtime_columns = ['date', 'downtime_pct']
    downtime_df = pd.DataFrame(columns=downtime_columns)
    
    # Convert stored data back to dataframe if it exists
    if st.session_state.downtime_data:
        try:
            # First try tab separator
            downtime_df = pd.read_csv(io.StringIO(st.session_state.downtime_data), sep='\t')
            if len(downtime_df.columns) == 1:
                # If tab separator didn't work, try comma
                downtime_df = pd.read_csv(io.StringIO(st.session_state.downtime_data), sep=',')
            # Ensure all required columns exist
            for col in downtime_columns:
                if col not in downtime_df.columns:
                    downtime_df[col] = None
            # Ensure date column is in string format
            if 'date' in downtime_df.columns:
                downtime_df['date'] = pd.to_datetime(downtime_df['date']).dt.strftime('%Y-%m-%d')
        except Exception as e:
            st.error(f"Error parsing downtime data: {str(e)}")
            downtime_df = pd.DataFrame(columns=downtime_columns)
    else:
        # Initialize with 5 empty rows
        downtime_df = pd.DataFrame({
            'date': [''] * 5,
            'downtime_pct': [None] * 5
        })
    
    # Display editable dataframe
    edited_downtime_df = st.data_editor(
        downtime_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn(
                "Date",
                help="Enter date in YYYY-MM-DD format",
                width="fixed",
                default="",
            ),
            "downtime_pct": st.column_config.NumberColumn(
                "Downtime Percentage",
                help="Enter downtime percentage (0-100 or 0-1)",
                min_value=0,
                max_value=100,
                format="%.2f",
                width="fixed",
                default=None,
            ),
        }
    )
    
    # Only update session state if the dataframe has changed
    if not edited_downtime_df.equals(downtime_df):
        st.session_state.downtime_data = edited_downtime_df.to_csv(sep='\t', index=False)

# Process and Clear buttons in the same row
col1, col2 = st.columns([1, 5])

with col1:
    process_button = st.button("Process Data", type="primary")
    
with col2:
    if st.button("Clear Input"):
        # Clear all session state variables
        st.session_state.forecast_data = ""
        st.session_state.downtime_data = ""
        st.session_state.processed_data = None
        st.session_state.figure = None
        st.experimental_rerun()

def process_and_store_data():
    """Process input data and store results in session state"""
    if not st.session_state.forecast_data or not st.session_state.downtime_data:
        st.error("Please provide both forecast and downtime data.")
        return False

    try:
        # Store input data in session state
        st.session_state.forecast_data = st.session_state.forecast_data
        st.session_state.downtime_data = st.session_state.downtime_data

        # Process input data
        forecast_df, downtime_df = process_input_data(st.session_state.forecast_data, st.session_state.downtime_data)
        if forecast_df is None or downtime_df is None:
            return False

        # Validate data
        errors, warnings = validate_input_data(forecast_df, downtime_df)
        if errors:
            for error in errors:
                st.error(error)
            return False
        
        # Display warnings but continue processing
        if warnings:
            with st.expander("⚠️ Warnings - Click to expand", expanded=True):
                for warning in warnings:
                    st.warning(warning)

        # Combine and process data
        df = forecast_df.join(downtime_df, how='left')
        df = df.sort_index()
        df_out, df_interpolate = process_production_data(df)

        # Store results in session state
        st.session_state.processed_data = df_out
        st.session_state.figure = create_plotly_figure(df_out)

        return True
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return False


# Process button
st.markdown("<div class='section-break'></div>", unsafe_allow_html=True)
if process_button:
    process_and_store_data()

# Display results if data is processed
if st.session_state.processed_data is not None:
    df_out = st.session_state.processed_data

    # Display plot
    st.markdown("<div class='section-break'></div>", unsafe_allow_html=True)
    st.markdown("### Results")
    st.plotly_chart(st.session_state.figure, use_container_width=True)

    # Display tables section
    st.markdown("<div class='section-break'></div>", unsafe_allow_html=True)
    st.markdown("<p class='plot-title'>Processed Data Tables</p>", unsafe_allow_html=True)

    # Prepare data for display
    rate_cols = ['adj_oil_rate', 'adj_gas_rate', 'adj_water_rate']
    volume_cols = ['adj_oil_volume', 'adj_gas_volume', 'adj_water_volume']
    rate_names = ['Oil Rate (bopd)', 'Gas Rate (mcfd)', 'Water Rate (bwpd)']
    volume_names = ['Oil Volume (mbo)', 'Gas Volume (mmcf)', 'Water Volume (mbw)']

    rate_df = df_out[rate_cols].copy()
    volume_df = df_out[volume_cols].copy()

    # Format dates and rename columns
    rate_df.index = rate_df.index.strftime('%Y-%m')
    volume_df.index = volume_df.index.strftime('%Y-%m')
    rate_df.columns = rate_names
    volume_df.columns = volume_names

    # Display tables
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="table-title">Production Rates</p>', unsafe_allow_html=True)
        formatted_rate_df = format_numbers(rate_df, is_rate=True)
        st.dataframe(formatted_rate_df, height=400, use_container_width=True)

    with col2:
        st.markdown('<p class="table-title">Production Volumes</p>', unsafe_allow_html=True)
        formatted_volume_df = format_numbers(volume_df, is_rate=False)
        st.dataframe(formatted_volume_df, height=400, use_container_width=True)

    # Buttons section
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
    button_cols = st.columns([1, 1, 1])

    with button_cols[0]:
        rates_expander = st.expander("Export Rates Data")
        with rates_expander:
            copy_to_clipboard('rates')

    with button_cols[1]:
        volumes_expander = st.expander("Export Volumes Data")
        with volumes_expander:
            copy_to_clipboard('volumes')

    with button_cols[2]:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            rate_df.to_excel(writer, sheet_name="Rates")
            volume_df.to_excel(writer, sheet_name="Volumes")

        st.download_button(
            label="Download Complete Excel",
            data=buffer.getvalue(),
            file_name="processed_data_complete.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.markdown("""
        <style>
        .vertical-align-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .vertical-align-container > div {
            margin-right: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Mosaic template section
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("### Generate Mosaic Template")

    mosaic_cols = st.columns([1, 1, 1])

    with mosaic_cols[0]:
        entity_name = st.text_input("Entity Name", placeholder="Enter entity name...")

    with mosaic_cols[1]:
        reserve_category = st.selectbox(
            "Reserve Category",
            ["PDP", "2PDP", "3PDP"]
        )

    with mosaic_cols[2]:
        if st.button("Generate & Download Template", type="primary", use_container_width=True):
            if not entity_name:
                st.error("Please enter an Entity Name")
            elif st.session_state.processed_data is None:
                st.error("No processed data available. Please process the data first.")
            else:
                title = f"Mosaic_Loader_{entity_name}_{reserve_category}.xlsx"
                mosaic_df = create_mosaic_template(
                    df_out=st.session_state.processed_data,
                    entity_name=entity_name,
                    reserve_category=reserve_category
                )

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    mosaic_df.to_excel(writer, index=False, sheet_name='Sheet1')
                    worksheet = writer.sheets['Sheet1']

                    # Format headers
                    for cell in worksheet[1]:
                        cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                        cell.font = Font(color='FFFFFF', bold=True)

                    # Set column widths
                    worksheet.column_dimensions['A'].width = 30
                    worksheet.column_dimensions['B'].width = 18
                    worksheet.column_dimensions['C'].width = 12
                    worksheet.column_dimensions['D'].width = 10
                    worksheet.column_dimensions['E'].width = 19
                    worksheet.column_dimensions['F'].width = 25

                # Write and download file directly
                buffer.seek(0)
                st.success(f"Generated template for {entity_name}")
                st.download_button(
                    label="Click here to Download Template",
                    data=buffer.getvalue(),
                    file_name=title,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    # Add extra spacing at the bottom
    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
