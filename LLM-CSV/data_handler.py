import pandas as pd
import streamlit as st 
import os 

_df = None 
@st.cache_data
def _load_data_internal():
    """
    Loads the synthetic blood bank data from a CSV file.
    Caches the data to avoid re-loading on every rerun.
    """
    try:
        df_loaded = pd.read_csv('RAG/synthetic_data_blood_bank.csv')
        df_loaded['TRANSFUSION_DT'] = pd.to_datetime(df_loaded['TRANSFUSION_DT'])
        return df_loaded
    except FileNotFoundError:
        st.error("Error: 'RAG/synthetic_data_blood_bank.csv' not found. Please ensure the CSV file is in the correct location.")
        st.stop() 
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check your CSV file.")
        st.stop()

# load the dataframe once when the module is imported.
df = _load_data_internal()

# function to get unique values in a column
def get_unique_values(column_name: str) -> dict:
    """
    Retrieves all unique, non-null values from a specified column in the dataset.
    Returns a dictionary with 'result' or 'error'.
    """
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found in the dataset."}
    
    unique_vals = df[column_name].dropna().unique().tolist()
    return {"result": unique_vals}

def query_data(
    filters: dict = None,
    aggregations: dict = None,
    group_by: list = None,
    time_resample_period: str = None
) -> dict:
    """
    A powerful and general function to query the blood bank dataset.
    It can filter, aggregate, group, and create time series data based on the provided parameters.
    Returns a dictionary with 'result' or 'error'.
    """
    df_filtered = df.copy() # use the module-level df

    # apply filters
    if filters:
        for column, conditions in filters.items():
            if column not in df_filtered.columns:
                return {"error": f"Invalid column name in filters: {column}"}

            for op, value in conditions.items():
                try:
                    if column == 'TRANSFUSION_DT':
                        if isinstance(value, str):
                            value = pd.to_datetime(value)
                    if op == 'eq':
                        df_filtered = df_filtered[df_filtered[column] == value]
                    elif op == 'neq':
                        df_filtered = df_filtered[df_filtered[column] != value]
                    elif op == 'gt':
                        df_filtered = df_filtered[df_filtered[column] > value]
                    elif op == 'lt':
                        df_filtered = df_filtered[df_filtered[column] < value]
                    elif op == 'gte':
                        df_filtered = df_filtered[df_filtered[column] >= value]
                    elif op == 'lte':
                        df_filtered = df_filtered[df_filtered[column] <= value]
                    elif op == 'contains':
                        df_filtered = df_filtered[df_filtered[column].astype(str).str.contains(value, case=False, na=False)]
                    else:
                        return {"error": f"Unsupported operator '{op}' for column '{column}'."}
                except Exception as e:
                    return {"error": f"Failed to apply filter on column '{column}' with operator '{op}' and value '{value}': {e}"}

    if df_filtered.empty:
        return {"result": "No data found for the given criteria."}

    # default to a simple record count if no aggregation is specified
    if not aggregations:
        return {"result": {"record_count": len(df_filtered)}}

    try:
        period_map = {"D": "D", "W": "W", "M": "ME"}
        period = period_map.get(time_resample_period)

        if time_resample_period and group_by:
            grouped_resampled_df = df_filtered.set_index('TRANSFUSION_DT').groupby(group_by).resample(period)

            agg_result = grouped_resampled_df.agg(aggregations)

            # flatten MultiIndex and format for LLM readability
            formatted_results = {}
            for index, row_data in agg_result.iterrows():
                category_value = index[:-1] if len(group_by) > 1 else index[0] # handle single or multiple group_by columns
                date_str = index[-1].strftime('%Y-%m-%d')

                # create a unique key for the dictionary based on group_by values and date
                key_parts = [str(cv) for cv in category_value] if isinstance(category_value, tuple) else [str(category_value)]
                formatted_key = f"{' - '.join(key_parts)} - {date_str}"

                # ensure row_data is a dictionary
                if isinstance(row_data, pd.Series):
                    formatted_results[formatted_key] = row_data.to_dict()
                else: # for single aggregation, row_data might be a scalar
                    agg_col, agg_func = list(aggregations.items())[0]
                    formatted_results[formatted_key] = {f"{agg_col}_{agg_func}": row_data}

            return {"result": formatted_results}

        elif time_resample_period:
            # only time resampling
            result_df = df_filtered.set_index('TRANSFUSION_DT').resample(period).agg(aggregations)
            result_df.index = result_df.index.strftime('%Y-%m-%d')
            return {"result": result_df.to_dict(orient='index')}

        elif group_by:
            # only grouping
            result_df = df_filtered.groupby(group_by).agg(aggregations)
            if isinstance(result_df.index, pd.MultiIndex):
                # convert MultiIndex tuples to a readable string format
                result_df.index = result_df.index.map(lambda x: str(x) if isinstance(x, tuple) else x)
            return {"result": result_df.to_dict(orient='index')}
        else:
            # simple aggregation without grouping or time resampling
            result_series = df_filtered.agg(aggregations)
            return {"result": result_series.to_dict()}

    except Exception as e:
        return {"error": f"An error occurred during data processing: {str(e)}"}