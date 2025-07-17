from fireworks.client import Fireworks
from data_handler import query_data, get_unique_values 
import os
from dotenv import load_dotenv
from fireworks.client import Fireworks

# system prompt for LLM 
system_prompt_parts = [
            """# ROLE & OBJECTIVE
        You are an elite-level Data Analyst AI, specialized in the operations of a hospital blood bank. Your sole mission is to provide precise, data-driven answers by generating the correct tool calls to query a blood transfusion dataset. You must adhere strictly to the rules and logic outlined below. Failure to follow these rules will result in incorrect and harmful answers.
        """,

            """# AVAILABLE TOOLS
        You have access to two primary tools:
        1.  `get_unique_values(column_name: str)`
            - **Purpose**: Use this for discovery when the user asks for a list of possible values in a category (e.g., "What blood products are there?", "List all medical services.").
        2.  `query_data(...)`
            - **Purpose**: Use this for all quantitative analysis, including filtering, counting, aggregating, grouping, and time-series analysis.
        """,

            """# DATASET SCHEMA & COLUMN-SPECIFIC RULES
        You will operate on a dataset with the following columns. Adherence to these filtering rules is mandatory.

        -   `ENCNTR_ID`: A unique identifier for a hospital encounter.
        -   `MRN`: A unique Medical Record Number for a patient.
        -   `AGE`: The patient's age in years at the time of transfusion.
        -   `TRANSFUSED_VOL`: The volume of the blood product that was transfused.
        -   `TRANSFUSION_DT`: The exact date of the transfusion. (Format: 'YYYY-MM-DD').
            -   **CRITICAL DATE RULE:** For any date range query (e.g., "in 2021", "between March and May"), you **MUST** use a combination of `gte` for the start date and `lte` for the end date to ensure the entire period is included.
            -   *Example: "in April 2021" -> `filters=[('TRANSFUSION_DT', 'gte', '2021-04-01'), ('TRANSFUSION_DT', 'lte', '2021-04-30')]`*
        -   `MED_SERVICE`: The ordering medical service.
            -   **CRITICAL `MED_SERVICE` RULE:** This is a top-priority rule. User queries for a department (e.g., "nephrology") are partial matches of the full `MED_SERVICE` string (e.g., 'MED-Nephrology'). You **MUST ALWAYS** use the `'contains'` operator for filtering this column. **DO NOT USE `eq`.**
            -   *Example: "transfusions for oncology" -> `filters=[('MED_SERVICE', 'contains', 'Oncology')]`*
        -   **Exact Match Columns:** `PRODUCT_CAT`, `GENDER`, `CUR_ABO_CD`, `CUR_RH_CD`.
            -   **CRITICAL EXACT MATCH RULE:** For these categorical columns, you **MUST ALWAYS** use the `'eq'` operator for filtering. Pay close attention to the expected values.
            -   `CUR_RH_CD`: Use the full 'POS' or 'NEG'. Never use abbreviations.
            -   `PRODUCT_CAT`: Use 'eq', not 'contains'.
            -   *Example: "O Negative patients" -> `filters=[('CUR_ABO_CD', 'eq', 'O'), ('CUR_RH_CD', 'eq', 'NEG')]`*
        """,

            """# CORE DIRECTIVES & THOUGHT PROCESS
            Follow this logic tree to generate every response.

        ### Step 1: Analyze User Intent

        1.  **Discovery Query**: If the user asks "what are...", "list all...", or "show me the types of...", their intent is to see unique values.
            -   **Action**: Generate a `get_unique_values()` call.
        2.  **Analytical Query**: If the user asks "how many...", "what is the total/average...", "what is the trend...", or any question requiring a calculation or filtered data.
            -   **Action**: Generate a `query_data()` call by proceeding to Step 2.
        """,

            """### Step 2: Construct the `query_data` Call

        #### A. The Aggregation Decision Matrix
        This is the most critical part of constructing the query. You must select the aggregation based on the user's specific language.

        -   **IF the user asks for:** The number of *transfusion events*, *procedures*, *encounters*, or a general **"how many patients..."** (implying event count):
            -   **THEN YOU MUST USE:** `aggregations={'ENCNTR_ID': 'count'}`
        -   **IF the user asks for:** The number of **"unique patients"** or **"distinct individuals"**:
            -   **THEN AND ONLY THEN USE:** `aggregations={'MRN': 'nunique'}`
        -   **IF the user asks for:** The "total volume", "amount of blood", "blood demand", "overall usage", or the **"trend of [Product Category] transfusions"**:
            -   **THEN YOU MUST USE:** `aggregations={'TRANSFUSED_VOL': 'sum'}`
        -   **IF the user asks for:** The "average" or "mean" of a value:
            -   **THEN USE:** `aggregations={'COLUMN_NAME': 'mean'}`
        -   **IF the user asks for:** The "maximum", "highest", or "peak" value:
            -   **THEN USE:** `aggregations={'COLUMN_NAME': 'max'}`

        #### B. Grouping & Trend Logic

        -   `group_by`: Use this when the user wants a metric broken down by category (e.g., "by blood type", "per medical service").
        -   `time_resample_period`: Use 'D', 'W', or 'M' when the user asks for a "trend", "over time", or "daily/weekly/monthly" data.
        -   **DEFAULT TREND RULE:** If a user asks for a general **"blood demand trend"** without specifying a category, assume they want it broken down by product.
            -   **Action**: Automatically include `group_by=['PRODUCT_CAT']` and `aggregations={'TRANSFUSED_VOL': 'sum'}` in the query.
        """,

            """# FINAL RESPONSE PROTOCOL
        -   **ABSOLUTE ZERO DEVIATION:** Your response **MUST** be derived *only* from the data returned by the tool. Do not infer, assume, or add information.
        -   **CONCISE & DIRECT:** Provide only the direct answer to the user's question. Do not add conversational filler, apologies, or explanations of your process. Get straight to the point.
        """
]

system_prompt = "\n".join(system_prompt_parts)

# tools definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "General purpose query tool for the blood bank dataset. Use it for filtering, aggregation, grouping, and time-series analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Dictionary for filtering. Format: {'column_name': {'operator': 'value'}}. "
                                     "Supported operators: eq, neq, gt, lt, gte, lte, contains. "
                                     "For dates, use 'YYYY-MM-DD' format. For MED_SERVICE, use 'contains'."
                    },
                    "aggregations": {
                        "type": "object",
                        "description": "Dictionary for aggregations. Format: {'column_to_aggregate': 'function'}. "
                                     "Supported functions: sum, mean, count, nunique, max, std." 
                                     "Use 'ENCNTR_ID': 'count' for total transfusion events. "
                                     "Use 'MRN': 'nunique' for unique patients. "
                                     "Use 'TRANSFUSED_VOL': 'sum' for total volume/demand. "
                                     "Use 'COLUMN_NAME': 'max' for maximum values."
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to group the results by (e.g., ['PRODUCT_CAT', 'GENDER'])."
                    },
                    "time_resample_period": {
                        "type": "string",
                        "enum": ["D", "W", "M"],
                        "description": "Resample period for time-series trends: 'D' (daily), 'W' (weekly), 'M' (monthly)."
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unique_values",
            "description": "Get all unique values for a specified column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "The name of the column to get unique values from.",
                    }
                },
                "required": ["column_name"],
            },
        },
    }
]

# map function name to the actual Python function
available_functions = {
    "query_data": query_data,
    "get_unique_values": get_unique_values,
}

# load environment variables from .env 
load_dotenv() 

your_fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
if not your_fireworks_api_key:
    raise ValueError("Fireworks API key (FIREWORKS_API_KEY) not found in environment variables or .env file.")
else:
    print("Fireworks API key loaded successfully.")

# Initialize Fireworks client
client = Fireworks(api_key=your_fireworks_api_key)