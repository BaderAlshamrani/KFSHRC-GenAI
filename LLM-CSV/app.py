import streamlit as st
import pandas as pd
import os

# Import functions and data from our custom modules
from data_handler import df 
from conversation_manager import run_conversation
from evaluation import TEST_CASES, evaluate_bert_score

# --- Set page config ---
st.set_page_config(
    page_title=" KFSHRC LLM Chat Blood Bank Assistant",
    layout="wide"
)

# --- Initialize session state ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# --- Streamlit UI ---
st.title("🩸 LLM Chat Blood Bank Assistant")
chat_tab, eval_tab = st.tabs(["Chat Assistant", "Model Evaluations"])

with chat_tab:
    # Dataset Overview: only the expander
    with st.expander("Click to view detailed available columns and data info"):
        st.markdown(f"**Total Records:** {len(df):,}")
        st.markdown(f"**Date Range:** {df['TRANSFUSION_DT'].min().strftime('%Y-%m-%d')} to {df['TRANSFUSION_DT'].max().strftime('%Y-%m-%d')}")
        st.markdown("""
        The dataset contains the following columns related to blood transfusion records:
        - **ENCNTR_ID**: Unique identifier for a hospital encounter.
        - **MRN**: Unique Medical Record Number for a patient.
        - **AGE**: Patient's age in years at the time of transfusion.
        - **GENDER**: Patient's gender ('F' or 'M').
        - **TRANSFUSED_VOL**: Volume of the blood product transfused.
        - **PRODUCT_CAT**: Category of the blood product (e.g., 'Red Cells', 'Plasma Thawed').
        - **CUR_ABO_CD**: Patient's ABO blood type ('A', 'B', 'AB', 'O').
        - **CUR_RH_CD**: Patient's Rh blood type ('POS' or 'NEG').
        - **TRANSFUSION_DT**: Exact date of the transfusion.
        - **MED_SERVICE**: Medical service/department (e.g., 'MED-Nephrology').
        
        *(Hint: You can also ask me: 'What are the unique values for [column_name]?' for more details!)*
        """)
    
    st.markdown("---") 

    st.subheader("Chat with Your Blood Bank Assistant")

    # Display conversation history
    for query, response in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(response)

    user_query = st.chat_input("Ask me anything about blood transfusion data...", key="chat_input_main")
    if user_query:
        st.session_state.conversation_history.append((user_query, "..."))
        st.rerun() # Rerun to display the user's message immediately

    # Process the latest query if it hasn't been answered yet
    if st.session_state.conversation_history and st.session_state.conversation_history[-1][1] == "...":
        query_to_process, _ = st.session_state.conversation_history[-1]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question and fetching data..."):
                response = run_conversation(query_to_process)
                st.session_state.conversation_history[-1] = (query_to_process, response)
                st.rerun() 

    # Quick Questions to Try
    st.markdown("---")
    with st.expander("Quick Questions to Try"):
        st.markdown("Click an example below to start a conversation:")
        example_cols = st.columns(3) # Use columns for example buttons to save space
        examples = [
            "How much total blood volume was needed for nephrology in April 2021?",
            "What are the different types of blood products?",
            "What was the blood demand trend over time in 2021?",
            "How many transfusions were given to female patients in January 2022?",
            "Show me the monthly trend of Red Cells transfusions",
            "What's the average age of patients who received Platelets?",
            "How many O negative blood type patients received transfusions?",
            "Show me the monthly blood demand trend by product category in 2021."
        ]
        # Distribute example buttons across columns
        for idx, example in enumerate(examples):
            with example_cols[idx % 3]: # Cycle through the 3 columns
                if st.button(example, use_container_width=True, key=f"example_btn_{idx}"):
                    st.session_state.conversation_history.append((example, "..."))
                    st.rerun()
    
    # Clear Chat History Button
    st.markdown("---")
    if st.button("Clear Chat History", help="Clears all messages from the current session.", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

with eval_tab: # Content for the "Model Evaluations"
    st.subheader("Model Evaluation Tools")
    st.markdown("""
    Run these evaluations to check the model's performance on predefined test cases.
    - **BERTScore Evaluation**: Measures the semantic similarity between the model's generated natural language response and an expected reference response.
    """)
    
    if st.button("Run BERTScore Evaluation", type="secondary", use_container_width=True):
        st.session_state.conversation_history = [] 
        evaluate_bert_score()

st.markdown("---") #
st.caption("Developed for KFSHRC Blood Bank LLM Chat")