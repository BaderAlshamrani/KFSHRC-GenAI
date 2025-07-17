import streamlit as st
import pandas as pd
import os
from pathlib import Path

from functools import partial
from src.llm import generate_answer
from src.vector_store import VectorStore
from src.constants import TEST_QUESTIONS_PER_PDF, CHUNK_RECORDS_FILE
from src.evaluation import evaluate_bert_score_rag

st.set_page_config(
    page_title="LLM chat assistant KFSHRC",
    page_icon="🩸",
    layout="wide"
)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# --- Data Loading and Caching (for the CSV file) ---
@st.cache_data
def load_data():
    csv_path = r'RAG\synthetic_data_blood_bank.csv'
    if not os.path.exists(csv_path):
        st.error(f"Error: CSV file not found at {csv_path}. Please ensure it exists.")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    df['TRANSFUSION_DT'] = pd.to_datetime(df['TRANSFUSION_DT'])
    return df

df = load_data()
vs = VectorStore(st=st)

def main():
    st.title("LLM chat assistant KFSHRC") 

    # initialize vector store and ingest data if not already done
    if 'vector_store_built' not in st.session_state:
        st.session_state.vector_store_built = False
        
    if not CHUNK_RECORDS_FILE.exists():
        st.error(f"Chunk records file not found at {CHUNK_RECORDS_FILE}. Please run your `parse_ingest.py` script first to process your PDFs.")
        return

    if not st.session_state.vector_store_built or vs.collection.count() == 0:
        with st.spinner("Building/Loading vector store (first run can take a few minutes)..."):
            vs.ingest_from_jsonl()
        st.session_state.vector_store_built = True
        if vs.collection.count() > 0:
            st.success(f"Vector store built/loaded successfully with {vs.collection.count()} chunks!")
        else:
            st.error("Vector store is empty. Please ensure `chunks.jsonl` has content and `parse_ingest.py` ran successfully.")


    try:
        pdf_files = sorted([p.name for p in Path("RAG").glob("*.pdf")])
        if not pdf_files:
            st.error("No PDFs found in the `RAG` folder. Please add your documents.")
            return
    except FileNotFoundError:
        st.error("The `RAG` directory was not found. Please create it and add your PDFs.")
        return
        
    # create tabs for UI organization
    chat_tab, eval_tab = st.tabs(["Document Assistant", "Model Evaluations"])

    with chat_tab:
        st.subheader("Chat with Your Document Assistant")
        selected_pdf_for_chat = st.selectbox("Select a PDF to ask questions about", pdf_files, key="chat_pdf_select")

        # display conversation history
        for query, response in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                st.markdown(response)

        # chat input
        user_question = st.chat_input("Ask me anything about the selected PDF...", key="chat_input_main")
        if user_question:
            st.session_state.conversation_history.append((user_question, "..."))
            st.rerun() 

        # process the latest query if it hasn't been answered
        if st.session_state.conversation_history and st.session_state.conversation_history[-1][1] == "...":
            query_to_process, _ = st.session_state.conversation_history[-1]
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context & generating answer..."):

                    contexts = vs.query(query_to_process, k=15, source_pdf=selected_pdf_for_chat)
                    answer = generate_answer(query_to_process, contexts)
                    
                    if contexts:
                        st.markdown("---")
                        st.markdown("**Contexts Used (for debugging):**")
                        for i, ctx in enumerate(contexts):
                            with st.expander(f"Context {i+1} from {ctx['source_pdf']} page {ctx['page']} (distance {ctx['distance']:.2f})"):
                                st.markdown(f"> {ctx['text']}")
                                if ctx.get("image_path") and Path(ctx["image_path"]).exists():
                                    st.image(str(ctx["image_path"]), width=300)
                    st.markdown("---") 

                st.session_state.conversation_history[-1] = (query_to_process, answer)
                st.rerun() # rerun to display assistant's full answer

        # quick Questions to Try
        st.markdown("---")
        with st.expander("Quick Questions to Try"):
            st.markdown("Click an example below to start a conversation:")
            example_cols = st.columns(3) # use columns for example buttons to save space
            examples = [
                "What is the primary objective of this document?",
                "Can you describe the methodology used?",
                "Are there any figures or tables on page 3?",
                "Summarize the conclusions.",
                "What is mentioned about data privacy?",
                "Explain the results section."
            ]
            for idx, example in enumerate(examples):
                with example_cols[idx % 3]:
                    if st.button(example, use_container_width=True, key=f"chat_example_btn_{idx}"):
                        st.session_state.conversation_history.append((example, "..."))
                        st.rerun()
        
        st.markdown("---")
        if st.button("Clear Chat History", help="Clears all messages from the current session.", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

    with eval_tab: 
        st.subheader("Run Model Evaluations")
        st.markdown("Select a PDF to run pre-defined BERTScore evaluations based on specific questions and expected answers.")
        
        pdf_files_for_eval = list(TEST_QUESTIONS_PER_PDF.keys())
        if not pdf_files_for_eval:
            st.warning("No PDF test cases defined in `TEST_QUESTIONS_PER_PDF`. Please populate this dictionary with your PDF filenames and 76 questions each.")
            if pdf_files:
                st.info(f"Available PDFs in your RAG folder: {', '.join(pdf_files)}")

        selected_pdf_for_eval = st.selectbox(
            "Select PDF for Evaluation", 
            pdf_files_for_eval, 
            key="eval_pdf_select",
            help="Choose a PDF whose pre-defined test questions you want to evaluate."
        )

        st.markdown("---")
        if st.button(f"Run BERTScore Evaluation for '{selected_pdf_for_eval}'", type="primary", use_container_width=True):
            if selected_pdf_for_eval:
                evaluate_bert_score_rag(st, vs, selected_pdf_for_eval)
            else:
                st.warning("Please select a PDF to run the evaluation.")
        
        st.markdown("---")
        st.markdown("*(Note: BERTScore evaluation can take time as it re-runs the RAG pipeline for each question.)*")
        
    st.markdown("---") 

if __name__ == "__main__":
    if not CHUNK_RECORDS_FILE.exists():
        print(f"Warning: {CHUNK_RECORDS_FILE} not found. Please run your `parse_ingest.py` script first.")
    
    # The main Streamlit app execution
    main()