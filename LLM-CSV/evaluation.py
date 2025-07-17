import streamlit as st
import pandas as pd
from bert_score import score as bert_score_score

# import run_conversation from conversation_manager for evaluation
from conversation_manager import run_conversation

# define test cases for BERTScore (Natural Language Response) Evaluation
TEST_CASES = [
{
    "question": "How many female patients received Plasma Thawed?",
    "expected_response": "A total of 1,970 female patients received Plasma Thawed products."
},
{
    "question": "What is the maximum transfused volume recorded?",
    "expected_response": "The highest single transfused volume recorded was 11 units."
},
{
    "question": "How many patients have AB-negative blood type?",
    "expected_response": "There are 2,492 patients with AB-negative blood type."
},
{
    "question": "How many transfusions occurred in 2021?",
    "expected_response": "There were 19,941 transfusions recorded in the year 2021."
},
{
    "question": "What is the average age of male patients?",
    "expected_response": "The average age of male patients who received transfusions is approximately 53.78 years."
},
{
    "question": "What is the total volume of Bone Marrow transfusions?",
    "expected_response": "The total volume of Bone Marrow transfusions is 24,527 units."
},
{
    "question": "How many patients older than 50 received any product?",
    "expected_response": "A total of 10,789 patients older than 50 received transfusion products."
},
{
    "question": "What is the earliest transfusion date recorded?",
    "expected_response": "The earliest transfusion date recorded in the dataset is January 1, 2021."
},
{
    "question": "How many transfusions were done in KFHI-Adult Cardiac Surgery?",
    "expected_response": "There were 1,351 transfusions performed in the KFHI-Adult Cardiac Surgery department."
},
{
    "question": "What is the average transfused volume for patients with O blood type?",
    "expected_response": "The average transfused volume for patients with O blood type is approximately 6.03 units."
}
]

# function to run BERTScore evaluation
def evaluate_bert_score():
    st.subheader("Model Natural Language Response (BERTScore) Evaluation Results")
    st.info("Note: Expected responses are based on the provided `synthetic_data_blood_bank.csv` and may vary if the data changes.")
    
    bert_scores = []
    total_questions = len(TEST_CASES)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        expected_response = test_case["expected_response"]

        status_text.text(f"Evaluating LLM Response: '{question}' ({i+1}/{total_questions})")
        
        # get the actual generated response from the full conversation flow
        generated_response = run_conversation(question)
        
        if generated_response and generated_response != "Sorry, I encountered an error while trying to connect to the AI service.":
            if not generated_response.strip():
                gen_res_list = ["<EMPTY_RESPONSE>"]
            else:
                gen_res_list = [generated_response]
            
            if not expected_response.strip():
                exp_res_list = ["<EMPTY_EXPECTED>"]
            else:
                exp_res_list = [expected_response]

            P, R, F1 = bert_score_score(gen_res_list, exp_res_list, lang="en", verbose=False)
            
            bert_scores.append({
                "Question": question,
                "Generated Response": generated_response,
                "Expected Response": expected_response,
                "BERTScore Precision": P.mean().item(),
                "BERTScore Recall": R.mean().item(),
                "BERTScore F1": F1.mean().item()
            })
        else:
            bert_scores.append({
                "Question": question,
                "Generated Response": generated_response or "No response/Error during generation",
                "Expected Response": expected_response,
                "BERTScore Precision": 0.0,
                "BERTScore Recall": 0.0,
                "BERTScore F1": 0.0
            })
        
        progress_bar.progress((i + 1) / total_questions)

    status_text.empty()
    st.markdown("---")
    st.subheader("BERTScore Summary")

    if bert_scores:
        scores_df = pd.DataFrame(bert_scores)
        avg_precision = scores_df["BERTScore Precision"].mean()
        avg_recall = scores_df["BERTScore Recall"].mean()
        avg_f1 = scores_df["BERTScore F1"].mean()

        st.info(f"**Average BERTScore Precision: {avg_precision:.4f}**")
        st.info(f"**Average BERTScore Recall: {avg_recall:.4f}**")
        st.info(f"**Average BERTScore F1: {avg_f1:.4f}**")

        st.markdown("---")
        st.subheader("Detailed BERTScore Results:")
        st.dataframe(scores_df.set_index("Question"), use_container_width=True)
        st.markdown("*(Note: BERTScore measures semantic similarity. A low score might indicate a functionally incorrect answer if the tool call was wrong, or a poorly phrased correct answer.)*")
    else:
        st.warning("No BERTScore results to display. Check for API errors or empty responses.")