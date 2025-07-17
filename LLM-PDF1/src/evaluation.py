import pandas as pd
from bert_score import score as bert_score_score
from src.llm import generate_answer
from src.constants import TEST_QUESTIONS_PER_PDF

def evaluate_bert_score_rag(st, vs, selected_pdf_for_eval: str):
    st.subheader(f"BERTScore Evaluation Results for: {selected_pdf_for_eval}")
    
    test_cases_for_pdf = TEST_QUESTIONS_PER_PDF.get(selected_pdf_for_eval, [])
    if not test_cases_for_pdf:
        st.warning(f"No test cases defined for '{selected_pdf_for_eval}'. Please add questions and expected responses to `TEST_QUESTIONS_PER_PDF`.")
        return

    bert_scores = []
    total_questions = len(test_cases_for_pdf)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, test_case in enumerate(test_cases_for_pdf):
        question = test_case["question"]
        expected_response = test_case["expected_response"]

        status_text.text(f"Evaluating LLM Response: {question} ({i+1}/{total_questions})")
        
        # get contexts for the current question and PDF
        contexts = vs.query(question, k=15, source_pdf=selected_pdf_for_eval)
        
        # get the actual generated response from the RAG pipeline
        generated_response = generate_answer(question, contexts)
        
        if generated_response and generated_response != "Sorry, I was unable to generate an answer due to an API error.":
            # BERTScore needs lists of strings
            P, R, F1 = bert_score_score([generated_response], [expected_response], lang="en", verbose=False)
            
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
                "Generated Response": generated_response or "[Empty/Error Response]",
                "Expected Response": expected_response,
                "BERTScore Precision": 0.0,
                "BERTScore Recall": 0.0,
                "BERTScore F1": 0.0
            })
        
        progress_bar.progress((i + 1) / total_questions)

    status_text.empty()
    st.markdown("---")
    st.subheader("Summary")

    if bert_scores:
        scores_df = pd.DataFrame(bert_scores)
        avg_precision = scores_df["BERTScore Precision"].mean()
        avg_recall = scores_df["BERTScore Recall"].mean()
        avg_f1 = scores_df["BERTScore F1"].mean()

        st.info(f"**Average BERTScore Precision: {avg_precision:.4f}**")
        st.info(f"**Average BERTScore Recall: {avg_recall:.4f}**")
        st.info(f"**Average BERTScore F1: {avg_f1:.4f}**")

        st.markdown("---")
        st.subheader("Detailed Results:")
        st.dataframe(scores_df.set_index("Question"), use_container_width=True)
        st.markdown("*(Note: BERTScore measures semantic similarity. A low score might indicate a functionally incorrect answer, or a poorly phrased correct answer.)*")
    else:
        st.warning("No BERTScore results to display. This might mean no test cases were loaded or all API calls failed.")
