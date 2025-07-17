from pathlib import Path

PDF_DIR = Path("RAG")
OUTPUT_DIR = Path("data")
IMAGES_DIR = OUTPUT_DIR / "images"
CHUNK_RECORDS_FILE = OUTPUT_DIR / "chunks.jsonl"
COLLECTION_NAME = "pdf_chunks" 


# test Cases for BERTScore evaluation 
TEST_QUESTIONS_PER_PDF = {
    "Modeling Patient No-Show History and Predicting Future Outpatient Appointment Behavior.pdf": [
        {
            "question": "What was the primary objective of the study as stated in the abstract?",
            "expected_response": "The primary objective was to develop and test a predictive model that identifies patients with a high probability of missing their outpatient appointments. [1]",
        },
        {
            "question": "Which three variables were consistently related to a patient's no-show probability in all 24 models developed?",
            "expected_response": "The three variables consistently related to a patient's no-show probability in all 24 models were past attendance behavior, the age of the appointment (appointment age), and having multiple appointments scheduled on that day. [1, 4]",
        },
        {
            "question": "What was the estimated total cost of incomplete appointments for the Veterans Health Administration (VHA) in fiscal year (FY) 2008?",
            "expected_response": "The Veterans Health Administration (VHA) estimated the total cost of incomplete appointments in fiscal year (FY) 2008 at $564 million annually. [1]",
        },
        {
            "question": "According to the pilot study results for successfully contacted patients, what was the no-show rate after the intervention was implemented, and what was the expected no-show rate before the intervention?",
            "expected_response": "After the intervention, the no-show rate in the pilot group was reduced from an expected value of 35% to 12.16%. [1, 5]",
        },
        {
            "question": "What modeling technique was adopted for the predictive model development, and why was it chosen?",
            "expected_response": "Logistic regression was adopted as the modeling technique because its coefficients can easily be interpreted and the model could be implemented in an Excel routine. [2]",
        },
    ],
    "Patient No-Show Prediction A Systematic Lit Review.pdf": [
        {
            "question": "What was the primary objective of this systematic literature review, as stated in the abstract?",
            "expected_response": "The primary objective was to conduct a systematic review of the literature on predicting patient no-shows, aiming to establish the current state-of-the-art. [1]",
        },
        {
            "question": "Which predictive modeling technique was identified as the most frequently used in the analyzed articles?",
            "expected_response": "The most widely used predictive modeling technique identified was Logistic Regression (LR), appearing in 30 articles, which is more than 50% of the total. [13]",
        },
        {
            "question": "According to the introduction, what was the estimated annual economic cost caused by patient non-shows in the United Kingdom?",
            "expected_response": "In the United Kingdom, the annual economic cost caused by non-shows was estimated at 600 million pounds. [1]",
        },
        {
            "question": "What is a significant trend observed regarding the size of databases used for no-show prediction models over recent years?",
            "expected_response": "There has been a significant growth in the size of the databases used to build classifiers in recent years, which is attributed to the recent availability of Electronic Health Records (EHR). [1, 8]",
        },
        {
            "question": "What common characteristic of the analyzed studies often biases algorithms and leads to lower accuracy than the attendance rate?",
            "expected_response": "The class imbalance is a common characteristic of all analyzed studies, often biasing algorithms to predict each observation as a 'show,' resulting in accuracy lower than the attendance rate in many cases. [14]",
        },
    ],
    "Predicting appointment misses in hospitals using data analytics.pdf": [
        {
            "question": "What was the main intention behind starting Project Predicting Appointment Misses (PAM)?",
            "expected_response": "Project Predicting Appointment Misses (PAM) was started with the intention of being able to predict the type of patients who would not come for appointments after making bookings. [1]",
        },
        {
            "question": "According to the study's abstract, which types of variables influenced missed appointments, and which previously assumed major contributors did not show a major effect?",
            "expected_response": 'Variables related to "class," "time," and "demographics" had an effect on missed appointments, while previously assumed major contributors such as "age" and "distance" did not have a major effect. [1]',
        },
        {
            "question": "What was the overall no-show rate reported for the hospital in the year 2013?",
            "expected_response": "The overall no-show rate for the hospital in the year 2013 was 18.59%. [2]",
        },
        {
            "question": "What were the primary software tools used for analysis and predictive modeling in this study?",
            "expected_response": "SAS 9.3 & JMP Pro11 were the primary tools used for the analysis and predictive modeling. [2]",
        },
        {
            "question": "Based on the conclusions of the study, what was found regarding the influence of patient age on missed appointments?",
            "expected_response": "Before the study, the hospital assumed that the age group of 21 and below significantly contributed to missed appointments; however, no such trend was found after analyzing the data. [8]",
        },
    ],
    "Prediction of hospital no-show appointments.pdf": [
        {
            "question": "What was the primary objective of this study?",
            "expected_response": "The objective was to use artificial intelligence to build a model that predicts no-shows for individual appointments. [1]",
        },
        {
            "question": "According to the study's findings, which predictor had the highest information-gain ranking for predicting no-shows?",
            "expected_response": "The history of no-shows (noShow-rate) had the highest information-gain ranking at 0.3596. [1, 3]",
        },
        {
            "question": "What was the total sample size of outpatient clinic appointments used in this study, and what was the overall no-show rate?",
            "expected_response": "The sample size was 1,087,979 outpatient clinic appointments, with an overall no-show rate of 11.3% (123,299 instances). [1, 3]",
        },
        {
            "question": "Which two specific artificial intelligence algorithms were used to build the predictive models in this study?",
            "expected_response": "The predictive models were independently built using JRip and Hoeffding tree algorithms. [1]",
        },
        {
            "question": "Based on the results, how did the Hoeffding tree algorithm's performance compare to JRip in terms of AUC?",
            "expected_response": "The Hoeffding tree algorithm achieved an AUC of 0.861, indicating excellent discrimination, while JRip had an AUC of 0.776, characterized as acceptable discrimination power. [1, 5]",
        },
    ],
    "Predictors of outpatients noshow big data analytics using apache spark.pdf": [
        {
            "question": "What was the primary objective of this study, as stated in the abstract?",
            "expected_response": "The aim of this paper was to explore factors that affect no-show rate and can be used to formulate predictions using big data machine learning techniques. [1]",
        },
        {
            "question": "Which machine learning technique performed best in this study, and what were its accuracy and ROC values?",
            "expected_response": "The Gradient Boosting (GB) performed best, resulting in an increase of accuracy and ROC to 79% and 81%, respectively. [1]",
        },
        {
            "question": "What was the total number of outpatient visits included in the study, and what was the overall proportion of no-shows?",
            "expected_response": "A total of 2,011,813 outpatient visits were included, and the overall proportion of no-shows at all outpatients' clinics was 26.71%. [6]",
        },
        {
            "question": "According to the feature importance ranking, what were the top four predictors of no-show appointments?",
            "expected_response": "The top four predictors were: number of no-show appointments, medical department, lead-time, and number of show appointments. [6]",
        },
        {
            "question": "What is mentioned as a key factor to consider when selecting an algorithm for huge datasets, besides performance metrics?",
            "expected_response": "For huge datasets, the time is a factor to select one of the quicker algorithms, considering that the time values of models depends on the choice of algorithms parameters. [10]",
        },
    ],
}
