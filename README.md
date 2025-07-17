# King Faisal Specialist Hospital & Research Centre (KFSHRC) LLM Chat Assistants

This repository contains the project files for two AI-powered chat assistants developed for the King Faisal Specialist Hospital & Research Centre (KFSHRC). These tools are designed to streamline access to critical hospital information, enhancing data-driven decision-making and operational efficiency.

---

## About This Project

This project introduces two advanced AI chat assistants for King Faisal Specialist Hospital & Research Centre (KFSHRC). Both assistants leverage a sophisticated **Retrieval-Augmented Generation (RAG)** framework, each tailored to a specific type of hospital information to make valuable data more accessible.

*   **Structured Data RAG Assistant (Blood Bank):** This assistant applies the RAG model to query and analyze structured data from the hospital's blood bank. It translates natural language questions into precise data retrieval operations, then uses the retrieved data as context to generate insightful, analytical responses about inventory, usage trends, and other key metrics.

*   **Unstructured Document RAG Assistant (PDF Library):** This assistant is engineered to navigate the challenge of unstructured information locked within a vast library of PDF documents. It uses its RAG capabilities to perform semantic searches, locate relevant text passages and images, and generate accurate answers complete with direct source citations.

---

## Key Features

*   **💬 Conversational AI:** Intuitive chat interfaces for seamless interaction with complex data.
*   **🧠 Dual RAG Framework:** Specialized assistants for both structured (database) and unstructured (document) information.
*   **📊 Intelligent Data Analysis:** Surfaces trends and answers from the blood bank database.
*   **📄 Precise Document Retrieval:** Locates exact information and images from PDFs, with verifiable source citations.
*   **🌱 Scalable Architecture:** Built for future expansion with additional data sources and documents.

---

## How KFSHRC Benefits

These AI assistants are powerful tools that will bring big improvements to the hospital, leading to:

*   **Faster, smarter decisions**
*   **Much better work efficiency**
*   **Knowledge for everyone**
*   **Stronger research and patient care**
*   **More accurate, trustworthy answers**

---

## Repository Content

This project includes four primary deliverables:

1.  📄 **`KFSHRC-GenAI-Reprot`**: A comprehensive document detailing the project's objectives, methodology, and outcomes.
2.  🖥️ **`KFSHRC-GenAI-Slieds`**: A slide deck summarizing the project's highlights for stakeholders.
3.  🩸 **`LLM-CSV`**: The Python source code for the RAG assistant that interacts with structured blood bank data.
4.  🔍 **`LLM-PDF`**: The Python source code for the RAG assistant for searching PDF documents.

---

## 🛠️ Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.9+
*   Pip package installer

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/your_username/KFSHRC-LLM-Chat-Assistants.git
    ```
2.  Navigate to the project directory
    ```sh
    cd KFSHRC-LLM-Chat-Assistants
    ```
3.  Install Python packages from the requirements file
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing all necessary libraries, e.g., `pandas`, `langchain`, `streamlit`, `faiss-cpu`, etc.)*

### Usage

To run one of the assistants, use Streamlit from your terminal:

```sh
# To run the blood bank assistant
streamlit run blood_bank_rag_assistant.py

# To run the document search assistant
streamlit run document_rag_assistant.py
