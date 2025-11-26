# P-GALM: Probabilistic Graph-Augmented Language Model

P-GALM is a research prototype that connects the ScienceQA dataset with a verbalized probabilistic graphical model (vPGM) pipeline. It uses Large Language Models (LLMs) to perform structured reasoning by filling in a Bayesian network template with probabilities and text justifications.

## Quick Start

### Prerequisites

- Python 3.8+
- An OpenAI API Key

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set your OpenAI API Key**:
    You must set the `OPENAI_API_KEY` environment variable.
    
    **Windows (PowerShell):**
    ```powershell
    $env:OPENAI_API_KEY="sk-..."
    ```
    
    **Linux/macOS:**
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

    Alternatively, create a `.env` file in the root directory (if using `python-dotenv`):
    ```
    OPENAI_API_KEY=sk-...
    ```

### Running the System

1.  Start the server:
    ```bash
    python server.py
    ```
    This will load the ScienceQA dataset (downloading it on first run) and start the backend API.

2.  Open the Web UI:
    Open `web_ui/index.html` in your web browser.

3.  **Use the UI**:
    -   Browse or search for questions.
    -   Click **"Run Inference"** to trigger the vPGM pipeline.
    -   View the results in the UI (Answer Posterior, Latent Variables).
    -   **Monitor the terminal** where `server.py` is running to see the full JSON response and detailed logs.

### Evaluation

To evaluate the system's performance on a sample of the dataset:

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook evaluate_pgalm.ipynb
    ```
2.  Run the cells to:
    -   Load the validation dataset.
    -   Select a random sample of questions.
    -   Run the vPGM inference pipeline.
    -   Calculate accuracy and analyze reasoning quality.

## How It Works

P-GALM implements a **Verbalized Probabilistic Graphical Model (vPGM)**. Instead of using traditional numerical tables for Conditional Probability Distributions (CPDs), it uses an LLM to generate the posterior distributions directly based on textual instructions and observed data.

### The Pipeline

1.  **Loader (`scienceqa_vpgm_loader.py`)**: Loads a question from the ScienceQA dataset and maps it to a "skeleton" JSON structure.
2.  **Prompt Builder (`build_vpgm_llm_prompt.py`)**: Constructs a detailed prompt that includes:
    -   The observed data (Question, Options, Context, Lecture).
    -   The vPGM template structure.
    -   Instructions for each latent variable.
3.  **LLM Client (`vpgm_llm_client.py`)**: Sends the prompt to the LLM (e.g., GPT-4). The LLM acts as the inference engine.
4.  **Inference**: The LLM "fills in" the missing parts of the skeleton:
    -   It estimates the probability distribution for each latent variable.
    -   It provides a text justification for these probabilities.
    -   It computes the final answer posterior.

### Latent Variables

The system uses a 4-variable latent structure (`scienceqa_vpgm_4latent_generic`) to model the reasoning process:

1.  **Z1_relevance_assessment**:
    -   **Meaning**: How relevant are the input data (image, context, lecture) to the question?
    -   **Goal**: Filter out noise and focus on useful signals.

2.  **Z2_knowledge_quality**:
    -   **Meaning**: How accurate, complete, and trustworthy is the knowledge derived from the sources?
    -   **Goal**: Assess the reliability of the information being used.

3.  **Z3_question_clarity**:
    -   **Meaning**: Is the question clear and unambiguous given the relevant data?
    -   **Goal**: Identify if the problem itself is well-posed.

4.  **Z4_logical_reasoning**:
    -   **Meaning**: How strong is the step-by-step logical reasoning used to evaluate the options?
    -   **Goal**: Evaluate the coherence of the deduction process.

### Posterior Calculation

In this "verbalized" approach, the **posterior is calculated by the LLM itself** during generation.

-   The prompt explicitly instructs the LLM to follow the dependencies in the graph (e.g., "Using the relevance (Z1) and knowledge quality (Z2), evaluate question clarity (Z3)...").
-   For each variable, the LLM outputs a **`state_probabilities`** dictionary (e.g., `{"low": 0.1, "high": 0.9}`) and a **`justification`**.
-   The final **Answer Posterior (`Y`)** is derived by the LLM weighing the options based on the outcome of the logical reasoning node (`Z4`).

This allows the system to capture uncertainty and provide explainable reasoning steps ("justifications") for every part of the decision-making process.
