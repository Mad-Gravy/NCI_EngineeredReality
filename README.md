# NCI Engineered Reality PSYOP Scorer

This application is a Python-based tool designed to analyze text for signs of manipulation, based on the "NCI Engineered Reality Scoring System." It uses the Google Gemini API to score a given text against 20 distinct criteria, providing a final score that indicates the likelihood of the text being a psychological operation (PSYOP).

## Features

*   **Multi-Criteria Analysis**: Scores text against 20 detailed criteria, from emotional manipulation to logical fallacies.
*   **Flexible Input**: Accepts input from three sources:
    *   A direct URL to an article.
    *   A local file path (e.g., `C:\docs\my_article.txt`).
    *   Raw text pasted directly into the terminal.
*   **Large Text Handling**: Automatically summarizes very long texts before analysis to ensure efficiency and avoid API limits.
*   **Robust API Interaction**: Includes exponential backoff and retry logic to handle API rate limits and temporary network issues.

## Setup

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites

*   **Python 3.8+**: Ensure you have Python installed. You can download it from python.org.

### 2. Create the Environment File

This project requires a Google Gemini API key.

1.  Obtain an API key from Google AI Studio.
2.  In the project directory, create a new file named `.env`.
3.  Add your API key to the `.env` file in the following format:

    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

**Important**: Never share this file or commit it to version control.

### 3. Install Dependencies

Open your terminal or command prompt and navigate to the project directory. Run the following command to install the necessary Python libraries:

```bash
pip install requests python-dotenv beautifulsoup4
```

## How to Run the Application

1.  Open your terminal or command prompt.
2.  Navigate to the project directory:
    ```bash
    cd c:\Users\grave\Desktop\intro_to_ai\TermProject\
    ```
3.  Run the script using Python:
    ```bash
    python NCI_EngineeredReality.py
    ```

## How to Use

Once the application is running, it will prompt you for input. You can provide one of the following:

*   **To analyze a webpage**: Paste the full URL (e.g., `https://www.example.com/article`) and press Enter.
*   **To analyze a local file**: Paste the full file path (e.g., `C:\Users\grave\Documents\article.txt`) and press Enter.
*   **To analyze raw text**: Paste the text directly into the terminal. After pasting, press Enter on a new, empty line to submit.

The application will then fetch and analyze the text, showing its progress for each of the 20 criteria. Finally, it will print a detailed report with a total score and breakdown.

## Testing

This project includes a suite of unit tests to ensure the code is working correctly. The tests use Python's built-in `unittest` framework and do not make any real API calls.

To run the tests, navigate to the project directory in your terminal and run the following command:

```bash
python -m unittest test_NCI_EngineeredReality.py
```

You should see output indicating that all tests passed successfully.
