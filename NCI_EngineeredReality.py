# Joseph Teague, Intro to AI w/ Brian Bird

import requests
import json
import time
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# --- Configuration and Constants ---

# Load environment variables from a .env file
load_dotenv()

# Load the API key from the environment.
# The script will check if it's missing and show a warning.
API_KEY = os.getenv("GEMINI_API_KEY") or ""
GEMINI_MODEL = "gemini-flash-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Create a reusable session for all API calls
api_session = requests.Session()
api_session.headers.update({"x-goog-api-key": API_KEY})

# Maximum character length before triggering summarization
MAX_TEXT_LENGTH = 15000


# NCI Engineered Reality Scoring System Criteria (Questions 1-20)
SCORING_CRITERIA = [
    "Timing: Does the timing feel suspicious or coincidental with other events?",
    "Emotional Manipulation: Does it provoke fear, outrage, or guilt without solid evidence?",
    "Uniform Messaging: Are key phrases or ideas repeated across media?",
    "Missing Information: Are alternative views or critical details excluded?",
    "Simplistic Narratives: Is the story reduced to 'good vs. evil' frameworks?",
    "Tribal Division: Does it create an 'us vs. them' dynamic?",
    "Authority Overload: Are questionable 'experts' driving the narrative?",
    "Call for Urgent Action: Does it demand immediate decisions without reflection?",
    "Overuse of Novelty: Is the event framed as shocking or unprecedented?",
    "Financial/Political Gain: Do powerful groups benefit disproportionately?",
    "Suppression of Dissent: Are critics silenced or labeled negatively?",
    "False Dilemmas: Are only two extreme options presented?",
    "Bandwagon Effect: Is there pressure to conform because 'everyone is doing it'?",
    "Emotional Repetition: Are the same emotional triggers repeated excessively?",
    "Cherry-Picked Data: Are statistics presented selectively or out of context?",
    "Logical Fallacies: Are flawed arguments used to dismiss critics?",
    "Manufactured Outrage: Does outrage seem sudden or disconnected from facts?",
    "Framing Techniques: Is the story shaped to control how you perceive it?",
    "Rapid Behavior Shifts: Are groups adopting symbols or actions without clear reasoning?",
    "Historical Parallels: Does the story mirror manipulative past events?",
]

# NCI Interpretation Ranges
SCORE_RANGES = [
    (0, 25, "Low likelihood of a PSYOP"),
    (26, 50, "Moderate likelihood - look deeper"),
    (51, 75, "Strong likelihood - manipulation likely"),
    (76, 100, "Overwhelming signs of a PSYOP"),
]

# --- Helper Functions ---

def fetch_and_clean_url(url: str) -> str | None:
    """
    Fetches content from a URL, parses the HTML, and extracts clean text.
    Returns the text content or None if fetching fails.
    """
    print(f"\nFetching content from URL: {url}...")
    try:
        # Use a common user-agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Use BeautifulSoup to parse the HTML and extract text from paragraph tags
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # A simple but effective way to get article text is to join all paragraphs
        paragraphs = soup.find_all('p')
        article_text = "\n".join([p.get_text() for p in paragraphs])

        print("Successfully extracted text from URL.")
        return article_text

    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to fetch URL: {e}")
        return None

def read_text_from_file(file_path: str) -> str | None:
    """Reads text content from a given file path."""
    print(f"\nReading content from file: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("Successfully read text from file.")
        return content
    except (IOError, UnicodeDecodeError) as e:
        print(f"[Error] Failed to read file: {e}")
        return None

# --- API Interaction Function ---

def make_gemini_request(prompt: str, max_retries: int = 5) -> int:
    """
    Sends a request to the Gemini API, requesting a structured JSON response
    containing only the score. Implements exponential backoff for retries.

    The LLM is given a system instruction to act as a scoring agent, and a JSON
    schema to ensure the output is a single integer (1-5).
    """
    system_prompt = (
        "You are an expert analytical scoring engine for the NCI Engineered Reality "
        "Scoring System. Your task is to analyze the provided text against a specific "
        "criterion and assign a score from 1 to 5. "
        "Score 1: Not Present (No signs of manipulation). "
        "Score 5: Overwhelmingly Present (Clear, strong evidence of manipulation). "
        "You must return ONLY a JSON object conforming to the schema with the score."
    )

    # Define the JSON schema for the response structure
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "score": {
                "type": "INTEGER",
                "description": "The NCI score from 1 to 5 based on the criterion and text."
            }
        },
        "required": ["score"],
    }

    # Construct the full payload
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    for attempt in range(max_retries):
        try:
            # Using 5 seconds timeout for a robust call
            response = api_session.post(API_URL, json=payload, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # Extract the raw JSON text from the response structure
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

            if not json_text:
                raise ValueError("API returned an empty or malformed text response.")

            # The model is instructed to return *only* the JSON structure
            parsed_json = json.loads(json_text)
            score = parsed_json.get('score')

            if isinstance(score, int) and 1 <= score <= 5:
                print(f"  -> Score received: {score}")
                return score
            else:
                # Log non-conformant score and try again
                print(f"  [Warning] LLM returned non-conformant score: {score}. Retrying...")
                raise ValueError("Score is not a valid integer between 1 and 5.")

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  [Error] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [Error] Final attempt failed after {max_retries} retries. Defaulting to score 1.")
                return 1 # Fallback to the lowest score to proceed

    return 1 # Should not be reached, but ensures function always returns an integer

def summarize_large_text(text: str, max_retries: int = 3) -> str:
    """
    Summarizes text that is too long for direct analysis.
    """
    print("\n--- Text is very long. Performing pre-analysis summarization... ---")
    system_prompt = (
        "You are an expert summarizer. Your task is to read the following text and "
        "produce a concise but comprehensive summary. The summary must capture the "
        "key arguments, emotional tone, specific claims, and any potentially "
        "manipulative language used. The goal is to create a summary that is "
        "detailed enough for a subsequent forensic analysis of its content."
    )
    prompt = f"Please summarize the following text:\n\n{text}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    for attempt in range(max_retries):
        try:
            response = api_session.post(API_URL, json=payload, timeout=60) # Longer timeout for summarization
            response.raise_for_status()
            result = response.json()
            summary = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

            if summary:
                print("--- Summarization complete. Proceeding with analysis. ---")
                return summary
            else:
                raise ValueError("API returned an empty summary.")

        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  [Error] Summarization attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [Error] Summarization failed after {max_retries} retries.")
                print("  Proceeding with a truncated version of the original text.")
                return text[:MAX_TEXT_LENGTH] # Fallback to truncated text

    return text[:MAX_TEXT_LENGTH] # Should not be reached
# --- Main Logic ---

def score_message(message: str) -> tuple[int, list[dict]]:
    """
    Iterates through all 20 criteria, sends a separate API request for each,
    and calculates the total score.
    """
    total_score = 0
    detailed_scores = []
    
    print("\n--- Starting NCI Analysis (20 LLM Calls) ---")
    print(f"Target Text:\n---\n{message[:150]}...\n---")

    for i, criterion in enumerate(SCORING_CRITERIA):
        category = f"Criterion {i + 1}/{len(SCORING_CRITERIA)}: {criterion.split(':')[0]}"
        print(f"\nEvaluating {category}")

        # Construct the specific prompt for the LLM
        prompt = (
            f"Analyze the following text against this criterion: '{criterion}'. "
            f"Provide a score from 1 (Not Present) to 5 (Overwhelmingly Present) in the 'score' field of the JSON.\n\n"
            f"TEXT TO ANALYZE: {message}"
        )

        score = make_gemini_request(prompt)
        total_score += score
        
        detailed_scores.append({
            "id": i + 1,
            "criterion": criterion,
            "score": score
        })
        
        # Add a 5-second delay to safely stay under the API rate limit (Be nice to API or it will cry))
        time.sleep(5)

    return total_score, detailed_scores

def generate_report(total_score: int, detailed_scores: list[dict], message: str):
    """
    Generates and prints the final report based on the total score.
    """
    print("\n" + "="*50)
    print("      NCI ENGINEERED REALITY SCORING REPORT")
    print("="*50)

    # Find the interpretation
    interpretation = "Score outside of defined range."
    for low, high, result in SCORE_RANGES:
        if low <= total_score <= high:
            interpretation = result
            break

    # Summary
    print("\n--- SUMMARY ---")
    print(f"Total NCI Score: {total_score}/100")
    print(f"Likelihood of PSYOP: {interpretation}")
    print("\n" + "-"*50)

    # Detailed Breakdown
    print("\n--- DETAILED CRITERIA BREAKDOWN (Score 1-5) ---")
    for item in detailed_scores:
        # Pad the score for alignment
        score_str = str(item['score']).ljust(1)
        print(f"[ {score_str} ] {item['criterion']}")

    print("\n" + "-"*50)
    print("NOTE: This score is a heuristic generated by an LLM and is for")
    print("educational and analytical purposes only. Consult reliable sources.")
    print("="*50)


def main():
    """
    Runs the main application loop, collecting input and initiating scoring.
    """
    if not API_KEY:
        print("ALERT: API_KEY is empty. The application expects the runtime environment to provide the key.")
        print("Proceeding with API calls. If they fail, ensure the key is correctly provided.")

    print("\n" + "="*70)
    print("NCI ENGINEERED REALITY PSYOP SCORER - AI Term Project")
    print("This app analyzes a text message against 20 criteria to assess its")
    print("likelihood of being a psychological operation (PSYOP).")
    print("="*70)

    input_text = ""
    analyzed_text = ""
    while not analyzed_text:
        print("\nPlease paste a URL, a file path, or the raw text you want to analyze.")
        print("(Press Enter on an empty line to submit):")

        user_input = input().strip()
        if not user_input:
            print("\nInput cannot be empty. Please try again.")
            continue

        # Check if the input is a URL
        if user_input.startswith(('http://', 'https://')):
            analyzed_text = fetch_and_clean_url(user_input)
            if not analyzed_text:
                print("Could not process the URL. Please try another URL or paste the text directly.")
                continue # Go back to the start of the loop
        # Check if the input is a file path
        elif os.path.isfile(user_input):
            analyzed_text = read_text_from_file(user_input)
            if not analyzed_text:
                print("Could not process the file. Please check the path or paste the text directly.")
                continue
        else:
            # Handle multi-line pasted text
            lines = [user_input]
            while (line := input()):
                lines.append(line)
            analyzed_text = "\n".join(lines).strip()
    
    # If the text is too long, summarize it first
    if len(analyzed_text) > MAX_TEXT_LENGTH:
        final_text_to_analyze = summarize_large_text(analyzed_text)
        # Add a delay after summarization to avoid hitting rate limits immediately
        time.sleep(5)
    else:
        final_text_to_analyze = analyzed_text

    # Run the analysis
    total_score, detailed_scores = score_message(final_text_to_analyze)
    
    # Generate the final output
    generate_report(total_score, detailed_scores, final_text_to_analyze)

if __name__ == "__main__":
    main()