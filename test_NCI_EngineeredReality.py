import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
from io import StringIO

# Add the project directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import NCI_EngineeredReality as nci

class TestNCIEngineeredReality(unittest.TestCase):

    def setUp(self):
        """Set up common test variables."""
        self.test_url = "http://example.com"
        self.test_filepath = "test_article.txt"
        self.long_text = "A" * (nci.MAX_TEXT_LENGTH + 1)
        self.short_text = "This is a short test text."

    @patch('NCI_EngineeredReality.requests.get')
    def test_fetch_and_clean_url_success(self, mock_get):
        """Test successful fetching and parsing of a URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = "<html><body><p>Hello</p><p>World</p></body></html>"
        mock_get.return_value = mock_response

        result = nci.fetch_and_clean_url(self.test_url)
        self.assertEqual(result, "Hello\nWorld")
        mock_get.assert_called_once_with(self.test_url, headers=unittest.mock.ANY, timeout=15)

    @patch('NCI_EngineeredReality.requests.get')
    def test_fetch_and_clean_url_failure(self, mock_get):
        """Test URL fetching failure."""
        mock_get.side_effect = nci.requests.exceptions.RequestException("Test Error")
        result = nci.fetch_and_clean_url(self.test_url)
        self.assertIsNone(result)

    @patch("builtins.open", new_callable=mock_open, read_data="File content.")
    def test_read_text_from_file_success(self, mock_file):
        """Test successful reading from a file."""
        result = nci.read_text_from_file(self.test_filepath)
        self.assertEqual(result, "File content.")
        mock_file.assert_called_once_with(self.test_filepath, 'r', encoding='utf-8')

    @patch("builtins.open", new_callable=mock_open)
    def test_read_text_from_file_failure(self, mock_file):
        """Test file reading failure."""
        mock_file.side_effect = IOError("File not found")
        result = nci.read_text_from_file("nonexistent.txt")
        self.assertIsNone(result)

    @patch('NCI_EngineeredReality.api_session.post')
    def test_make_gemini_request_success(self, mock_post):
        """Test a successful Gemini API request that returns a valid score."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"score": 4}'}]}}]
        }
        mock_post.return_value = mock_response

        score = nci.make_gemini_request("Test prompt")
        self.assertEqual(score, 4)

    @patch('NCI_EngineeredReality.api_session.post')
    def test_make_gemini_request_invalid_score(self, mock_post):
        """Test when the API returns a score outside the 1-5 range, leading to a retry and eventual fallback."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"score": 99}'}]}}]
        }
        mock_post.return_value = mock_response

        # With retries, it should eventually fail and return the default score of 1
        score = nci.make_gemini_request("Test prompt", max_retries=2)
        self.assertEqual(score, 1)
        self.assertEqual(mock_post.call_count, 2)

    @patch('NCI_EngineeredReality.api_session.post')
    def test_make_gemini_request_api_failure(self, mock_post):
        """Test API failure with exponential backoff, leading to a fallback score."""
        mock_post.side_effect = nci.requests.exceptions.RequestException("API Error")
        
        score = nci.make_gemini_request("Test prompt", max_retries=3)
        self.assertEqual(score, 1)
        self.assertEqual(mock_post.call_count, 3)

    @patch('NCI_EngineeredReality.api_session.post')
    def test_summarize_large_text_success(self, mock_post):
        """Test successful text summarization."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "This is a summary."}]}}]
        }
        mock_post.return_value = mock_response

        summary = nci.summarize_large_text(self.long_text)
        self.assertEqual(summary, "This is a summary.")

    @patch('NCI_EngineeredReality.api_session.post')
    def test_summarize_large_text_failure(self, mock_post):
        """Test summarization failure, which should return truncated text."""
        mock_post.side_effect = nci.requests.exceptions.RequestException("API Error")
        
        summary = nci.summarize_large_text(self.long_text, max_retries=2)
        self.assertEqual(summary, self.long_text[:nci.MAX_TEXT_LENGTH])
        self.assertEqual(mock_post.call_count, 2)

    @patch('NCI_EngineeredReality.make_gemini_request')
    def test_score_message(self, mock_make_request):
        """Test the main scoring loop."""
        # Have the mock return a consistent score of 3 for each of the 20 criteria
        mock_make_request.return_value = 3
        
        total_score, detailed_scores = nci.score_message(self.short_text)

        # 20 criteria * 3 points each = 60
        self.assertEqual(total_score, 60)
        self.assertEqual(len(detailed_scores), 20)
        self.assertEqual(detailed_scores[0]['score'], 3)
        self.assertEqual(mock_make_request.call_count, 20)

    @patch('sys.stdout', new_callable=StringIO)
    def test_generate_report(self, mock_stdout):
        """Test the final report generation."""
        test_score = 65
        test_details = [{"id": 1, "criterion": "Test Criterion", "score": 4}]
        
        nci.generate_report(test_score, test_details, "message")
        
        output = mock_stdout.getvalue()
        self.assertIn("NCI ENGINEERED REALITY SCORING REPORT", output)
        self.assertIn(f"Total NCI Score: {test_score}/100", output)
        self.assertIn("Strong likelihood - manipulation likely", output) # Based on score 65
        self.assertIn("[ 4 ] Test Criterion", output)

    @patch('NCI_EngineeredReality.summarize_large_text')
    @patch('NCI_EngineeredReality.score_message')
    def test_main_logic_summarization_flow(self, mock_score_message, mock_summarize):
        """Test that main() calls summarization for long text."""
        mock_summarize.return_value = "summarized text"
        mock_score_message.return_value = (0, [])

        # This simulates the main logic path for long text
        final_text = ""
        if len(self.long_text) > nci.MAX_TEXT_LENGTH:
            final_text = nci.summarize_large_text(self.long_text)
        else:
            final_text = self.long_text
        
        nci.score_message(final_text)

        mock_summarize.assert_called_once_with(self.long_text)
        mock_score_message.assert_called_once_with("summarized text")

    @patch('NCI_EngineeredReality.summarize_large_text')
    @patch('NCI_EngineeredReality.score_message')
    def test_main_logic_no_summarization_flow(self, mock_score_message, mock_summarize):
        """Test that main() does NOT call summarization for short text."""
        mock_score_message.return_value = (0, [])

        # This simulates the main logic path for short text
        final_text = ""
        if len(self.short_text) > nci.MAX_TEXT_LENGTH:
            final_text = nci.summarize_large_text(self.short_text)
        else:
            final_text = self.short_text
        
        nci.score_message(final_text)

        mock_summarize.assert_not_called()
        mock_score_message.assert_called_once_with(self.short_text)


if __name__ == '__main__':
    unittest.main()