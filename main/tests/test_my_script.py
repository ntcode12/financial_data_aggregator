import os
import unittest
from FinancialDataAggregator import FinancialDataAggregator
from FinancialDataAggregator import Scrapper
from FinancialDataAggregator import StockDataFetcher
from FinancialDataAggregator import SentimentAnalyzer
from FinancialDataAggregator import HeadlinesProcessor
from FinancialDataAggregator import DataPersister
import csv
import sqlite3
import tempfile
from unittest import mock
import httpx
import json

class TestFinancialDataAggregator(unittest.TestCase):

    def setUp(self):
        scrapper = Scrapper()
        stock_data_fetcher = StockDataFetcher()
        sentiment_analyzer = SentimentAnalyzer()
        headlines_processor = HeadlinesProcessor()
        data_persister = DataPersister()

        self.fda = FinancialDataAggregator(scrapper, stock_data_fetcher, sentiment_analyzer, headlines_processor, data_persister)
        df_news = self.df_news = self.fda.scrapper.scrape_financial_news_from_cnbc()
        print(df_news)

    def test_news_scraping(self):
        # Define the mock response content
        mock_content = """
        <html>
            <body>
                <a class="Card-title" href="https://www.cnbc.com/test-url-1/">Test Headline 1</a>
                <a class="Card-title" href="/test-url-2/">Test Headline 2</a>
            </body>
        </html>
        """

        # Mock the HTTP client
        with mock.patch('httpx.Client', autospec=True) as mock_client:
            # Set up the mock client to return the mock response
            mock_response = mock.MagicMock()
            mock_response.content = mock_content.encode()
            mock_client.return_value.get.return_value = mock_response

            # Perform the test
            df_news = self.fda.scrapper.scrape_financial_news_from_cnbc()
         
        # Perform the assertions
        self.assertIsNotNone(df_news)
        self.assertTrue(all(item in df_news.columns for item in ['headline', 'timestamp', 'url']))
        self.assertEqual(len(df_news), 2)

    def test_stock_price_retrieval(self):
        # Mock data
        mock_response = httpx.Response(200, content=json.dumps({
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2023-01-31",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": {
                "2023-01-31": {
                    "1. open": "152.0000",
                    "2. high": "152.4900",
                    "3. low": "149.6300",
                    "4. close": "149.9500",
                    "5. volume": "71297200"
                },
                "2023-01-30": {
                    "1. open": "154.5500",
                    "2. high": "154.6900",
                    "3. low": "151.4600",
                    "4. close": "152.3300",
                    "5. volume": "68847100"
                }
            }
        }), headers={'content-type': 'application/json'})
        
        # Use patch to mock the httpx.Client class
        with mock.patch('httpx.Client') as MockClient:
            instance = MockClient.return_value
            instance.get.return_value = mock_response

            df_stocks = self.fda.stock_data_fetcher.get_stock_prices_alphaVantage_api(
                ["AAPL"], 
                "2023-01-01", 
                "2023-01-31",
                os.getenv("ALPHA_VANTAGE_API_KEY")
            )

        # Add your assertions
        self.assertIsNotNone(df_stocks)
        self.assertTrue(all(item in df_stocks.columns for item in ['date', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'adjusted_close', 'volume', 'symbol']))
        self.assertGreater(len(df_stocks), 0)

    def test_categorize_polarity(self):
        # Define test cases with expected outputs
        test_cases = [
            (-0.5, "Negative"),
            (0, "Neutral"),
            (0.3, "Positive")
        ]

        # Test the function with each test case
        for input_value, expected_output in test_cases:
            result = self.fda.sentiment_analyzer.categorize_polarity(input_value)
            self.assertEqual(result, expected_output)

        # Separate test case for invalid input that should trigger the TypeError
        with self.assertRaises(TypeError):
            invalid_input = 'invalid_input'
            self.fda.sentiment_analyzer.categorize_polarity(invalid_input)

    def test_sentiment_analysis(self):
        df_news = self.fda.sentiment_analyzer.analyze_sentiment(self.df_news)
        self.assertIsNotNone(df_news)
        self.assertTrue(all(item in df_news.columns for item in ['sentiment_score', 'sentiment']))
        self.assertGreater(len(df_news), 0)

    def test_word_extraction(self):
        df_news = self.fda.headlines_processor.extract_relevant_words(self.df_news)
        self.assertIsNotNone(df_news)
        self.assertTrue('relevant_words' in df_news.columns)
        self.assertGreater(len(df_news), 0)

    def test_headline_translation(self):
        df_news = self.fda.headlines_processor.translate_headlines(self.df_news)
        self.assertIsNotNone(df_news)
        self.assertTrue(all(item in df_news.columns for item in ['headline_spanish', 'headline_it']))
        self.assertGreater(len(df_news), 0)

    def test_load_data(self):
        # Mock the csv.writer() and csv.reader() functions to return controlled data
        with mock.patch('csv.writer') as mock_writer, mock.patch('csv.reader') as mock_reader, mock.patch('sqlite3.connect') as mock_connect:

            # Define the return values for the csv.reader() calls
            mock_reader.side_effect = [
                iter([("2023-01-01", 100.0, 105.0, 95.0, 102.0, 102.0, 1000000, "AAPL")]),
                iter([("Test Headline", "2023-01-01 00:00:00", "http://testurl.com", 0.5, "Positive", "test, headline", "Prueba de titular", "Test di titolo")])
            ]

            # Mock the SQLite cursor object and its execute() method
            mock_cursor = mock.MagicMock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            # Call the function to be tested
            self.fda.data_persister.load_data("mock_db", "mock_stocks", "mock_headlines")

            # Check the execute() calls
            mock_cursor.execute.assert_any_call("INSERT INTO stock_prices VALUES (?,?,?,?,?,?,?,?)", ("2023-01-01", 100.0, 105.0, 95.0, 102.0, 102.0, 1000000, "AAPL"))
            mock_cursor.execute.assert_any_call("INSERT INTO headline_news VALUES (?,?,?,?,?,?,?,?)", ("Test Headline", "2023-01-01 00:00:00", "http://testurl.com", 0.5, "Positive", "test, headline", "Prueba de titular", "Test di titolo"))

    def test_load_data_exception_handling(self):
        #with mock.patch('csv.writer') as mock_writer, mock.patch('csv.reader') as mock_reader, mock.patch('sqlite3.connect') as mock_connect:
        with mock.patch('csv.writer') as mock_writer, mock.patch('csv.reader') as mock_reader, mock.patch('sqlite3.connect') as mock_connect, mock.patch('builtins.print') as mock_print:
            # Mock the SQLite cursor object and its execute() method to raise an exception
            mock_cursor = mock.MagicMock()
            mock_cursor.execute.side_effect = Exception("Mocked exception")
            mock_connect.return_value.cursor.return_value = mock_cursor
            self.fda.data_persister.load_data("mock_db", "mock_stocks", "mock_headlines")

            # Check if the correct error message was printed
            mock_print.assert_any_call("An error occurred when loading data into SQLite: Mocked exception")

            #with self.assertRaises(Exception) as context:
            #    self.fda.data_persister.load_data("mock_db", "mock_stocks", "mock_headlines")

            #self.assertTrue("An error occurred when loading data into SQLite: Mocked exception" in str(context.exception))

    def check_db_contents(self, db_name, expected_stocks, expected_headlines):
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        c.execute("SELECT * FROM stock_prices")
        rows = c.fetchall()
        self.assertEqual(len(rows), len(expected_stocks))
        for row, expected in zip(rows, expected_stocks):
            self.assertEqual(row, expected)

        c.execute("SELECT * FROM headline_news")
        rows = c.fetchall()
        self.assertEqual(len(rows), len(expected_headlines))
        for row, expected in zip(rows, expected_headlines):
            self.assertEqual(row, expected)

        conn.close()

if __name__ == '__main__':
    unittest.main()
