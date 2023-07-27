# import libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from googletrans import Translator
import os
import sqlite3
import csv

def scrape_financial_news_from_cnbc():
    """
    Scrapes financial news from CNBC.

    Returns:
        DataFrame: A DataFrame with headlines, the scrape timestamp and the URL.
    """
    # Get the HTML of the CNBC homepage.
    response = requests.get("https://www.cnbc.com/markets/")
    soup = BeautifulSoup(response.content, "html.parser")

    # Get the headlines on the CNBC homepage.
    headlines = soup.find_all("a", class_="Card-title")

    # Create lists to store the headlines and their URLs.
    headlines_list = [headline.text.strip() for headline in headlines]
    url_list = [headline['href'] for headline in headlines]

    # Make sure to prepend the main domain to relative URLs.
    base_url = 'https://www.cnbc.com'
    url_list = [base_url + url if url.startswith('/') else url for url in url_list]

    # Save the timestamp of the moment you realized the request.
    timestamp = pd.Timestamp.now()

    # Create a DataFrame with the headlines, the timestamp and the URLs.
    df = pd.DataFrame({
        'headline': headlines_list,
        'timestamp': [timestamp] * len(headlines_list),
        'url': url_list
    })

    return df

if __name__ == "__main__":
    # Call the function to scrape the data.
    df_news = scrape_financial_news_from_cnbc()
    # Print the DataFrame.
    print(df_news)


def get_stock_prices_alphaVantage_api(symbols, start_date, end_date, api_key):
    """
    Gets stock prices from Alpha Vantage API for the specified dates and symbols.

    Args:
        symbols (list): A list of stock symbols to get data for.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        A DataFrame of stock prices.
    """

    # Create a DataFrame to store the stock prices.
    df = pd.DataFrame()

    # Iterate over the symbols.
    for symbol in symbols:

        # Construct the API endpoint URL.
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full"

        # Make a GET request to the API.
        response = requests.get(url)

        # Check the response status code.
        if response.status_code == 200:

            # Get the JSON response data.
            data = json.loads(response.content.decode("utf-8"))

            # Iterate over the dates.
            for date in data["Time Series (Daily)"]:
                # Check if the date is in the desired date range.
                if start_date <= date <= end_date:
                    # Get the stock price and other information.
                    price_data = data["Time Series (Daily)"][date]
                    open_price = float(price_data['1. open'])
                    highest_price = float(price_data['2. high'])
                    lowest_price = float(price_data['3. low'])
                    close_price = float(price_data['4. close'])
                    adjusted_close = float(price_data['5. adjusted close'])
                    volume = int(price_data['6. volume'])

                    # Create a temp DataFrame and append it to main DataFrame
                    temp_df = pd.DataFrame({
                        'date': [pd.to_datetime(date)],
                        'open_price': [open_price],
                        'highest_price': [highest_price],
                        'lowest_price': [lowest_price],
                        'close_price': [close_price],
                        'adjusted_close': [adjusted_close],
                        'volume': [volume],
                        'symbol': [symbol]})

                    df = pd.concat([df, temp_df], ignore_index=True)

        else:

            # Print an error message.
            print(f"Error getting stock price for {symbol}")

    # Sort the DataFrame by date.
    df = df.sort_values(by='date')

    # Return the DataFrame.
    return df


if __name__ == "__main__":
    # Set the stock symbols.
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # api key
    api_key = 'EO54HUMS797MTK6N'

    # Set the start and end dates.
    start_date = "2023-01-01"
    end_date = "2023-06-20"

    # Get the stock prices.
    df_stocks = get_stock_prices_alphaVantage_api(symbols, start_date, end_date, api_key)

    # Print the DataFrame.
    print(df_stocks)

def categorize_polarity(polarity):
    """
    Classifies the sentiment as positive, neutral or negative based on polarity.

    Args:
        polarity (float): The polarity score from TextBlob.

    Returns:
        str: The sentiment classification.
    """

    if polarity < -0.2:
        return "Negative"
    elif polarity <= 0.2:
        return "Neutral"
    else:
        return "Positive"

def analyze_sentiment(df_news):
    """
    Analyzes the sentiment of news headlines.

    Args:
        df_news (DataFrame): The news headlines.

    Returns:
        DataFrame: The DataFrame with the sentiment analysis.
    """

    # Copy the DataFrame.
    df = df_news.copy()

    # Get the sentiment polarity of the headlines.
    df['sentiment_score'] = df['headline'].apply(lambda headline: TextBlob(headline).sentiment.polarity)

    # Categorize the sentiment.
    df['sentiment'] = df['sentiment_score'].apply(categorize_polarity)

    return df

if __name__ == "__main__":
    # Analyze the sentiment of the news headlines.
    df_news = analyze_sentiment(df_news)

    # Print the DataFrame with sentiment analysis.
    print(df_news)


def extract_relevant_words(df):
    """
    Extracts relevant words from the headlines.

    Args:
        df (DataFrame): DataFrame containing the headlines.

    Returns:
        DataFrame: The original DataFrame with a new column containing the tokenized headlines.
    """
    # Tokenize the headlines and store the result in a new column.
    df['relevant_words'] = df['headline'].apply(word_tokenize)

    return df


if __name__ == "__main__":
    # Extract the relevant words.
    df_news = extract_relevant_words(df_news)
    # Print the DataFrame.
    print(df_news)

def translate_headlines(df):
  """
  Translates the headlines in a DataFrame to Spanish and Italian.

  Args:
    df (DataFrame): DataFrame containing the headlines.

  Returns:
    A DataFrame with the translated headlines in two new columns: `headline_spanish` and `headline_it`.
  """
  # Create a translator object.
  translator = Translator()

  # Translate the headlines to Spanish and store the result in a new column.
  df['headline_spanish'] = df['headline'].apply(lambda x: translator.translate(x, dest='es').text)

  # Translate the headlines to Italian and store the result in a new column.
  df['headline_it'] = df['headline'].apply(lambda x: translator.translate(x, dest='it').text)

  # Return the DataFrame.
  return df
if __name__ == "__main__":
    # Extract the relevant words.
    df_news = translate_headlines(df_news)
    # Print the DataFrame.
    print(df_news)

def save_headlines_to_csv(df, path='data/headlines/headlines_data.csv'):
    """
    Saves headlines data into a CSV file.

    Args:
        df (DataFrame): DataFrame with the headlines data.
        path (str): The path to save the CSV file to. By default, it's 'data/headlines/headlines_data.csv'.
    """
    # Check if the directory exists, if not create it.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the DataFrame to a CSV file.
    df.to_csv(path, index=False)

if __name__ == "__main__":
    # Save the data to a CSV file.
    save_headlines_to_csv(df_news)

def save_stock_info_to_csv(df, path='data/stocks/stocks_data.csv'):
    """
    Saves stock info data into a CSV file.

    Args:
        df (DataFrame): DataFrame with the stocks data.
        path (str): The path to save the CSV file to. By default, it's 'data/stocks/stocks_data.csv'.
    """
    # Check if the directory exists, if not create it.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the DataFrame to a CSV file.
    df.to_csv(path, index=False)

if __name__ == "__main__":
    # Save the data to a CSV file.
    save_stock_info_to_csv(df_stocks)

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('etl_extended_case.db')

# Create a cursor object
c = conn.cursor()

# Create the 'stock_prices' table
c.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        date TEXT,
        open_price FLOAT,
        highest_price FLOAT,
        lowest_price FLOAT,
        close_price FLOAT,
        adjusted_close FLOAT,
        volume INTEGER,
        symbol TEXT
    )
""")

# Create the 'headline_news' table
c.execute("""
    CREATE TABLE IF NOT EXISTS headline_news (
        headline TEXT,
        timestamp TEXT,
        url TEXT,
        sentiment_score FLOAT,
        sentiment TEXT,
        relevant_words TEXT,
        headline_spanish TEXT,
        headline_it TEXT
    )
""")

# Open the files and insert data into tables
with open('data/stocks/stocks_data.csv', 'r') as f:
    stocks_reader = csv.reader(f)
    for row in stocks_reader:
        c.execute(""" INSERT INTO stock_prices VALUES (?,?,?,?,?,?,?,?) """, row)

with open('data/headlines/headlines_data.csv', 'r') as f:
    headlines_reader = csv.reader(f)
    for headline in headlines_reader:
        c.execute(""" INSERT INTO headline_news VALUES (?,?,?,?,?,?,?,?) """, headline)

# Commit the transactions
conn.commit()

# Close the connection
conn.close()

# Verify that the data has been loaded correctly
conn = sqlite3.connect('etl_extended_case.db')
c = conn.cursor()

# Select first 5 rows from stock_prices table
c.execute("SELECT * FROM stock_prices LIMIT 5")
stock_prices_data = c.fetchall()

# Select first 5 rows from headline_news table
c.execute("SELECT * FROM headline_news LIMIT 5")
headline_news_data = c.fetchall()

# Close the connection
conn.close()

# Print the data
print("First 5 rows of stock_prices:")
for row in stock_prices_data:
    print(row)

print("\nFirst 5 rows of headline_news:")
for row in headline_news_data:
    print(row)