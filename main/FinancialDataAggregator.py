## Nichollas Tidow
## June 2023
## ETL Financial Data Aggregator

import time
import os
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
from googletrans import Translator
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import sqlite3
import csv
import ssl
import httpx
import tracemalloc
import logging
import pymysql
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, String, Float, Integer, create_engine
from sqlalchemy.sql import text
from google.cloud.sql.connector import Connector, IPTypes
import pg8000
import sqlalchemy
import psycopg2
tracemalloc.start()


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler for logging
handler = logging.FileHandler('main/logs/FinancialDataAggregator.log')
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)



def timing(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info(f"Elapsed time for {f.__name__}: {end - start} s")
        return result
    return wrapper

class Scrapper:
    def __init__(self):
        pass

    @timing
    def scrape_financial_news_from_cnbc(self):
        try:
            """
                Scrapes financial news from CNBC.

            Returns:
            DataFrame: A DataFrame with headlines, the scrape timestamp and the URL.
            """
            
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

            # Get the HTML of the CNBC homepage.
            with Client(http2=True, verify=False) as client:
                response = client.get("https://www.cnbc.com/markets/")
                soup = BeautifulSoup(response.content, "html.parser")
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
        except Exception as e:
            logger.error(f"An unexpected error occurred when scraping financial news from CNBC: {e}")
            return None



class StockDataFetcher:
    def __init__(self):
        pass

    @timing
    def get_stock_prices_alphaVantage_api(self, symbols, start_date, end_date, api_key):
        try:
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
                with httpx.Client() as client:
                    response = client.get(url)

                    # Check the response status code.
                    if response.status_code != 200:
                        logger.error(f"Error {response.status_code}: Failed to get stock price for {symbol}")
                        continue  # Skip to the next symbol.

                # Get the JSON response data.
                data = response.json()
                

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

            # Sort the DataFrame by date.
            df = df.sort_values(by='date')

            # Return the DataFrame.
            return df

        except HTTPStatusError as e:
            logger.error(f"A HTTP status error occurred: {e}")
        except RequestError as e:
            logger.error(f"An error occurred when getting stock prices from AlphaVantage: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred when getting stock prices from AlphaVantage: {e}")
        return None
    


class SentimentAnalyzer:
    def __init__(self):
        pass


    @timing
    def categorize_polarity(self, polarity):
        try:
            """
            Categorizes the sentiment polarity.

            Args:
                polarity (float): The polarity score.

            Returns:
                str: The sentiment category.
            """

            polarity = float(polarity)

            # Categorize the sentiment polarity.
            if polarity < -0.2:
                return "Negative"
            elif polarity <= 0.2:
                return "Neutral"
            else:
                return "Positive"
        except ValueError:
            raise TypeError("polarity must be a float or an int")
        except Exception as e:
            logger.error(f"An error occurred when categorizing polarity: {e}")
            return None
        
    @timing
    def analyze_sentiment(self, df_news):
        try:
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
            df['sentiment'] = df['sentiment_score'].apply(self.categorize_polarity)

            return df
        except Exception as e:
            logger.error(f"An error occurred when analyzing sentiment: {e}")
            return None

class HeadlinesProcessor:
    def __init__(self):
        pass

    @timing
    def extract_relevant_words(self, df):
        try:
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
        except Exception as e:
            logger.error(f"An error occurred when extracting relevant words: {e}")
            return None

    @timing
    def translate_headlines(self, df):
        try:
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
        except Exception as e:
            logger.error(f"An error occurred when translating headlines: {e}")
            return None


class DataPersister:
    def __init__(self):
        pass
    @timing
    def save_data(self, df, path):
        try:
            """
            Saves data from a DataFrame into a CSV file.

            Args:
                df (DataFrame): DataFrame with the data.
                path (str): The path to save the CSV file to.
            """
            # Check if the directory exists, if not create it.
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the DataFrame to a CSV file.
            df.to_csv(path, index=False)
        except Exception as e:
            logger.error(f"An error occurred when saving data to a CSV file: {e}")



    # Loads data into SQLlite
    @timing
    def load_data(self, db_name, stocks_path, headlines_path):
        try:
            """
            Save data from csv files into a SQLite database.

            Args:
                db_name (str): The name of the SQLite database.
                stocks_path (str): The path of the stocks csv file.
                headlines_path (str): The path of the headlines csv file.
            """
            # Connect to SQLite database (it will be created if it doesn't exist)
            conn = sqlite3.connect(db_name)
 
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

            # Define the composite key columns for each table.
            stock_key_columns = ['date', 'symbol']  
            headlines_key_columns = ['headline', 'timestamp']  

            # Open the files and insert data into tables
            with open(stocks_path, 'r') as f:
                stocks_reader = csv.reader(f)
                for row in stocks_reader:
                    c.execute(""" INSERT INTO stock_prices VALUES (?,?,?,?,?,?,?,?) """, row)

            with open(headlines_path, 'r') as f:
                headlines_reader = csv.reader(f)
                for headline in headlines_reader:
                    c.execute(""" INSERT INTO headline_news VALUES (?,?,?,?,?,?,?,?) """, headline)

            # Commit the transactions
            conn.commit()

            # Close the connection
            conn.close()
        except Exception as e:
            logger.error(f"An error occurred when loading data into SQLite: {e}")
    @timing
    # def connect_with_connector():
    #     try:
    #         """Initializes a connection pool for a Google Cloud SQL instance of MySQL """
    #         instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")
    #         db_user = os.environ.get("DB_USER")
    #         db_pass = os.environ.get("DB_PASS")
    #         db_name = os.environ.get("DB_NAME")

    #         ip_type = IPTypes.PRIVATE if os.getenv("PRIVATE_IP") else IPTypes.PUBLIC

    #         connector = Connector(ip_type)

    #         engine = create_engine(f"mysql+mysqldb://{db_user}:{db_pass}@/{db_name}?unix_socket=/cloudsql/{instance_connection_name}")
    #         return engine

    #     except Exception as e:
    #         print(f"Error connecting to database: {e}")




    #     """
    #         Save data from csv files into a google cloud SQL database.

    #         Args:
    #             engine: SQLAlchemy engine instance
    #             stocks_path (str): The path of the stocks csv file.
    #             headlines_path (str): The path of the headlines csv file.
    #             table1 (str): The name of the first table.
    #             table2 (str): The name of the second table.
    #     """
    #     try:

    #         # Create metadata instance
    #         metadata = MetaData()

    #         # Define tables
    #         stocks_data = Table( table1, metadata,
    #         Column('date',String(10), primary_key=True),
    #         Column('symbol', String(10), primary_key=True),
    #         Column('open_price', Float),
    #         Column('highest_price', Float),
    #         Column('lowest_price', Float),
    #         Column('close_price', Float),
    #         Column('adjusted_close', Float),
    #         Column('volume', Integer)
    #         )

    #         headline_news = Table(
    #         table2, metadata,
    #         Column('headline', String, primary_key=True),
    #         Column('timestamp', String(10), primary_key=True),
    #         Column('url', String),
    #         Column('sentiment_score', Float),
    #         Column('sentiment', String),
    #         Column('relevant_words', String),
    #         Column('headline_spanish', String),
    #         Column('headline_it', String)
    #         )


            
    #         # Create tables
    #         metadata.create_all(engine)

    #         # Load the stocks data into a DataFrame
    #         with open(stocks_path, 'r') as f:
    #             stocks_df = pd.read_csv(f)
    #         # Write stocks data into stocks table in database
    #         stocks_df.to_sql(table1, engine, if_exists='append', index=False)

    #         # Load the headlines data into a DataFrame
    #         with open(headlines_path, 'r') as f:
    #             headlines_df = pd.read_csv(f)

    #         # Write headlines data into headlines table in database
    #         headlines_df.to_sql(table2, engine, if_exists='append', index=False)

    #     except Exception as e:
    #         logger.error(f"An error occurred when loading data into Google Cloud SQL: {e}")
    


    @timing
    def insert_if_not_exists(self, cursor, table, key_columns, row):
        try:
            """
            Inserts a row into a table, but only if the row does not already exist.

            Args:
                cursor: SQLite cursor.
                table (str): Name of the table.
                key_columns (list): List of column names that make up the unique key.
                row (list): List of values to insert.
            """
            # Prepare SQL to check for existing data.
            check_sql = f'SELECT * FROM {table} WHERE ' + ' AND '.join([f'{col} = ?' for col in key_columns])

            # Extract key values from the row.
            key_values = [row[i] for i in range(len(key_columns))]

            # Execute the SQL.
            cursor.execute(check_sql, key_values)
            if cursor.fetchone() is None:
                # The data does not exist in the database, so we insert it.
                placeholders = ', '.join(['?'] * len(row))
                insert_sql = f'INSERT INTO {table} VALUES ({placeholders})'
                cursor.execute(insert_sql, row)
        except Exception as e:
            logger.error(f"An error occurred when inserting data into SQLite: {e}")
        except sqlite3.IntegrityError:
            # The data already exists in the database, so we do nothing.
            pass




class CloudSQL:
    def __init__(self):
        load_dotenv()
        self.instance_connection_name = os.environ[
            "INSTANCE_CONNECTION_NAME"
        ]  # e.g. 'project:region:instance'
        self.db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
        self.db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
        self.db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
        self.ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
        self.connector = Connector()

    def getconn(self) -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = self.connector.connect(
            self.instance_connection_name,
            "pg8000",
            user=self.db_user,
            password=self.db_pass,
            db=self.db_name,
            ip_type=self.ip_type,
        )
        return conn

    def connect_with_connector(self) -> sqlalchemy.engine.base.Engine:
        """
        Initializes a connection pool for a Cloud SQL instance of Postgres.

        Uses the Cloud SQL Python Connector package.
        """
        try:
            # The Cloud SQL Python Connector can be used with SQLAlchemy
            # using the 'creator' argument to 'create_engine'
            pool = sqlalchemy.create_engine(
                "postgresql+pg8000://",
                creator=self.getconn,
            )
            return pool
        except Exception as e:
            logging.error(f"Error connecting to Cloud SQL: {e}")
            raise

    def create_table(self):
        try:
            # Connect to the instance using psycopg2
            conn = psycopg2.connect(self.instance_connection_name)

            # Create a cursor object
            cursor = conn.cursor()

            # Define the SQL query to create the stocks_data table
            create_stocks_data_query = """
            CREATE TABLE IF NOT EXISTS stocks_data (
                date DATE,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                adjusted_close FLOAT,
                volume BIGINT,
                ticker TEXT
            );
            """


            # Define the SQL query to create the headline_news table
            create_headline_news_query = """
            CREATE TABLE IF NOT EXISTS headline_news (
                headline TEXT PRIMARY KEY,
                timestamp TEXT(10) PRIMARY KEY,
                url TEXT,
                sentiment_score FLOAT,
                sentiment TEXT,
                relevant_words TEXT,
                headline_spanish TEXT,
                headline_it TEXT
            );
            """

            # Execute the SQL queries to create the tables
            cursor.execute(create_stocks_data_query)
            cursor.execute(create_headline_news_query)

            # Commit the transaction
            conn.commit()

            # Close the cursor and connection
            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error creating table: {e}")
            raise
            raise

    def load_csv_to_cloud_sql(self, stocks_csv_file_path, news_csv_file_path):
        try:
            # Connect to the instance using psycopg2
            conn = psycopg2.connect(self.instance_connection_name)

            # Create a cursor object
            cursor = conn.cursor()

            # Define the SQL query to create the stocks_data table
            create_stocks_data_query = """
            CREATE TABLE IF NOT EXISTS stocks_data (
                date DATE,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                adjusted_close FLOAT,
                volume BIGINT,
                ticker TEXT
            );
            """

            # Define the SQL query to create the headline_news table
            create_headline_news_query = """
            CREATE TABLE IF NOT EXISTS headline_news (
                headline TEXT PRIMARY KEY,
                timestamp TEXT(10) PRIMARY KEY,
                url TEXT,
                sentiment_score FLOAT,
                sentiment TEXT,
                relevant_words TEXT,
                headline_spanish TEXT,
                headline_it TEXT
            );
            """

            # Execute the SQL queries to create the tables
            cursor.execute(create_stocks_data_query)
            cursor.execute(create_headline_news_query)

            # Commit the transaction
            conn.commit()

            # Load data from the stocks CSV file into a Pandas DataFrame
            stocks_df = pd.read_csv(stocks_csv_file_path)

            # Insert the stocks data into the Cloud SQL database
            cursor = conn.cursor()
            for index, row in stocks_df.iterrows():
                values = tuple(row)
                placeholders = ",".join(["%s"] * len(values))
                query = f"INSERT INTO stocks_data VALUES ({placeholders})"
                cursor.execute(query, values)
            conn.commit()

            # Load data from the headline news CSV file into a Pandas DataFrame
            news_df = pd.read_csv(news_csv_file_path)

            # Insert the headline news data into the Cloud SQL database
            cursor = conn.cursor()
            for index, row in news_df.iterrows():
                values = tuple(row)
                placeholders = ",".join(["%s"] * len(values))
                query = f"INSERT INTO headline_news VALUES ({placeholders})"
                cursor.execute(query, values)
            conn.commit()

            # Close the connection
            cursor.close()
            conn.close()

            logging.info("Data imported successfully!")
        except Exception as e:
            logging.error(f"Error loading CSV to Cloud SQL: {e}")
            raise
           

    def fetch_and_print_records(self):
        try:
            # Connect to the instance using psycopg2
            conn = psycopg2.connect(self.instance_connection_name)

            # Create a cursor object
            cursor = conn.cursor()

            # Define a SQL query to fetch the first 10 records from the stocks_data table
            fetch_stocks_query = "SELECT * FROM stocks_data ORDER BY date DESC LIMIT 10;"

            # Execute the query
            cursor.execute(fetch_stocks_query)

            # Fetch the records
            stocks_records = cursor.fetchall()

            # Convert records to DataFrame for better visualization
            stocks_df = pd.DataFrame(stocks_records, columns=['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'ticker'])

            # Print the records
            print("Stocks Data:")
            print(stocks_df)

            # Define a SQL query to fetch the first 10 records from the headline_news table
            fetch_news_query = "SELECT * FROM headline_news ORDER BY timestamp DESC LIMIT 10;"

            # Execute the query
            cursor.execute(fetch_news_query)

            # Fetch the records
            news_records = cursor.fetchall()

            # Convert records to DataFrame for better visualization
            news_df = pd.DataFrame(news_records, columns=['headline', 'timestamp', 'url', 'sentiment_score', 'sentiment', 'relevant_words', 'headline_spanish', 'headline_it'])

            # Print the records
            print("Headline News Data:")
            print(news_df)

            # Close the connection
            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error fetching and printing records: {e}")
            raise
import logging
import os

from google.cloud import storage
import pandas as pd
import psycopg2


class CloudDataLoader:
    def __init__(self, instance_connection_name, bucket_name):
        self.instance_connection_name = instance_connection_name
        self.bucket_name = bucket_name

    def load_data_to_gcs(self):
        try:
            # Connect to the instance using psycopg2
            conn = psycopg2.connect(self.instance_connection_name)

            # Create a cursor object
            cursor = conn.cursor()

            # Define a SQL query to fetch all records from the stocks_data table
            fetch_stocks_query = "SELECT * FROM stocks_data;"

            # Execute the query
            cursor.execute(fetch_stocks_query)

            # Fetch the records
            stocks_records = cursor.fetchall()

            # Convert records to DataFrame
            stocks_df = pd.DataFrame(stocks_records, columns=['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'ticker'])

            # Define a SQL query to fetch all records from the headline_news table
            fetch_news_query = "SELECT * FROM headline_news;"

            # Execute the query
            cursor.execute(fetch_news_query)

            # Fetch the records
            news_records = cursor.fetchall()

            # Convert records to DataFrame
            news_df = pd.DataFrame(news_records, columns=['timestamp', 'source', 'title', 'description', 'url'])

            # Close the connection
            cursor.close()
            conn.close()

            # Write the data to CSV files
            stocks_df.to_csv('stocks_data.csv', index=False)
            news_df.to_csv('headline_news.csv', index=False)

            # Upload the CSV files to the GCS bucket
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)
            stocks_blob = bucket.blob('stocks_data.csv')
            stocks_blob.upload_from_filename('stocks_data.csv')
            news_blob = bucket.blob('headline_news.csv')
            news_blob.upload_from_filename('headline_news.csv')

            logging.info("Data loaded to GCS successfully!")
        except Exception as e:
            logging.error(f"Error loading data to GCS: {e}")
            raise



# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

class FinancialDataAggregator:
    def __init__(self, scrapper, stock_data_fetcher, sentiment_analyzer, headlines_processor, data_persister, cloud):
        self.scrapper = scrapper
        self.stock_data_fetcher = stock_data_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.headlines_processor = headlines_processor
        self.data_persister = data_persister
        self.gcloud_sql = gcloudsql
        self.cloud

    def run(self):
        df_news = self.scrapper.scrape_financial_news_from_cnbc()
        df_stocks = self.stock_data_fetcher.get_stock_prices_alphaVantage_api(
            ["AAPL", "MSFT", "GOOG", "AMZN", "META"], 
            "2023-01-01", 
            "2023-06-20",
            os.getenv("ALPHA_VANTAGE_API_KEY")
        )
        df_news = self.sentiment_analyzer.analyze_sentiment(df_news)
        df_news = self.headlines_processor.extract_relevant_words(df_news)
        df_news = self.headlines_processor.translate_headlines(df_news)
        self.data_persister.save_data(df_news, 'data/headlines/headlines_data.csv')
        self.data_persister.save_data(df_stocks, 'data/stocks/stocks_data.csv')
        
        # Save data into SQLite database
        self.data_persister.load_data('etl_extended_case.db', 'data/stocks/stocks_data.csv', 'data/headlines/headlines_data.csv')
        # Save data into Google Cloud SQL database
        engine = self.data_persister.connect_with_connector()
        self.data_persister.load_data_to_gcloud_sql(engine, 'data/stocks/stocks_data.csv', 'data/headlines/headlines_data.csv', 'stock_prices', 'headline_news')

        # Get memory usage statistics
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        # Write memory usage statistics to log file
        for stat in top_stats[:10]:
            logging.debug(stat)

        # Connect to Google Cloud SQL database
        self.gcloud_sql.connect()

        # Create tables in Google Cloud SQL database
        self.gcloud_sql.create_tables()

        # Load data into Google Cloud SQL database
        self.gcloud_sql.load_data('data/stocks/stocks_data.csv', 'data/headlines/headlines_data.csv')

        # Disconnect from Google Cloud SQL database
        self.gcloud_sql.disconnect()

        loader = CloudDataLoader(instance_connection_name='your-connection-name', bucket_name='your-bucket-name')
        loader.load_data_to_gcs()

if __name__ == "__main__":
    try:
        scrapper = Scrapper()
        stock_data_fetcher = StockDataFetcher()
        sentiment_analyzer = SentimentAnalyzer()
        headlines_processor = HeadlinesProcessor()
        data_persister = DataPersister()

        fda = FinancialDataAggregator(scrapper, stock_data_fetcher, sentiment_analyzer, headlines_processor, data_persister)
        fda.run()
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
    logging.info("ETL finished successfully")