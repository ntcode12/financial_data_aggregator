# financial_data_aggregator

I developed Financial Data Aggregator, a comprehensive Python-based ETL (Extract, Transform, Load) solution designed to aggregate and analyze financial news and stock data. This project showcases my ability to conduct intricate ETL processes and data analysis while ensuring a modular and extensible design.

The Financial Data Aggregator is composed of several key components:

•	Scrapper Class: Utilizing the httpx library, this component extracts financial news from CNBC's website, with BeautifulSoup employed for parsing the HTML responses to isolate relevant data.

•	StockDataFetcher Class: This segment fetches stock prices and volume from a select list of companies from the Alpha Vantage API, harnessing the httpx library for HTTP requests and pandas for subsequent data manipulation.

•	SentimentAnalyzer Class: To determine the tonality of financial news headlines, I integrated sentiment analysis using the TextBlob library.

•	HeadlinesProcessor Class: This class employs the NLTK library to tokenize news headlines and extract relevant words, adding further depth to the data analysis.

•	Translator Class: Recognizing the global nature of finance, I incorporated a translation feature using the Googletrans library to translate news headlines into Spanish and Italian, thereby broadening the tool's reach across different languages.

•	DataPersister Class: The final component manages the critical task of data persistence. By using the sqlite3 library and google.cloud.sql.connector, this class loads the transformed data into various data storage systems, including SQLite databases and Google Cloud SQL databases.

In addition to these core components, the project includes various helper functions like 'get_current_time' for time formatting and 'get_logger' for efficient logging throughout the script.
In summary, the Financial Data Aggregator project represents my proficiency in developing complex ETL operations and data analysis using Python. The modular and extensible design I implemented effectively streamlines data processing and analysis, demonstrating my skills in creating efficient and robust data solutions. 

Technology Stack: Python, Google Cloud SQL, SQLlite Alpha Vantage API, BeatifulSoup, TextBlob

