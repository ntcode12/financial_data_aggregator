
# Financial Data Aggregator: A Comprehensive ETL Solution for Financial News and Stock Data

Financial Data Aggregator is a comprehensive Python-based ETL (Extract, Transform, Load) solution designed to aggregate and analyze financial news and stock data.

## Overview

This project showcases the ability to conduct intricate ETL processes and data analysis while ensuring a modular and extensible design. The application is composed of several key components:

- **Scrapper Class**: Utilizing the httpx library, this component extracts financial news from CNBC's website, with BeautifulSoup employed for parsing the HTML responses to isolate relevant data.
- **StockDataFetcher Class**: This component fetches stock prices and volume from a select list of companies from the Alpha Vantage API. It uses the httpx library for HTTP requests and pandas for subsequent data manipulation.
- **SentimentAnalyzer Class**: To determine the tonality of financial news headlines, sentiment analysis is performed using the TextBlob library.
- **HeadlinesProcessor Class**: This class employs the NLTK library to tokenize news headlines and extract relevant words, adding further depth to the data analysis.
- **Translator Class**: Recognizing the global nature of finance, a translation feature is incorporated using the Googletrans library to translate news headlines into Spanish and Italian, thereby broadening the tool's reach across different languages.
- **DataPersister Class**: This component manages the critical task of data persistence. By using the sqlite3 library and google.cloud.sql.connector, it loads the transformed data into various data storage systems, including SQLite databases and Google Cloud SQL databases.

In addition to these core components, the project includes various helper functions like 'get_current_time' for time formatting and 'get_logger' for efficient logging throughout the script.

## Conclusion

In summary, the Financial Data Aggregator project represents proficiency in developing complex ETL operations and data analysis using Python. The modular and extensible design effectively streamlines data processing and analysis, demonstrating the ability to create efficient and robust data solutions.


Technology Stack: Python, Google Cloud SQL, SQLlite Alpha Vantage API, BeatifulSoup, TextBlob

