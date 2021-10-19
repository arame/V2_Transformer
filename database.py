import sqlite3 as lite
import sys
import pandas as pd
from helper import Helper
from config import Hyper

class Data:
        
    def __init__(self) -> None:
        self.db = Hyper.db
        self.create_connection()
        Helper.printline("Database opened successfully") 
        self.create_tweet_view_on_database()
         
    def create_connection(self):
        """ create a database connection to the SQLite database
            :param:
            :return:
        """
        self.con = None
        try:
            self.con = lite.connect(self.db, 
                            detect_types=lite.PARSE_DECLTYPES | lite.PARSE_COLNAMES)
        except Exception as e:
            sys.exit(f"Error with database connection: {e}")
            
    def create_tweet_view_on_database(self):
        sql_script = """ CREATE VIEW IF NOT EXISTS vw_tweets AS
                                SELECT  country_code,
                                        clean_text,
                                        sentiment,
                                        is_facemask,
                                        is_lockdown, 
                                        is_vaccine
                                FROM tweets WHERE length(country_code) = 2 AND (is_facemask + is_lockdown + is_vaccine > 0);
                        """
        try:
            c = self.con.cursor()
            c.execute(sql_script)
            c.close
        except Exception as e:
            sys.exit(f"Error with view creation: {e}")                 
      
    def get_tweets(self):
        content = Hyper.curr_content
        sql_script = f''' SELECT * FROM vw_tweets WHERE is_{content} = 1'''
        Helper.printline("Read Tweets")
        try:
            df_tweets = pd.read_sql_query(sql_script, self.con)
        except Exception as e:
            sys.exit(f"Error retreiving {content} tweets: {e}")

        Helper.printline(f"Tweets with {content} successfully retreived") 
        return df_tweets       
        
    def get_countries(self):
        sql_script = '''SELECT code, country FROM countries'''
        try:
            c = self.con.cursor()
            c.execute(sql_script)
            list = c.fetchall() 
            c.close()
            dict_countries = dict(list)
            return dict_countries
        except Exception as e:
            sys.exit(f"Error retreiving countries: {e}")
