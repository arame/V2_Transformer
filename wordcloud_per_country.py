import pandas as pd
from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from config import Constants, Hyper
from helper import Helper
from charts import Chart

# See https://www.datacamp.com/community/tutorials/wordcloud-python
class WordcloudCountry():
    def __init__(self, df) -> None:
        # Create stopword list:
        self.stopwords = set(STOPWORDS)
        self.stopwords.update(["coronavirus", "covid", "time", "today", "know", "support", "update", "say", "take", "please", "need", "well", "think", "virus", "thank", "read", "new", "going", "read", "help", "people", "let", "will", "one", "said", "due", "see", "day", "via", "make", "call", "really", "every", "great", "still", "keep", "now", "im", "case", "patient", "everyone", "many", "corona", "says", "go", "even", "week", "dont", "outbreak", "first", "cant", "way", "good", "work", "spread", "live", "right", "come", "back", "news", "stop", "number", "want", "may", "home", "country", "hope", "got", "US", "pandemic", "crisis", "cases", "stay", "thing", "amid", "look"])
        self.df = df
    
    def calculate(self):
        Helper.printline(f"     Start wordcount")
        for country_code in Hyper.selected_country_codes:
            country = Hyper.dict_countries[country_code]
            Helper.printline(f"     Generate wordcloud for country {country}")
            df_for_country = self.df.query(f'country_code == "{country_code}"')
            text_arr = df_for_country["clean_text"].values
            words = []
            for line in text_arr:
                words.append(" ".join([word for word in line.split(" ") if word.isalpha()]))

            text = " ".join(words)
            Helper.printline(f"     There are {len(text)} words used in the selected tweets")
            wordcloud = WordCloud(stopwords=self.stopwords, background_color="white").generate(text)
            Chart.show_wordcloud(wordcloud, country)
            Helper.printline(f"     Wordcloud output for country {country}")
    
        Helper.printline(" End wordcloud")
