# Standard dependencies
import numpy as np
import pandas as pd
import datetime
import pickle

# Preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning
from keras.models import load_model


class DataWrangler:
    """
    Sentiment analysis
    """
    def __init__(self):
        self.df = pd.read_csv('extracted_data.csv')
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.vectorizer = TfidfVectorizer()
        self.nn_model = load_model('nn_sentiment_model.h5')

        self.klm_incoming = []
        self.ba_incoming = []
        self.klm_outgoing = []
        self.ba_outgoing = []

    @staticmethod
    def tokenize(sentence):
        """
        Splits up words and makes a list of all words in the tweet
        """
        tokenized_sentence = word_tokenize(sentence)
        return tokenized_sentence

    def remove_stopwords(self, sentence):
        """
        Removes stopwords like 'a', 'the', 'and', etc.
        """
        filtered_sentence = []
        for w in sentence:
            if w not in self.stopwords and len(w) > 1 and w[:2] != '//' and w != 'https':
                filtered_sentence.append(w)
        return filtered_sentence

    def stem(self, sentence):
        """
        Stems certain words to their root form.
        For example, words like 'computer', 'computation'
        all get trunacated to 'comput'
        """
        return [self.stemmer.stem(word) for word in sentence]

    @staticmethod
    def join_to_string(sentence):
        """
        Joins the tokenized words to one string.
        """
        return ' '.join(sentence)

    def vectorize(self, data):
        """
        Vectorizes a preprocessed sentence into a TF-IDF format
        Returns a sparse matrix
        """
        _ = self.vectorizer.fit_transform(np.load('vector.npy', allow_pickle=True))
        return self.vectorizer.transform(data)

    def preprocess(self):
        """
        Preprocess a sentence and
        connect back to string
        """
        # Perform preprocessing
        preprocessed = []
        for sentence in self.df['text']:
            tokenized = self.tokenize(sentence)
            cleaned = self.remove_stopwords(tokenized)
            stemmed = self.stem(cleaned)
            joined = self.join_to_string(stemmed)
            preprocessed.append(joined)
        self.df['cleaned_text'] = preprocessed

    def get_sentiments(self):
        self.df = self.df[self.df['lang'] == 'en']
        self.preprocess()
        vectorized_data = self.vectorize(self.df['cleaned_text'])
        self.df['sentiments'] = [np.argmax(self.nn_model.predict(data)) - 1 for data in vectorized_data]

    # def get_dates(self):
    #     self.df['created_at'] = pd.to_datetime(self.df['created_at'], format="%a %b %d %H:%M:%S +0000 %Y")
    #     self.df['hour'] = [str(date.hour) for date in self.df['created_at']]
    #     self.df['hour'] = ['0' + str(hour) if hour != 'nan' and int(hour) < 10 else str(hour) for hour in self.df['hour']]
    #     self.df['weekday'] = [date.weekday() for date in self.df['created_at']]
    #     print(self.df['weekday'])
    #     self.df['weekday'] = self.df['weekday'].astype("category", categories=['Mon', 'Tue',
    #                                                                            'Wed', 'Thu',
    #                                                                            'Fri', 'Sat',
    #                                                                            'Sun']).cat.codes
    #     self.df['weekday_hour'] = self.df['weekday'].astype(str) + self.df['hour'].astype(str)

    # def get_incoming_outgoing(self):
    #     # Get Outgoing info
    #     self.klm_outgoing = self.df[self.df["('user', 'id_str')"] == "56377143"]['weekday_hour']
    #     self.ba_outgoing = self.df[self.df["('user', 'id_str')"] == "18332190"]['weekday_hour']
    #
    #     # Get incoming info
    #     for i, item in self.df[['text', 'weekday_hour']].iterrows():
    #         if '@KLM' in item['text']:
    #             self.klm_incoming.append(item[['weekday_hour']])
    #         elif '@British_Airways' in item['text']:
    #             self.ba_incoming.append(item[['weekday_hour']])

    def sentiment_wrangle(self):
        print('Getting sentiments')
        self.get_sentiments()
        # print('Getting dates')
        # self.get_dates()
        # print('Getting incoming outcoming')
        # self.get_incoming_outgoing()
        # print('Creating cleaned dataframe')

        # cleaned_df = pd.DataFrame({'grouped_sentiments': self.df.groupby('weekday_hour').mean()['sentiments'],
        #                            'klm_outgoing': pd.Series(self.klm_outgoing).value_counts().sort_index(),
        #                            'ba_outgoing': pd.Series(self.ba_outgoing).value_counts().sort_index(),
        #                            'klm_incoming': pd.Series(self.klm_incoming).value_counts().sort_index(),
        #                            'ba_incoming': pd.Series(self.ba_incoming).value_counts().sort_index()})
        # print('Saving cleaned dataframe')
        self.df.to_csv('cleaned_data.csv', index=False)

    def timedelta(self, date1, date2):
            timedelta = date2 - date1
            return(timedelta.seconds)

    def replytime_wrangle(self, airlineids=False):
        if airlineids == False:
            print('Set airlineids to on of the following: 56377143, 106062176, 18332190, 22536055, 124476322, 26223583, 2182373406, 38676903, 1542862735, 253340062, 218730857, 45621423, 20626359]')
            return False
        full_df = self.df
        full_df['created_at'] = pd.to_datetime(full_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
        full_df = full_df.sort_values(by='created_at', ascending=False)

        response = {}
        replytime_series = []
        for data in full_df[["('user', 'id_str')", 'id_str', 'in_reply_to_status_id', 'created_at']].values:
            if str(data[0]) == str(airlineids):
                try:
                    response[int(data[2])] = data[3]
                except ValueError:
                    pass
            else:
                if int(data[1]) in response.keys():
                    if response[int(data[1])] > data[3]:
                        td = self.timedelta(data[3], response[int(data[1])])
                        replytime_series.append(td)

        # #print(response)
        # print(replytime_series)
        # print(len(replytime_series))

        with open('replytimefile', 'wb') as fp:
            pickle.dump(replytime_series, fp)



if __name__ == '__main__':
    # For testing
    wrangler = DataWrangler()
    wrangler.replytime_wrangle(airlineids=56377143)
