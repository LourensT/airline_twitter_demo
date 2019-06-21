# Standard dependencies
import numpy as np
import pandas as pd

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
    def __init__(self, data):
        self.df = pd.read_csv(data)
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

    def full_wrangle(self):
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


if __name__ == '__main__':
    # pass
    # For testing
    wrangler = DataWrangler('extracted_data.csv')
    wrangler.full_wrangle()
