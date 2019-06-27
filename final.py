# Import standard dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Import modules for demo
from unzipper import Unzipper
from data_extractor import DataExtractor
from data_wrangler import DataWrangler

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Demo:
    """
    A class that calls on other modules in the
    repository and outputs the plot
    """
    def __init__(self):
        pass

    @staticmethod
    def sent_bar(self, data):
        plt.figure(figsize=(8, 5))
        data['sentiments'].value_counts().plot(kind='bar')
        plt.xticks(fontsize=15, rotation=90)
        plt.xlabel('Sentiment', fontsize=17)
        plt.yticks(fontsize=15)
        plt.ylabel('Frequency', fontsize=17)
        plt.title('Sentiment distribution in dataset', weight='bold', fontsize=20)
        plt.tight_layout()
        plt.savefig('sent_result.png', dpi=300)

    def replytime_hist(self, replytime_data):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(replytime_data["KLM"], normed=True, color ='b', alpha=0.5, bins=500, label="KLM")
        plt.hist(replytime_data["BA"], normed=True, color ='g', alpha=0.5, bins=500, label="British Airways")
        plt.xlim(0, 21600)
        plt.legend(prop={'size': 14})
        plt.xticks([0, 3600, 7200, 10800, 14400, 18000, 21600], ['0 hours', '1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours'])

        plt.title('Histogram of Reply Time for KLM and British Airways', size=16)
        plt.ylabel('Density', size=14)
        plt.grid(True)
        plt.xlabel('Reply Time', size=14)
        plt.savefig("Histogram_replytime_klm_ba.svg", dpi=300)
        plt.savefig('replytime_result.png', dpi=300)

    def demo_replytime(self):
        """
        demo for histogram of replytime
        """
        print('Unzipping')
        zipper = Unzipper(os.path.abspath('zipped_data'))
        zipper.unzip_all()
        print('Extracting Data')
        extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'lang',
                                                                   'created_at', ('user', 'id_str'), 'in_reply_to_status_id'])
        extractor.save_csv()
        print('Creating new features')
        wrangler = DataWrangler()
        series = {}
        wrangler.replytime_wrangle(airlineids=[56377143,18332190])
        print('Making plot')

        #read in replytimedata
        with open ('replytime_extracted', 'rb') as fp:
            replytime_data = pickle.load(fp)
        #print(type(replytime_data['KLM']))
        self.replytime_hist(replytime_data)

    def demo_sentiment(self):
        """
         demo for sentiment analysis and visualizations
        """
        print('Unzipping')
        zipper = Unzipper(os.path.abspath('zipped_data'))
        zipper.unzip_all()
        print('Extracting Data')
        extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'lang',
                                                                   'created_at', ('user', 'id_str'), 'in_reply_to_status_id'])
        extractor.save_csv()
        print('Creating new features')
        wrangler = DataWrangler()
        wrangler.sentiment_wrangle(airlineids=56377143)
        print('Making plot')
        df = pd.read_csv('cleaned_data.csv')
        self.sent_bar(df)

demo = Demo()
demo.demo_replytime()
