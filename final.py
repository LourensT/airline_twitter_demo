# Import standard dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt

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
        self.df = pd.read_csv('cleaned_data.csv')
        #self.zipper = Unzipper(os.path.abspath('zipped_data'))
        #self.extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'created_at', ('user', 'id_str')])
        #self.wrangler = DataWrangler()

    def sent_bar(self, data):
        plt.figure(figsize=(8, 5))
        data['sentiments'].value_counts().plot(kind='bar')
        plt.xticks(fontsize=15, rotation=90)
        plt.xlabel('Sentiment', fontsize=17)
        plt.yticks(fontsize=15)
        plt.ylabel('Frequency', fontsize=17)
        plt.title('Sentiment distribution in dataset', weight='bold', fontsize=20)
        plt.tight_layout()
        plt.savefig('result.png', dpi=300)

    def demo(self):
        """
        The final demo that is run at the presentation
        """
        print('Unzipping')
        zipper = Unzipper(os.path.abspath('zipped_data'))
        zipper.unzip_all()
        print('Extracting Data')
        extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'lang',
                                                                   'created_at', ('user', 'id_str')])
        extractor.make_csv()
        print('Creating new features')
        wrangler = DataWrangler()
        wrangler.full_wrangle()
        print('Making plot')
        df = pd.read_csv('cleaned_data.csv')
        self.sent_bar(df)


if __name__ == ' __main__':
    # Run demo!
    demo = Demo()
    demo.demo()
