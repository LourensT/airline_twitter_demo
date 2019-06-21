# Import Stuff
import os
import pandas as pd
import matplotlib.pyplot as plt
from demo.unzipper import Unzipper
from demo.data_extractor import DataExtractor
from demo.data_wrangler import DataWrangler


class Demo:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.zipper = Unzipper(os.path.abspath('demo/zipped_data'))
        self.extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'created_at', ('user', 'id_str')])
        self.wrangler = DataWrangler('extracted_data.csv')

    def sent_bar(self):
        plt.figure(figsize=(8, 5))
        self.df['sentiments'].value_counts().plot(kind='bar')
        plt.xticks(fontsize=16, rotation=90)
        plt.xlabel('Sentiment', fontsize=17)
        plt.yticks(fontsize=16)
        plt.ylabel('Frequency', fontsize=17)
        plt.title('Sentiment distribution in dataset', weight='bold', fontsize=20)
        plt.savefig('result.png', dpi=300)

    def demo(self):
        print('Unzipping')
        self.zipper.unzip_all()
        print('Extracting Data')
        self.extractor.make_csv()
        print('Creating new features')
        self.wrangler.full_wrangle()
        print('Making plot')
        self.sent_bar()


if __name__ == ' __main__':
    demo = Demo('cleaned_data.csv')
    demo.demo()
