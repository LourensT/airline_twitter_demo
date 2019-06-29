# Import standard dependencies
import os
import pickle
from PIL import Image

# Visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import modules for demo
from unzipper import Unzipper
from data_extractor import DataExtractor
from data_wrangler import DataWrangler

# Visualization for MAC OSX
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')


class Demo:
    """
    A class that calls on other modules in the
    repository and outputs the plot
    """
    def __init__(self):
        pass

    @staticmethod
    def save_plot(replytime_data):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(replytime_data["KLM"], normed=True, color='b', alpha=0.5, bins=500, label="KLM")
        plt.hist(replytime_data["BA"], normed=True, color='g', alpha=0.5, bins=500, label="British Airways")
        plt.xlim(0, 21600)
        plt.legend(prop={'size': 14})
        plt.xticks([0, 3600, 7200, 10800, 14400, 18000, 21600], ['0 hours', '1 hour', '2 hours', '3 hours',
                                                                 '4 hours', '5 hours', '6 hours'])
        plt.title('Histogram of Reply Time for KLM and British Airways', size=16)
        plt.ylabel('Density', size=14)
        plt.grid(True)
        plt.xlabel('Reply Time', size=14)
        plt.savefig('PLOT.png', dpi=300)

    @staticmethod
    def show_plot():
        print("showing PLOT.png..")
        Image.open("PLOT.png").show()

    def run(self):
        """
        demo for histogram of replytime
        """
        print('Initialized!')
        print("=====================================")
        print('Unzipping JSONs:')
        print("=====================================")
        zipper = Unzipper(os.path.abspath('zipped_data'))
        zipper.unzip_all()
        print("=====================================")
        print('Importing Raw Data from JSON\'s')
        print("=====================================")
        extractor = DataExtractor(directory='unzipped/', features=['id_str', 'created_at',
                                                                   ('user', 'id_str'), 'in_reply_to_status_id'])
        extractor.save_csv()

        print("=====================================")
        print('Extracting Reply Time data')
        print("=====================================")
        wrangler = DataWrangler()
        wrangler.replytime_wrangle()

        print("=====================================")
        print('Saving Visualization as "PLOT.png"')
        print("=====================================")
        # read in replytimedata
        with open('processed_data', 'rb') as fp:
            replytime_data = pickle.load(fp)
        self.save_plot(replytime_data)
        print('Ran Succesfully.')


if __name__ == "__main__":
    demo = Demo()
    demo.run()
    demo.show_plot()
