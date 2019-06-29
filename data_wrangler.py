# Standard dependencies
import pickle
import pandas as pd


class DataWrangler:
    """
    Creation of new features and
    wrangling of information from extracted_data
    """
    def __init__(self):
        print('Loading extracted data..')
        self.df = pd.read_csv('extracted_data.csv', engine='python')
        # self.df = pd.read_csv(open('extracted_data.csv','rU'), encoding='utf-8', engine='c')

    @staticmethod
    def timedelta(date1, date2):
        """
        Calculates the difference between dates in seconds
        """
        timedelta = date2 - date1
        return timedelta.seconds

    def replytime_wrangle(self):
        print('Transforming datestrings to datetime objects..')
        full_df = self.df
        full_df['created_at'] = pd.to_datetime(full_df['created_at'],
                                               format='%a %b %d %H:%M:%S +0000 %Y',
                                               errors='coerce')
        full_df = full_df.dropna(subset=['created_at'])
        full_df = full_df.sort_values(by='created_at', ascending=False)

        airlineids = ["56377143", "18332190"]

        klm_dict = {}
        ba_dict = {}

        klm_series = []
        ba_series = []

        print('Extracting replytime series..')
        for data in full_df[["('user', 'id_str')", 'id_str', 'in_reply_to_status_id', 'created_at']].values:
            if str(data[0]) == airlineids[0]:
                try:
                    klm_dict[int(data[2])] = data[3]
                except ValueError:
                    pass
            elif str(data[0]) == airlineids[1]:
                try:
                    ba_dict[int(data[2])] = data[3]
                except ValueError:
                    pass
            else:
                if int(data[1]) in klm_dict.keys():
                    if klm_dict[int(data[1])] > data[3]:
                        td = self.timedelta(data[3], klm_dict[int(data[1])])
                        klm_series.append(td)
                if int(data[1]) in ba_dict.keys():
                    if ba_dict[int(data[1])] > data[3]:
                        td = self.timedelta(data[3], ba_dict[int(data[1])])
                        ba_series.append(td)

        pickle_dictionary = {"KLM": klm_series, "BA": ba_series}
        with open('processed_data', 'wb') as fp:
            pickle.dump(pickle_dictionary, fp)

        print('Pickled replytime series succesfully..')
