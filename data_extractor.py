# Standard dependencies
import os
import json
import pandas as pd


class DataExtractor:
    """
    Converts multiple JSON files to a CSV with
    specified features
    """
    def __init__(self, directory, features):
        self.features = features
        self.generator = self.json_read()
        self.directory = directory
        self.items = os.listdir(directory)
        self.df = None

    def json_read(self):
        """
        Iterates over lines and
        creates a Python generator object
        out of JSON files
        """
        for i, file in enumerate(self.items):
            # Only handle .json files
            if not file.endswith('.json'):
                continue
            print(f'Loading file {i+1}/{len(self.items)}: {file}')

            # Load each line into a Python generator
            for n, line in enumerate(open(self.directory + file, mode='r')):
                try:
                    yield json.loads(line)
                except json.decoder.JSONDecodeError:
                    pass

    def add_content(self):
        """
        Creates a list of lists for constructing
        the DataFrame
        """
        rows = []
        for i, row in enumerate(self.generator):
            try:
                rows.append([row[x] if isinstance(x, str) else row[x[0]][x[1]] for x in self.features])
            except KeyError:
                pass
        self.df = pd.DataFrame(rows, columns=self.features)

    def save_csv(self):
        """
        Creates a csv file of all data
        excluding the ones with KeyErrors and JSONDecodeErrors
        """
        self.add_content()
        return self.df.to_csv('extracted_data.csv', index=False)
