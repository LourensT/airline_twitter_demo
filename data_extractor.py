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
        self.cwd = os.getcwd()
        self.directory = directory
        self.items = os.listdir(directory)

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
                    print(f'--JSONDecodeError at line: [{n+1}]--')

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
                print(f'--KeyError at line: [{i}]--')
        return rows

    def make_csv(self):
        """
        Creates a csv file of all data
        excluding the ones with KeyErrors and JSONDecodeErrors
        """
        content = self.add_content()
        return pd.DataFrame(content, columns=self.features).to_csv('extracted_data.csv', index=False)


if __name__ == '__main__':
    # pass
    # For testing
    extractor = DataExtractor(directory='unzipped/', features=['id_str', 'text', 'created_at', ('user', 'id_str')])
    extractor.make_csv()
