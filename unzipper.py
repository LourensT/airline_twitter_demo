import os
from zipfile import ZipFile


class Unzipper:
    def __init__(self, directory):
        # The directory from where to retrieve the files
        self.dir = directory

    @staticmethod
    def unzip(file):
        """
        Unzip file and save unzipped file in a directory called 'unzipped'
        """
        if not file.endswith('.DS_Store'):
            print(f'Unzipping: {file}')
            with ZipFile(file, 'r') as f:
                f.extractall('unzipped')

    def unzip_all(self):
        """
        Unzips all files in the given directory
        """
        for file in os.listdir(self.dir):
            self.unzip(f'{self.dir}/{file}')


if __name__ == '__main__':
    # pass
    # For testing
    zipper = Unzipper(os.path.abspath('zipped_data'))
    zipper.unzip_all()
