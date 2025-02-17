import pandas as pd


def open_file(path, columns=[], dtype=None):

    file = CSVFile(path, columns=columns, dtype=dtype)

    return file


def load_file(path, expected_columns=None):

    dataframe = pd.read_csv(path, index_col=0)

    if expected_columns is not None:
        columns = dataframe.columns.values.tolist()
        for expected_column in expected_columns:
            assert expected_column in columns, "column '{}' missing in file://{}".format(expected_column, path)

    return dataframe


class CSVFile:

    def __init__(self, path, columns=[], dtype=None):

        self._path = path
        self._columns = columns
        self._dtype = dtype

        self._list = []

    def append(self, **kwargs):

        self._list.append(kwargs)

        return

    def close(self):

        df = pd.DataFrame(self._list, columns=self._columns, dtype=self._dtype)
        df.to_csv(self._path)

        return
