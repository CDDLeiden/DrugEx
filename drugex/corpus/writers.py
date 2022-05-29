"""
writers

Created by: Martin Sicho
On: 28.04.22, 15:01
"""
from drugex.corpus.interfaces import CorpusWriter

class ListWriter(CorpusWriter):

    def __init__(self):
        self.result = []

    def write(self, data):
        self.result.append(data)

    def close(self):
        pass

    def get(self):
        return self.result

class FileWriter(CorpusWriter):

    def __init__(self, path):
        self.path = path
        self.file = open(path, mode="w", encoding="utf-8")

    def write(self, data):
        self.file.write(data['seq'] + "\t" + data['token'] + "\n")

    def close(self):
        self.file.close()

class SequenceFileWriter(FileWriter):

    def __init__(self, path, seq_col="Sequence", token_col="Token"):
        super().__init__(path)
        self.file.write(f"{seq_col}\t{token_col}\n")

