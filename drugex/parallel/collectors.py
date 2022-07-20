"""
collectors

Created by: Martin Sicho
On: 14.07.22, 10:58
"""
from drugex.parallel.interfaces import ListCollector


class ListExtend(ListCollector):

    def __init__(self):
        self.items = []

    def __call__(self, result):
        self.items.extend(result)

    def getList(self):
        return self.items

class ListAppend(ListExtend):

    def __call__(self, result):
        self.items.append(result)
