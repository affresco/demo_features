class Spans(object):

    def __init__(self, elements):
        self.__elements = sorted(elements)

    @property
    def elements(self):
        return self.__elements

    def __eq__(self, other):
        for elem in self.__elements:
            if elem not in other.elements:
                return False
        return True

    def __iter__(self):
        for elem in self.__elements:
            yield elem
