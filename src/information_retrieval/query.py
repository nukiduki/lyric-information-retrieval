"""Class for a users query"""


class Query():
    def __init__(self, text):
        self.plain: str = text
        self.tokens = text.split()

        # following attributes might be useful in the future idk
        self.terms = []
        self.ands = []
        self.ors = []
        self.nots = []
        self.parenthesis = []

    def __str__(self):
        return self.plain

    def __getitem__(self, index):
        return self.plain  # not sure about that
    
    def __iter__(self):
        return iter(self.plain)
    
    def export(self):
        return "_".join(self.tokens)