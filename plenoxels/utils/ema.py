
class EMA():
    def __init__(self, weighting=0.9):
        self.weighting = weighting
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.weighting * val + (1 - self.weighting) * self.val

    @property
    def value(self):
        return self.val

    def __str__(self):
        return f"{self.val:.2e}"
