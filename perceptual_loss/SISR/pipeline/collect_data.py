class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str
