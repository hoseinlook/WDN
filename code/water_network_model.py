class BaseModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self._model = self.load_model(file_path)

    def train(self):
        raise NotImplementedError()

    def load_model(self, file_path):
        raise NotImplementedError()

    def predict(self):
        pass
