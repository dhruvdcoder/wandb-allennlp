import torch
import allennlp
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary


@Model.register('dummy')
class DummyModel(Model):
    def __init__(self, vocab: Vocabulary, a: float, b: float = 0):
        super().__init__(vocab=vocab)
        self.a = a
        self.b = b
        self.param = torch.nn.Parameter(torch.tensor(10.0))
        self.x = 0

    def forward(self, *args, **kwargs):
        self.x += 1

        return {'loss': self.a * self.x + self.b + self.param}
