import torch
from typing import Dict, Any
import allennlp
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary


@Model.register("dummy-failing")
class DummyModel(Model):
    def __init__(self, vocab: Vocabulary, a: float, b: float = 0):
        super().__init__(vocab=vocab)
        self.a = a
        self.b = b
        self.param = torch.nn.Parameter(torch.tensor(10.0))
        self.x = 0

    def forward(self, *args, **kwargs):
        self.x += 1

        return {"loss": self.a * self.x + self.b + self.param}


@Model.register("parameter-tying-failing")
class DummyModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        a: float,
        b: float,
        d: float,
        bool_value: bool,
        bool_value_not: bool,
        int_value: int,
        int_value_10: int,
    ):
        super().__init__(vocab=vocab)
        self.a = a
        self.b = b
        self.d = d
        self.param = torch.nn.Parameter(torch.tensor(10.0))
        assert a == b
        assert isinstance(bool_value, bool)
        assert isinstance(bool_value_not, bool)
        assert bool_value == (not bool_value_not)
        assert isinstance(int_value, int)
        assert isinstance(int_value_10, int)
        assert int_value + 10 == int_value_10
        assert d == 1
        self.x = 0
        self.count = 0

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.x += 1

        if self.count > 12:
            raise Exception

        self.count += 1

        return {"loss": self.a * self.x + self.b + self.param}
