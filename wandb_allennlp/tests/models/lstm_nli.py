import torch
import allennlp
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np


class Seq2VecTwoClass(Model):
    RNNS = {'LSTM': torch.nn.LSTM, 'GRU': torch.nn.GRU, 'RNN': torch.nn.RNN}

    @classmethod
    def _create_encoder(cls, rnn: str, input_size: int, hidden_size: int,
                        num_layers: int,
                        bidirectional: bool) -> PytorchSeq2VecWrapper:
        """
        Create the encoder
        """
        try:
            rnn_type: Callable[[Any], Any] = cls.RNNS[rnn]
        except KeyError as ke:
            raise ValueError("rnn should be str and one from {}".format(
                list(cls.RNNS.keys()))) from ke

        return PytorchSeq2VecWrapper(
            rnn_type(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True))

    def _create_encoder_network(self, rnn: str, input_size: int,
                                hidden_size: int, num_layers: int,
                                bidirectional: bool, **kwargs) -> None:
        self.seq2box_p = self._create_encoder(rnn, input_size, hidden_size,
                                              num_layers, bidirectional)
        self.seq2box_h = self._create_encoder(rnn, input_size, hidden_size,
                                              num_layers, bidirectional)

    @classmethod
    def _create_embedder(cls,
                         vocab_size: int,
                         embd_dim: int,
                         padding_index: int = 0) -> BasicTextFieldEmbedder:
        emb = Embedding(vocab_size, embd_dim, padding_index=padding_index)

        return BasicTextFieldEmbedder({'tokens': emb})

    def __init__(self,
                 vocab: Vocabulary,
                 input_size: int,
                 hidden_size: int,
                 projection_size: Optional[int] = None,
                 rnn: str = 'LSTM',
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 debug=False) -> None:
        super().__init__(vocab)
        self.num_classes = 3
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.rnn = rnn
        self.num_layers = num_layers
        self.embedding: BasicTextFieldEmbedder = self._create_embedder(
            self.vocab.get_vocab_size(), input_size)
        self._create_encoder_network(rnn, input_size, hidden_size, num_layers,
                                     bidirectional)

        self.projection_size = projection_size

        if self.projection_size is not None:
            self.linear_p = torch.nn.Linear(
                self.num_directions * hidden_size,
                self.projection_size,
                bias=False)
            self.linear_h = torch.nn.Linear(
                self.num_directions * hidden_size,
                self.projection_size,
                bias=True)
            self.linear_final = torch.nn.Linear(
                self.projection_size, self.num_classes, bias=True)
        else:
            self.linear_p = torch.nn.Identity()
            self.linear_h = torch.nn.Identity()
            self.linear_final = torch.nn.Linear(
                self.num_directions * self.hidden_size,
                self.num_classes,
                bias=True)
        self.loss: torch.nn.Module = torch.nn.CrossEntropyLoss(
            reduction='mean')
        self.accuracy = allennlp.training.metrics.CategoricalAccuracy()
        self.debug = debug

    def _get_entailment_prob(self, prem: torch.Tensor,
                             hypo: torch.Tensor) -> torch.Tensor:
        linear_comb = torch.tanh(self.linear_p(prem) + self.linear_h(hypo))
        score = self.linear_final(linear_comb)

        return score

    def forward(
            self,
            premise: Dict[str, torch.Tensor],
            hypothesis: Dict[str, torch.Tensor],
            label: Optional[torch.LongTensor] = None,
            metadata: Optional[List[Any]] = None) -> Dict[str, torch.Tensor]:
        prem_mask: torch.LongTensor = get_text_field_mask(premise).to(
            device=(dict(premise).popitem()[1]
                    ).device)  # set device depending on input
        hypo_mask: torch.LongTensor = get_text_field_mask(hypothesis).to(
            device=(dict(premise).popitem()[1]
                    ).device)  # set device depending on input

        prem_emb: torch.Tensor = self.embedding(premise)
        hypo_emb: torch.Tensor = self.embedding(hypothesis)
        prem_rep: torch.Tensor = self.seq2box_p(prem_emb, prem_mask)
        hypo_rep: torch.Tensor = self.seq2box_h(hypo_emb, hypo_mask)

        if self.debug:
            temp = prem_rep.detach().cpu()

            if torch.isinf(temp).any():
                breakpoint()
            temp = hypo_rep.detach().cpu()

            if torch.isinf(temp).any():
                breakpoint()
        scores = self._get_entailment_prob(prem_rep, hypo_rep)
        output_dict = {'scores': scores.detach()}

        if label is not None:
            loss = self.loss(scores, label)  # label =1 for entailment

            if self.debug:
                check = loss.detach().cpu()

                if bool(torch.isnan(check).any()) or bool(
                        torch.isinf(check).any()):
                    breakpoint()
            output_dict['loss'] = loss
            output_dict['label'] = label
            with torch.no_grad():
                self.accuracy(scores, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('nli-lstm')
class Seq2VecSingleEncoderTwoClass(Seq2VecTwoClass):
    def _create_encoder_network(self, rnn: str, input_size: int,
                                hidden_size: int, num_layers: int,
                                bidirectional: bool, **kwargs) -> None:
        self.seq2box_p = self._create_encoder(rnn, input_size, hidden_size,
                                              num_layers, bidirectional)
        self.seq2box_h = self.seq2box_p
