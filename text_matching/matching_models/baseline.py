from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("baseline")
class TextMatchingBaseline(Model):
    """
    This is a baseline model for a text matching task.

    Model structure: Embed 2 sentences, then encode each of them with a same
    Seq2VecEncoder (Siamese network), getting a single vector representing each.  
    Then concatenate two vectors and pass through a fc layer.
    
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(TextMatchingBaseline, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, 
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise/hypothesis : Dict[str, Variable], required, output of ``TextField.as_array()``.
        
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional, a scalar loss to be optimised.
        """
        embedded_premise = self.text_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise)
        encoded_premise = self.sentence_encoder(embedded_premise, premise_mask)

        embedded_hypo = self.text_field_embedder(hypothesis)
        hypo_mask = util.get_text_field_mask(hypothesis)
        encoded_hypo = self.sentence_encoder(embedded_hypo, hypo_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_premise, encoded_hypo], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


