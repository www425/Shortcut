import torch.nn as nn

from short_test import GradientReversal
from transformers import DebertaModel, BertModel

class deBERTa_classifier_test(nn.Module):
    def __init__(self, n_labels, hp_lambda=float(0.0), n_group=2):
        super().__init__()
        self.feature_extractor = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False, output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)
        self.gr = GradientReversal(hp_lambda)
        self.biased_model = nn.Linear(768, n_group)

    def forward(self, x, att_mask, return_features=False):
        features = self.feature_extractor(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x1 = self.classifier(x)
        y = self.gr(x)
        x2 = self.biased_model(y)
        if return_features:
            return x1, x2, features[:, 0, :]
        else:
            return x1, x2
class deBERTa_classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.feature_extractor = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False, output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.feature_extractor(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        if return_features:
            return x, features[:, 0, :]
        else:
            return x

class Bert_classifier_test(nn.Module):
    def __init__(self, n_labels, hp_lambda=float(0.0), n_group=2):
        super().__init__()
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, output_attentions=False,
                                                           output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)
        self.gr = GradientReversal(hp_lambda)
        self.biased_model = nn.Linear(768, n_group)

    def forward(self, x, att_mask, return_features=False):
        features = self.feature_extractor(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x1 = self.classifier(x)
        y = self.gr(x)
        x2 = self.biased_model(y)
        if return_features:
            return x1, x2, features[:, 0, :]
        else:
            return x1, x2

class Bert_classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, output_attentions=False,
                                                           output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.feature_extractor(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        if return_features:
            return x, features[:, 0, :]
        else:
            return x
