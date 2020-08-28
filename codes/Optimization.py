import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch

from torch import optim
from skorch.utils import params_for
from warnings import filterwarnings

from Parameters import Params
from Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbedding

params = Params()
PATIENT_RECORDS_FILE = params.PATIENT_RECORDS_FILE_ACCUMULATE
CONCEPTID_FILE = params.CONCEPTID_FILE
EHR_MATRIX_FILE = params.EHR_MATRIX_FILE
device = params.device  # torch.device("cuda" if USE_CUDA else "cpu")
MEDICATION_COUNT = params.MEDICATION_COUNT  # 143
DIAGNOSES_COUNT = params.DIAGNOSES_COUNT  # 1999
PROCEDURES_COUNT = params.PROCEDURES_COUNT  # 1327

OPT_SPLIT_TAG_ADMISSION = params.OPT_SPLIT_TAG_ADMISSION  # -1
OPT_SPLIT_TAG_VARIABLE = params.OPT_SPLIT_TAG_VARIABLE  # -2
OPT_MODEL_MAX_EPOCH = params.OPT_MODEL_MAX_EPOCH
OPT_PATIENT_DDI_RATE = params.OPT_PATIENT_DDI_RATE

OPT_PRETRAINED_EMBEDDING_MEDICATION = params.PRETRAINED_EMBEDDING_MEDICATION  # None
OPT_PRETRAINED_EMBEDDING_DIAGNOSES = params.PRETRAINED_EMBEDDING_DIAGNOSES  # None
OPT_PRETRAINED_EMBEDDING_PROCEDURES = params.PRETRAINED_EMBEDDING_PROCEDURES  # None

LOSS_PROPORTION_BCE = params.LOSS_PROPORTION_BCE  # 0.9
LOSS_PROPORTION_MULTI = params.LOSS_PROPORTION_Multi_Margin  # 0.1


# define the encoder-decoder model
class MedRecSeq2Set(nn.Module):
    def __init__(self, device, encoder_type, decoder_type, input_size, hidden_size, **kwargs):
        super().__init__()

        self.encoder = EncoderLinearQuery(device=device, input_size=input_size, hidden_size=hidden_size,
                                          **params_for('encoder', kwargs))
        self.decoder = DecoderKeyValueGCNMultiEmbedding(device=device, hidden_size=hidden_size,
                                                        **params_for('decoder', kwargs))

        self.device = device

    def forward(self, x):
        # tensor to numpy
        # x = np.array(x.tolist()[0])
        split_x = np.split(x, np.where(x == OPT_SPLIT_TAG_VARIABLE)[0])
        medications, diagnoses, procedures = split_x[0], split_x[1][1:], split_x[2][1:]
        medications = self.split_into_admission(medications)
        diagnoses = self.split_into_admission(diagnoses)
        procedures = self.split_into_admission(procedures)

        query, memory_keys, memory_values = self.encoder(medications, diagnoses, procedures)
        predict_output = self.decoder(query, memory_keys, memory_values)
        return predict_output

    def split_into_admission(self, variables):
        split_variables = np.split(variables, np.where(variables == OPT_SPLIT_TAG_ADMISSION)[0])
        final_variables = [admission.tolist()[1:] for admission in split_variables[1:]]
        final_variables = [split_variables[0].tolist()] + final_variables
        return final_variables


# warp the encoder-decoder model in skorch
class MedRecTrainer(skorch.NeuralNet):
    def __init__(self, *args, optimizer_encoder=optim.Adam, optimizer_decoder=optim.Adam, **kwargs):
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder
        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, triggered_directly=True):
        kwargs = self._get_params_for('optimizer_encoder')
        self.optimizer_encoder_ = self.optimizer_encoder(self.module_.encoder.parameters(), **kwargs)
        kwargs = self._get_params_for('optimizer_decoder')
        self.optimizer_decoder_ = self.optimizer_decoder(self.module_.decoder.parameters(), **kwargs)

    def train_step(self, Xi, yi, **fit_params):
        yi = skorch.utils.to_numpy(yi).tolist()[0]

        self.module_.train()
        self.optimizer_encoder_.zero_grad()
        self.optimizer_decoder_.zero_grad()

        y_pred = self.infer(Xi)
        loss = self.get_loss(y_pred, yi)
        loss.backward()

        self.optimizer_encoder_.step()
        self.optimizer_decoder_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def infer(self, Xi, yi=None):
        Xi = skorch.utils.to_numpy(Xi)[0]
        return self.module_(Xi)

    def get_loss(self, y_pred, y_true, **kwargs):

        loss_bce_target = np.zeros((1, MEDICATION_COUNT))
        loss_bce_target[:, y_true] = 1
        loss_multi_target = np.full((1, MEDICATION_COUNT), -1)
        for idx, item in enumerate(y_true):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(y_pred, torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(y_pred),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = LOSS_PROPORTION_BCE * loss_bce + LOSS_PROPORTION_MULTI * loss_multi
        return loss

    def _predict(self, X, most_probable=True):
        filterwarnings('error')
        y_probas = []

        for output in self.forward_iter(X, training=False):
            if most_probable:
                predict_prob = skorch.utils.to_numpy(torch.sigmoid(output))[0]
                predict_multi_hot = predict_prob.copy()

                index_nan = np.argwhere(np.isnan(predict_multi_hot))
                if index_nan.shape[0] != 0:
                    predict_multi_hot = np.zeros_like(predict_multi_hot)

                predict_multi_hot[predict_multi_hot >= 0.5] = 1
                predict_multi_hot[predict_multi_hot < 0.5] = 0
                predict = np.where(predict_multi_hot == 1)[0]
            else:
                predict = skorch.utils.to_numpy(torch.sigmoid(output))[0]
            y_probas.append(predict)

            # try:
            #     for output in self.forward_iter(X, training=False):
            #         if most_probable:
            #             predict_prob = skorch.utils.to_numpy(torch.sigmoid(output))[0]
            #             predict_multi_hot = predict_prob.copy()
            #             predict_multi_hot[predict_multi_hot >= 0.5] = 1
            #             predict_multi_hot[predict_multi_hot < 0.5] = 0
            #             predict = np.where(predict_multi_hot == 1)[0]
            #         else:
            #             predict = skorch.utils.to_numpy(torch.sigmoid(output))[0]
            #         y_probas.append(predict)
            # except RuntimeWarning:
            #     print(output)

        return np.array(y_probas)

    def predict_proba(self, X):
        return self._predict(X, most_probable=False)

    def predict(self, X):
        return self._predict(X, most_probable=True)


class DiscriminatorTrainer(skorch.NeuralNet):
    def train_step(self, Xi, yi, **fit_params):
        self.module_.train()
        self.optimizer_.zero_grad()
        y_pred = self.infer(Xi)
        loss = self.get_loss(y_pred, yi)
        loss.backward()
        self.optimizer_.step()
        return {'loss': loss, 'y_pred': y_pred}
