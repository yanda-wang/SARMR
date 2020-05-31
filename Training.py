import torch
import os
import datetime
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from random import choices
from scipy.stats import entropy, boxcox
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import smtplib
from email.mime.text import MIMEText
from email.header import Header

from Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbedding
from Networks import DiscriminatorMLPPremium, DiscriminatorMLPPremiumForTuning
from Auxiliary import DataLoaderMedRec, DataLoaderGANSingleDistribution
from Parameters import Params
from Evaluation import EvaluationUtil

params = Params()
REAL_label = params.REAL_label  # 1
FAKE_label = params.FAKE_label  # 0

FITTED_DISTRIBUTION_FILE_NAME_TRAIN = params.fitted_distribution_file_name_train
FITTED_DISTRIBUTION_FILE_NAME_TEST = params.fitted_distribution_file_name_test
ENCODER_OUTPUT_FILE_NAME_TRAIN = params.ENCODER_OUTPUT_TRAIN_FILE_NAME
ENCODER_OUTPUT_FILE_NAME_TEST = params.ENCODER_OUTPUT_TEST_FILE_NAME


class TrainMedRec:
    def __init__(self, device, patient_records_file, ddi_matrix_file, concept2id_mapping_file, ehr_matrix_file=None):
        self.device = device
        self.patient_records_file = patient_records_file
        self.ddi_matrix_file = ddi_matrix_file
        self.concept2id_mapping_file = concept2id_mapping_file

        self.ddi_matrix = np.load(self.ddi_matrix_file)['ddi_matrix']
        concept2id_object = np.load(concept2id_mapping_file)
        self.concept2id_medications = concept2id_object['concept2id_prescriptions']
        self.medication_count = self.concept2id_medications.get_concept_count()
        self.diagnoses_count = concept2id_object['concept2id_diagnoses'].get_concept_count()
        self.procedures_count = concept2id_object['concept2id_procedures'].get_concept_count()

        self.ehr_matrix_file = ehr_matrix_file
        self.ehr_matrix = None
        if self.ehr_matrix_file is not None:
            self.ehr_matrix = np.load(self.ehr_matrix_file)

        self.evaluate_utils = EvaluationUtil(ddi_matrix_file)

    # loss value based on binary cross entropy (bce) loss and multi-label loss (multi)
    # target_medications: medical codes indicating current medications
    # predict_medications:results of decoder, dim=(1,medication count)
    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi):
        loss_bce_target = np.zeros((1, self.medication_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medication_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi
        return loss

    def get_performance_on_testset(self, encoder, decoder, data_loader, n_iteration, proportion_bce=0.9,
                                   proportion_multi=0.1):

        data_loader.shuffle()
        jaccard_avg, precision_avg, recall_avg, f1_avg, total_ddi_predict, total_ddi_groun_truth, loss_avg, prauc_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        predict_multi_hot_results = np.zeros(self.medication_count)  # for computing coverage

        for count in range(0, data_loader.patient_count):
            medications, diagnoses, procedures = data_loader.load_patient_record()
            query, memory_keys, memory_values = encoder(medications, diagnoses, procedures)
            predict_output = decoder(query, memory_keys, memory_values)
            target_medications = medications[-1]
            target_multi_hot = np.zeros(self.medication_count)
            target_multi_hot[target_medications] = 1
            predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
            predict_multi_hot = predict_prob.copy()

            index_nan = np.argwhere(np.isnan(predict_multi_hot))
            if index_nan.shape[0] != 0:
                predict_multi_hot = np.zeros_like(predict_multi_hot)

            predict_multi_hot[predict_multi_hot >= 0.5] = 1
            predict_multi_hot[predict_multi_hot < 0.5] = 0
            predict_medications = list(np.where(predict_multi_hot == 1)[0])

            jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
            precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
            recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
            f1 = self.evaluate_utils.metric_f1(precision, recall)
            prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

            loss = self.loss_function(medications[-1], predict_output, proportion_bce, proportion_multi)

            jaccard_avg += jaccard
            precision_avg += precision
            recall_avg += recall
            f1_avg += f1
            total_ddi_predict += self.evaluate_utils.get_ddi_rate(predict_medications)
            total_ddi_groun_truth += self.evaluate_utils.get_ddi_rate(target_medications)
            loss_avg += loss.item()
            prauc_avg += prauc
            predict_multi_hot_results[predict_medications] = 1

        jaccard_avg = jaccard_avg / count
        precision_avg = precision_avg / count
        recall_avg = recall_avg / count
        f1_avg = f1_avg / count
        delta_ddi_rate = (total_ddi_predict - total_ddi_groun_truth) / count
        nonzero_count = np.count_nonzero(predict_multi_hot_results)
        coverage = float(nonzero_count) / self.medication_count
        loss_avg = loss_avg / count
        prauc_avg = prauc_avg / count

        return jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate, coverage, loss_avg, prauc_avg

    def trainIters(self, encoder, decoder, encoder_optimizer, decoder_optimizer, data_loader_train, data_loader_test,
                   save_model_path, n_epoch, print_every_iteration=100, save_every_epoch=5,
                   proportion_bce=0.9, proportion_multi=0.1, medrec_trained_epoch=0, medrec_trained_iteration=0,
                   gan_trained_epoch=0, gan_trained_iteration=0, gan_patient_ddi_rate=0):

        start_epoch = medrec_trained_epoch + 1
        trained_n_iteration = medrec_trained_iteration
        total_planned_iteration = medrec_trained_iteration + n_epoch * data_loader_train.patient_count

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'medrec_loss.log'), 'a+')

        encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, patience=10, factor=0.1)
        decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, patience=10, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            data_loader_train.shuffle()
            print_loss = 0
            for iteration in range(1, 1 + int(data_loader_train.patient_count)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                trained_n_iteration += 1
                medication, diagnoses, procedures = data_loader_train.load_patient_record()
                query, memory_keys, memory_values = encoder(medication, diagnoses, procedures)
                predict_output = decoder(query, memory_keys, memory_values)
                loss = self.loss_function(medication[-1], predict_output, proportion_bce, proportion_multi)
                print_loss += loss.item()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                if iteration % print_every_iteration == 0:
                    print_loss_avg = print_loss / print_every_iteration
                    print_loss = 0.0

                    print(
                        'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}'.format(
                            epoch, datetime.datetime.now(), trained_n_iteration,
                            trained_n_iteration / total_planned_iteration * 100, print_loss_avg))
                    log_file.write(
                        'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}\n'.format(
                            epoch, datetime.datetime.now(), trained_n_iteration,
                            trained_n_iteration / total_planned_iteration * 100, print_loss_avg))

                    # encoder.eval()
                    # decoder.eval()
                    # jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate, coverage, print_loss_avg_on_test = self.get_performance_on_testset(
                    #     encoder, decoder, data_loader_test, print_every_iteration, proportion_bce, proportion_multi)
                    # encoder.train()
                    # decoder.train()
                    #
                    # print(
                    #     'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; converage_test: {:.4f}'.format(
                    #         epoch, datetime.datetime.now(), trained_n_iteration,
                    #         trained_n_iteration / total_planned_iteration * 100, print_loss_avg,
                    #         print_loss_avg_on_test, jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate,
                    #         coverage))
                    # log_file.write(
                    #     'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; converage_test: {:.4f}\n'.format(
                    #         epoch, datetime.datetime.now(), trained_n_iteration,
                    #         trained_n_iteration / total_planned_iteration * 100, print_loss_avg,
                    #         print_loss_avg_on_test, jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate,
                    #         coverage))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate, coverage, print_loss_avg_on_test, prauc_avg = self.get_performance_on_testset(
                encoder, decoder, data_loader_test, print_every_iteration, proportion_bce, proportion_multi)
            encoder.train()
            decoder.train()

            print(
                'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; converage_test: {:.4f}; prauc_test: {:.4f}'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration,
                    trained_n_iteration / total_planned_iteration * 100, print_loss_avg,
                    print_loss_avg_on_test, jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate,
                    coverage, prauc_avg))
            log_file.write(
                'epoch: {}; time: {}; Iteration: {}; Percent complet: {:.4f}%; train loss: {:.4f}; test loss: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; converage_test: {:.4f}; prauc_test: {:.4f}\n'.format(
                    epoch, datetime.datetime.now(), trained_n_iteration,
                    trained_n_iteration / total_planned_iteration * 100, print_loss_avg,
                    print_loss_avg_on_test, jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate,
                    coverage, prauc_avg))

            encoder_lr_scheduler.step(print_loss_avg)
            decoder_lr_scheduler.step(print_loss_avg)

            if epoch % save_every_epoch == 0:
                torch.save(
                    {'medrec_epoch': epoch,
                     'medrec_iteration': trained_n_iteration,
                     'encoder': encoder.state_dict(),
                     'decoder': decoder.state_dict(),
                     'encoder_optimizer': encoder_optimizer.state_dict(),
                     'decoder_optimizer': decoder_optimizer.state_dict(),
                     'medrec_avg_loss_train': print_loss_avg,
                     'medrec_avg_loss_test': print_loss_avg_on_test,
                     'medrec_patient_ddi_rate': data_loader_train.ddi_rate_threshold},
                    os.path.join(save_model_path,
                                 'medrec_{}_{}_{}_{}_{}_{}.checkpoint'.format(data_loader_train.ddi_rate_threshold,
                                                                              epoch, trained_n_iteration,
                                                                              gan_patient_ddi_rate, gan_trained_epoch,
                                                                              gan_trained_iteration)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
              encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, encoder_type, decoder_dropout_rate,
              decoder_regular_lambda, decoder_learning_rate, decoder_type, hop, attn_type_kv, attn_type_embedding,
              save_model_dir, proportion_bce, proportion_multi, medrec_patient_ddi_rate,
              n_epoch=50, print_every_iteration=100, save_every_epoch=5, load_model_name=None,
              pretrained_embedding_diagnoses=None, pretrained_embedding_procedures=None,
              pretrained_embedding_medications=None, discriminator_structure=None, discriminator_parameters=None):

        print('initializing >>>')
        embedding_diagnoses_np, embedding_procedures_np, embedding_medications_np = None, None, None
        if pretrained_embedding_diagnoses:
            embedding_diagnoses_np = np.load(pretrained_embedding_diagnoses)
        if pretrained_embedding_procedures:
            embedding_procedures_np = np.load(pretrained_embedding_procedures)
        if pretrained_embedding_medications:
            embedding_medications_np = np.load(pretrained_embedding_medications)

        if load_model_name:
            print('load model from checkpoint file: ', load_model_name)
            checkpoint = torch.load(load_model_name)

        print('build medrec model >>>')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                     self.procedures_count, encoder_n_layers, encoder_embedding_dropout_rate,
                                     encoder_gru_dropout_rate, embedding_diagnoses_np, embedding_procedures_np,
                                     bidirectional=encoder_bidirectional)
        if self.ehr_matrix is None:
            raise Exception("ehr matrix is required, which is None now.")
        decoder = DecoderKeyValueGCNMultiEmbedding(self.device, hidden_size, self.medication_count,
                                                   self.medication_count, hop, dropout_rate=decoder_dropout_rate,
                                                   attn_type_kv=attn_type_kv,
                                                   attn_type_embedding=attn_type_embedding, ehr_adj=self.ehr_matrix)

        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        encoder.train()
        decoder.train()

        print('build optimizer >>>')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate,
                                       weight_decay=encoder_regular_lambda)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate,
                                       weight_decay=decoder_regular_lambda)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        print('start training medrec model >>>')
        data_loader_train = DataLoaderMedRec(self.patient_records_file, medrec_patient_ddi_rate, 'train')
        data_loader_test = DataLoaderMedRec(self.patient_records_file, medrec_patient_ddi_rate, 'test')
        medrec_trained_epoch = 0
        medrec_trained_iteration = 0
        gan_trained_epoch = 0
        gan_trained_iteration = 0
        gan_patient_ddi_rate = 0

        if load_model_name:
            medrec_trained_n_epoch_sd = checkpoint['medrec_epoch']
            medrec_trained_n_iteration_sd = checkpoint['medrec_iteration']
            medrec_trained_epoch = medrec_trained_n_epoch_sd
            medrec_trained_iteration = medrec_trained_n_iteration_sd
            if 'gan_epoch' in checkpoint.keys():
                gan_trained_epoch_sd = checkpoint['gan_epoch']
                gan_trained_iteration_sd = checkpoint['gan_iteration']
                gan_patient_ddi_rate_sd = checkpoint['gan_patient_ddi_rate']
                gan_trained_epoch = gan_trained_epoch_sd
                gan_trained_iteration = gan_trained_iteration_sd
                gan_patient_ddi_rate = gan_patient_ddi_rate_sd

        # set save_model_path
        if discriminator_structure is not None:
            save_model_structure = encoder_type + '_' + decoder_type + '_' + str(encoder_n_layers) + '_' + str(
                input_size) + '_' + str(hidden_size) + '_' + str(encoder_bidirectional) + '_' + str(
                attn_type_kv) + '_' + str(attn_type_embedding) + '_' + str(discriminator_structure)
            save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(
                encoder_gru_dropout_rate) + '_' + str(encoder_regular_lambda) + '_' + str(
                encoder_learning_rate) + '_' + str(decoder_dropout_rate) + '_' + str(
                decoder_regular_lambda) + '_' + str(decoder_learning_rate) + '_' + str(hop) + '_' + str(
                discriminator_parameters)

        else:
            save_model_structure = encoder_type + '_' + decoder_type + '_' + str(encoder_n_layers) + '_' + str(
                input_size) + '_' + str(hidden_size) + '_' + str(encoder_bidirectional) + '_' + str(
                attn_type_kv) + '_' + str(attn_type_embedding) + '_None'
            save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(
                encoder_gru_dropout_rate) + '_' + str(encoder_regular_lambda) + '_' + str(
                encoder_learning_rate) + '_' + str(decoder_dropout_rate) + '_' + str(
                decoder_regular_lambda) + '_' + str(decoder_learning_rate) + '_' + str(hop) + '_None'
        save_model_path = os.path.join(save_model_dir, save_model_structure, save_model_parameters)

        self.trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, data_loader_train, data_loader_test,
                        save_model_path, n_epoch, print_every_iteration, save_every_epoch, proportion_bce,
                        proportion_multi, medrec_trained_epoch, medrec_trained_iteration, gan_trained_epoch,
                        gan_trained_iteration, gan_patient_ddi_rate)

    def get_hidden_states_between_thresholds(self, input_size, hidden_size, encoder_n_layers, encoder_bidirectional,
                                             load_model_name, patient_recodrs_file, data_mode,
                                             medrec_patient_ddi_rate_low, medrec_patient_ddi_rate_high, file_save_path,
                                             encoder_structure_str, encoder_parameters_str):
        """
        return the hidden states of patients whose ddi rate belongs to (medrec_patient_ddi_rate_low,medrec_patient_ddi_rate_high]
        """

        print('initializing >>>')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                     self.procedures_count, encoder_n_layers, bidirectional=encoder_bidirectional)
        print('load model from:', load_model_name)
        checkpoint = torch.load(load_model_name)
        encoder_sd = checkpoint['encoder']
        encoder.load_state_dict(encoder_sd)
        encoder = encoder.to(self.device)
        encoder.eval()

        print('load data from ', patient_recodrs_file)
        data_loader = DataLoaderMedRec(patient_recodrs_file, medrec_patient_ddi_rate_high, data_mode, True)

        print('build encoder output >>>')
        hidden_state = []
        for ddi_rate in np.arange(medrec_patient_ddi_rate_low + 0.1, medrec_patient_ddi_rate_high + 0.1, 0.1):
            current_ddi_rate = round(ddi_rate, 1)
            if current_ddi_rate <= 1:
                print(
                    'build encoder outputs for patients with ddi rate in ({:.1f},{:.1f}]'.format(current_ddi_rate - 0.1,
                                                                                                 current_ddi_rate))
                for _ in range(data_loader.patient_count_split[current_ddi_rate]):
                    medication, diagnoses, procedures = data_loader.load_patient_record(current_ddi_rate)
                    query, _, _ = encoder(medication, diagnoses, procedures)
                    hidden_state.append(query.squeeze(0).detach().cpu().numpy())

        hidden_state = np.array(hidden_state, dtype=np.float64)

        print('total number of samples:{}'.format(hidden_state.shape[0]))

        directory = os.path.join(file_save_path, encoder_structure_str, encoder_parameters_str)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('save data to directory:', directory)
        # pickle.dump(hidden_state, open(os.path.join(directory, 'encoder_hidden_state'), 'wb'), protocol=2)
        np.save(os.path.join(directory, 'encoder_hidden_state_{}_{}'.format(str(medrec_patient_ddi_rate_low),
                                                                            str(medrec_patient_ddi_rate_high))),
                hidden_state)

    def get_hidden_states(self, input_size, hidden_size, encoder_n_layers, encoder_bidirectional, encoder_type,
                          load_model_name, medrec_patient_ddi_rate, file_save_path, encoder_structure_str,
                          encoder_parameters_str, data_mode='train'):

        print('initializing >>>')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                     self.procedures_count, encoder_n_layers, bidirectional=encoder_bidirectional)
        print('load model from:', load_model_name)
        checkpoint = torch.load(load_model_name)
        encoder_sd = checkpoint['encoder']
        encoder.load_state_dict(encoder_sd)
        encoder = encoder.to(self.device)
        encoder.eval()
        print('build data loader based on:', self.patient_records_file)
        data_loader = DataLoaderMedRec(self.patient_records_file, medrec_patient_ddi_rate, data_mode)

        print('build hidden states >>>')
        hidden_state = []
        for _ in range(data_loader.patient_count):
            medication, diagnoses, procedures = data_loader.load_patient_record()
            query, _, _ = encoder(medication, diagnoses, procedures)
            hidden_state.append(query.squeeze(0).detach().cpu().numpy())
        hidden_state = np.array(hidden_state, dtype=np.float64)

        directory = os.path.join(file_save_path, encoder_structure_str, encoder_parameters_str)

        if not os.path.exists(directory):
            os.makedirs(directory)
        print('save data to directory:', directory)
        # pickle.dump(hidden_state, open(os.path.join(directory, 'encoder_hidden_state'), 'wb'), protocol=2)
        np.save(os.path.join(directory, 'encoder_hidden_state_{}'.format(str(medrec_patient_ddi_rate))), hidden_state)

    def isPD(self, B):
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def nearest_PD(self, A):
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if self.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    def fit_gaussian_from_encoder(self, input_size, hidden_size, encoder_n_layers, encoder_bidirectional,
                                  load_model_name, medrec_patient_ddi_rate, real_data_save_path, encoder_structure_str,
                                  encoder_parameters_str, decimals=6, boxcox_transform=False, boxcox_lambda=None,
                                  standard=False):

        print('initializing >>>')

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                     self.procedures_count, encoder_n_layers, bidirectional=encoder_bidirectional)
        print('load model from:', load_model_name)
        checkpoint = torch.load(load_model_name)
        encoder_sd = checkpoint['encoder']
        encoder.load_state_dict(encoder_sd)
        encoder = encoder.to(self.device)
        encoder.eval()
        print('build data loader based on:', self.patient_records_file)
        data_loader = DataLoaderMedRec(self.patient_records_file, medrec_patient_ddi_rate)

        print('build hidden states for distribution fitting >>>')
        hidden_state = []
        for _ in range(data_loader.patient_count):
            medication, diagnoses, procedures = data_loader.load_patient_record()
            query, _, _ = encoder(medication, diagnoses, procedures)
            hidden_state.append(query.squeeze(0).detach().cpu().numpy())
        hidden_state = np.array(hidden_state, dtype=np.float64)

        print('fit the gaussian distribution>>>')
        if boxcox_transform:
            min_value = np.min(hidden_state)
            if min_value <= 0:
                min_value = np.abs(min_value) + 0.0001
                hidden_state += min_value
            if boxcox_lambda is None:
                hidden_state = np.apply_along_axis(boxcox, 0, hidden_state, lmbda=boxcox_lambda)[0]
                hidden_state = np.column_stack(hidden_state)
            else:
                hidden_state = np.apply_along_axis(boxcox, 0, hidden_state, lmbda=boxcox_lambda)
            # boxcox_lambda: transform type for boxcox function
            #        lambda = -1. is a reciprocal transform.
            #        lambda = -0.5 is a reciprocal square root transform.
            #        lambda = 0.0 is a log transform.
            #        lambda = 0.5 is a square root transform.
            #        lambda = 1.0 is no transform.
            #        lambda=None: find the most likely lambda
        hidden_state = hidden_state.round(decimals=decimals)

        print('hidden state:', hidden_state)

        mean = np.mean(hidden_state, axis=0)
        if standard:
            cov = np.identity(hidden_state.shape[1])
        else:
            cov = np.cov(hidden_state, rowvar=False)

        if not self.isPD(cov):
            print('cov is not positive definite, covert to the nearest one')
            cov = self.nearest_PD(cov)
        print('mean:', mean)
        print('cov:', cov)

        print('sample data from the fitted distribution>>>')
        real_data = []
        for _ in range(hidden_state.shape[0]):
            real_data.append(np.random.multivariate_normal(mean, cov))
        real_data = np.array(real_data)

        directory = os.path.join(real_data_save_path, encoder_structure_str, encoder_parameters_str)
        print('save real data to:', directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        result = {'mean': mean,
                  'cov': cov,
                  'real_data': real_data}

        pickle.dump(result,
                    open(os.path.join(directory,
                                      'real_data_{}_{}_{}_ddi_rate_{}_standard_{}'.format(str(decimals),
                                                                                          str(boxcox_transform),
                                                                                          str(boxcox_lambda),
                                                                                          str(medrec_patient_ddi_rate),
                                                                                          str(standard))), 'wb'))


class TrainMedRecGANPremiumSingleDisNormal:
    def __init__(self, device, patient_records_file_accumulate, patient_records_file_separate, concept2id_mapping_file,
                 ddi_matrix_file, ehr_matrix_file, batch_size=8):
        self.device = device
        self.patient_records_file_accumulate = patient_records_file_accumulate
        self.patient_records_file_separate = patient_records_file_separate
        self.concept2id_file = concept2id_mapping_file
        self.ddi_matrix_file = ddi_matrix_file
        self.ehr_matrix_file = ehr_matrix_file
        self.batch_size = batch_size
        self.encoder_type = None

        self.ddi_matrix = np.load(self.ddi_matrix_file)['ddi_matrix']
        concept2id_object = np.load(concept2id_mapping_file)
        self.concept2id_medications = concept2id_object['concept2id_prescriptions']
        self.medication_count = self.concept2id_medications.get_concept_count()
        self.diagnoses_count = concept2id_object['concept2id_diagnoses'].get_concept_count()
        self.procedures_count = concept2id_object['concept2id_procedures'].get_concept_count()

        self.ehr_matrix_file = ehr_matrix_file
        self.ehr_matrix = None
        if self.ehr_matrix_file is not None:
            self.ehr_matrix = np.load(self.ehr_matrix_file)

        self.evaluate_utils = EvaluationUtil(ddi_matrix_file)

    def loss_function(self, target_medications, predict_medications, proportion_bce, proportion_multi):
        """
        loss value based on BCE loss and multi-label loss for MedRec
        :param target_medications: medical codes indicating ground truth medications
        :param predict_medications: results of decoder, dim=(1,medication count)
        :param proportion_bce:
        :param proportion_multi:
        :return:
        """
        loss_bce_target = np.zeros((1, self.medication_count))
        loss_bce_target[:, target_medications] = 1
        loss_multi_target = np.full((1, self.medication_count), -1)
        for idx, item in enumerate(target_medications):
            loss_multi_target[0][idx] = item

        loss_bce = F.binary_cross_entropy_with_logits(predict_medications,
                                                      torch.FloatTensor(loss_bce_target).to(self.device))
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(predict_medications),
                                              torch.LongTensor(loss_multi_target).to(self.device))
        loss = proportion_bce * loss_bce + proportion_multi * loss_multi
        return loss

    def get_performance_on_testset(self, encoder, decoder, data_loader, proportion_bce=0.9, proportion_multi=0.1):

        data_loader.shuffle()
        jaccard_avg, precision_avg, recall_avg, f1_avg, total_ddi_predict, total_ddi_groun_truth, loss_avg, prauc_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        predict_multi_hot_results = np.zeros(self.medication_count)  # for computing coverage

        for count in range(0, data_loader.patient_count):
            medications, diagnoses, procedures = data_loader.load_patient_record()
            query, memory_keys, memory_values = encoder(medications, diagnoses, procedures)
            predict_output = decoder(query, memory_keys, memory_values)
            target_medications = medications[-1]
            target_multi_hot = np.zeros(self.medication_count)
            target_multi_hot[target_medications] = 1
            predict_prob = torch.sigmoid(predict_output).detach().cpu().numpy()[0]
            predict_multi_hot = predict_prob.copy()

            index_nan = np.argwhere(np.isnan(predict_multi_hot))
            if index_nan.shape[0] != 0:
                predict_multi_hot = np.zeros_like(predict_multi_hot)

            predict_multi_hot[predict_multi_hot >= 0.5] = 1
            predict_multi_hot[predict_multi_hot < 0.5] = 0
            predict_medications = list(np.where(predict_multi_hot == 1)[0])

            jaccard = self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)
            precision = self.evaluate_utils.metric_precision(predict_medications, target_medications)
            recall = self.evaluate_utils.metric_recall(predict_medications, target_medications)
            f1 = self.evaluate_utils.metric_f1(precision, recall)
            prauc = self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

            loss = self.loss_function(medications[-1], predict_output, proportion_bce, proportion_multi)

            jaccard_avg += jaccard
            precision_avg += precision
            recall_avg += recall
            f1_avg += f1
            total_ddi_predict += self.evaluate_utils.get_ddi_rate(predict_medications)
            total_ddi_groun_truth += self.evaluate_utils.get_ddi_rate(target_medications)
            loss_avg += loss.item()
            prauc_avg += prauc
            predict_multi_hot_results[predict_medications] = 1

        jaccard_avg = jaccard_avg / count
        precision_avg = precision_avg / count
        recall_avg = recall_avg / count
        f1_avg = f1_avg / count
        delta_ddi_rate = (total_ddi_predict - total_ddi_groun_truth) / count
        nonzero_count = np.count_nonzero(predict_multi_hot_results)
        coverage = float(nonzero_count) / self.medication_count
        loss_avg = loss_avg / count
        prauc_avg = prauc_avg / count

        return jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate, coverage, loss_avg, prauc_avg

    def train_iters(self, encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, generator_optimizer,
                    discriminator_optimizer, data_loader_medrecGAN, data_loader_medrec_train,
                    data_loader_medrec_test, save_model_path, n_epoch, print_every_iteration=100,
                    save_every_epoch=5, gan_single_iteration=5, proportion_bce=0.9, proportion_multi=0.1,
                    trained_epoch=0, trained_medrec_iterations=0, trained_gan_iterations=0, gan_patient_ddi_rate=0,
                    encoder_combine_loss=False):
        start_epoch = trained_epoch + 1
        medrec_trained_iterations = trained_medrec_iterations
        gan_trained_iterations = trained_gan_iterations
        total_planned_iteration = medrec_trained_iterations + n_epoch * int(
            data_loader_medrecGAN.patient_count / self.batch_size)
        gan_criterion = nn.BCELoss()

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        log_file = open(os.path.join(save_model_path, 'MedRecGAN_loss.log'), 'a+')

        # encoder_lr_scheduler = ReduceLROnPlateau(encoder_optimizer, patience=10, factor=0.1)
        # decoder_lr_scheduler = ReduceLROnPlateau(decoder_optimizer, patience=10, factor=0.1)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            data_loader_medrecGAN.shullfe_all()

            print_loss_medrec = 0
            print_loss_D = 0
            print_loss_G = 0
            print_D_x = 0
            print_D_G_z1 = 0
            print_D_G_z2 = 0

            for iteration in range(1, int(data_loader_medrecGAN.patient_count / self.batch_size) + 1):

                """
                load data
                """
                batch_fake_medications = []
                batch_fake_diagnoses = []
                batch_fake_procedures = []
                batch_real_data = []

                # one class at a time for training GAN later, sample patients in the same DDI rate range
                sample_ddi_rate = choices(list(data_loader_medrecGAN.patient_count_split.keys()),
                                          list(data_loader_medrecGAN.patient_count_split.values()))[0]

                for batch in range(self.batch_size):
                    fake_medications, fake_diagnoses, fake_procedures, real_data = data_loader_medrecGAN.load_data(
                        sample_ddi_rate)

                    batch_fake_medications.append(fake_medications)
                    batch_fake_diagnoses.append(fake_diagnoses)
                    batch_fake_procedures.append(fake_procedures)
                    batch_real_data.append(real_data)

                real_data = torch.FloatTensor(batch_real_data).to(self.device)  # dim=(batch_size, hidden_size)

                """
                train medrec model
                """
                medrec_trained_iterations += 1
                for medication, diagnoses, procedures in zip(batch_fake_medications, batch_fake_diagnoses,
                                                             batch_fake_procedures):
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    query, memory_keys, memory_values = encoder(medication, diagnoses, procedures)
                    predict_output = decoder(query, memory_keys, memory_values)
                    loss = self.loss_function(medication[-1], predict_output, proportion_bce, proportion_multi)
                    print_loss_medrec += loss.item()
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                """
                train the GAN model for gan_single_iteration times
                """
                for gan_i in range(gan_single_iteration):
                    gan_trained_iterations += 1
                    ###
                    # update discriminator
                    ###
                    # train with all-real data
                    discriminator.zero_grad()
                    # soft noisy label & flip label
                    noisy_label = np.random.uniform(0.8, 1, (self.batch_size,))  # soft and noisy label
                    # flip the label with probability<=0.05
                    for label_i in range(noisy_label.shape[0]):
                        if np.random.uniform() <= 0.05:
                            noisy_label[label_i] = np.random.uniform(0, 0.2)
                    noisy_label = torch.Tensor(noisy_label).to(self.device)
                    _, output_real = discriminator(real_data)
                    output_real = output_real.view(-1)
                    errD_real = gan_criterion(output_real, noisy_label)
                    errD_real.backward()
                    D_x = output_real.mean().item()
                    print_D_x += D_x

                    # train with all-fake data
                    noisy_label = np.random.uniform(0, 0.2, (self.batch_size,))
                    for label_i in range(noisy_label.shape[0]):
                        if np.random.uniform() <= 0.05:
                            noisy_label[label_i] = np.random.uniform(0.8, 1)
                    noisy_label = torch.Tensor(noisy_label).to(self.device)

                    batch_fake_data_g = []
                    batch_fake_data_d = []
                    for fake_medications, fake_diagnoses, fake_procedures in zip(batch_fake_medications,
                                                                                 batch_fake_diagnoses,
                                                                                 batch_fake_procedures):
                        fake_data, _, _ = encoder(fake_medications, fake_diagnoses, fake_procedures)
                        batch_fake_data_g.append(fake_data)
                        batch_fake_data_d.append(fake_data.detach())
                    batch_fake_data_g = torch.cat(batch_fake_data_g, 0)
                    batch_fake_data_d = torch.cat(batch_fake_data_d, 0)

                    _, output_fake = discriminator(batch_fake_data_d)
                    output_fake = output_fake.view(-1)

                    errD_fake = gan_criterion(output_fake, noisy_label)
                    errD_fake.backward()
                    D_G_z1 = output_fake.mean().item()
                    print_D_G_z1 += D_G_z1
                    errD = errD_real + errD_fake
                    print_loss_D += errD.item()
                    discriminator_optimizer.step()

                    ###
                    # udpate generator(encoder): maximize log(D(G(z)))
                    ###
                    encoder.zero_grad()
                    noisy_label = np.random.uniform(0.8, 1, (self.batch_size,))  # soft and noisy label
                    # flip the label with probability<=0.05
                    for label_i in range(noisy_label.shape[0]):
                        if np.random.uniform() <= 0.05:
                            noisy_label[label_i] = np.random.uniform(0, 0.2)
                    noisy_label = torch.Tensor(noisy_label).to(self.device)
                    _, output = discriminator(batch_fake_data_g)
                    output = output.view(-1)
                    errG = gan_criterion(output, noisy_label)
                    errG.backward()
                    print_loss_G += errG.item()
                    D_G_z2 = output.mean().item()
                    generator_optimizer.step()
                    print_D_G_z2 += D_G_z2

                if iteration % print_every_iteration == 0:  # print log
                    print_loss_medrec_avg = print_loss_medrec / print_every_iteration
                    print_loss_D_avg = print_loss_D / print_every_iteration / gan_single_iteration
                    print_loss_G_avg = print_loss_G / print_every_iteration / gan_single_iteration
                    print_D_x_avg = print_D_x / print_every_iteration / gan_single_iteration
                    print_D_G_z1_avg = print_D_G_z1 / print_every_iteration / gan_single_iteration
                    print_D_G_z2_avg = print_D_G_z2 / print_every_iteration / gan_single_iteration
                    print_loss_medrec = 0
                    print_loss_D = 0
                    print_loss_G = 0
                    print_D_x = 0
                    print_D_G_z1 = 0
                    print_D_G_z2 = 0

                    gradient_discriminator = torch.mean(discriminator.output.weight.grad)
                    if self.encoder_type == 'LinearQuery':
                        gradient_generator = torch.mean(encoder.linear_embedding[-1].weight.grad)
                    else:
                        gradient_generator = 0

                    print(
                        'epoch: {};time: {}; iteration: {}; percent complete: {:.4f}%; gan iteration: {}; medrec_loss_train: {:.4f}; gan_loss_D: {:.4f}; gan_loss_G: {:.4f}; D(x):: {:.4f}; D(G(z1)): {:.4f}; D(G(z2)); {:.4f}; gradient_D: {:.5f}; gradient_G: {:.5f}'.format(
                            epoch, datetime.datetime.now(), medrec_trained_iterations,
                            medrec_trained_iterations / total_planned_iteration * 100, gan_trained_iterations,
                            print_loss_medrec_avg, print_loss_D_avg, print_loss_G_avg, print_D_x_avg, print_D_G_z1_avg,
                            print_D_G_z2_avg, gradient_discriminator, gradient_generator))

                    log_file.write(
                        'epoch: {};time: {}; iteration: {}; percent complete: {:.4f}%; gan iteration: {}; medrec_loss_train: {:.4f}; gan_loss_D: {:.4f}; gan_loss_G: {:.4f}; D(x):: {:.4f}; D(G(z1)): {:.4f}; D(G(z2)); {:.4f}; gradient_D: {:.5f}; gradient_G: {:.5f}\n'.format(
                            epoch, datetime.datetime.now(), medrec_trained_iterations,
                            medrec_trained_iterations / total_planned_iteration * 100, gan_trained_iterations,
                            print_loss_medrec_avg, print_loss_D_avg, print_loss_G_avg, print_D_x_avg, print_D_G_z1_avg,
                            print_D_G_z2_avg, gradient_discriminator, gradient_generator))

            encoder.eval()
            decoder.eval()
            jaccard_avg, precision_avg, recall_avg, f1_avg, delta_ddi_rate, coverage, print_loss_medrec_avg_test, prauc = self.get_performance_on_testset(
                encoder, decoder, data_loader_medrec_test, proportion_bce, proportion_multi)
            encoder.train()
            decoder.train()

            print(
                'epoch: {};time: {}; iteration: {}; percent complete: {:.4f}%; gan iteration: {}; medrec_loss_train: {:.4f}; medrec_loss_test: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; coverage_test: {:.4f}; prauc_test: {:.4f}; gan_loss_D: {:.4f}; gan_loss_G: {:.4f}; D(x): {:.4f}; D(G(z1)): {:.4f}; D(G(z2)): {:.4f}; gradient_D: {:.5f}; gradient_G: {:.5f}'.format(
                    epoch, datetime.datetime.now(), medrec_trained_iterations,
                    medrec_trained_iterations / total_planned_iteration * 100, gan_trained_iterations,
                    print_loss_medrec_avg, print_loss_medrec_avg_test, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    delta_ddi_rate, coverage, prauc, print_loss_D_avg, print_loss_G_avg, print_D_x_avg,
                    print_D_G_z1_avg, print_D_G_z2_avg, gradient_discriminator, gradient_generator))

            log_file.write(
                'epoch: {};time: {}; iteration: {}; percent complete: {:.4f}%; gan iteration: {}; medrec_loss_train: {:.4f}; medrec_loss_test: {:.4f}; jaccard_test: {:.4f}; precision_test: {:.4f}; recall_test: {:.4f}; f1_test: {:.4f}; delta_ddi_rate_test: {:.4f}; coverage_test: {:.4f}; prauc_test: {:.4f}; gan_loss_D: {:.4f}; gan_loss_G: {:.4f}; D(x): {:.4f}; D(G(z1)): {:.4f}; D(G(z2)): {:.4f}; gradient_D: {:.5f}; gradient_G: {:.5f}\n'.format(
                    epoch, datetime.datetime.now(), medrec_trained_iterations,
                    medrec_trained_iterations / total_planned_iteration * 100, gan_trained_iterations,
                    print_loss_medrec_avg, print_loss_medrec_avg_test, jaccard_avg, precision_avg, recall_avg, f1_avg,
                    delta_ddi_rate, coverage, prauc, print_loss_D_avg, print_loss_G_avg, print_D_x_avg,
                    print_D_G_z1_avg, print_D_G_z2_avg, gradient_discriminator, gradient_generator))

            if epoch % save_every_epoch == 0:
                torch.save({'epoch': epoch,
                            'trained_medrec_iterations': medrec_trained_iterations,
                            'trained_gan_iterations': gan_trained_iterations,
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'discriminator': discriminator.state_dict(),
                            'encoder_optimizer': encoder_optimizer.state_dict(),
                            'decoder_optimizer': decoder_optimizer.state_dict(),
                            'generator_optimizer': generator_optimizer.state_dict(),
                            'discriminator_optimizer': discriminator_optimizer.state_dict(),
                            'medrec_avg_loss_train': print_loss_medrec_avg,
                            'medrec_avg_loss_test': print_loss_medrec_avg_test,
                            'medrec_patient_ddi_rate': data_loader_medrecGAN.ddi_rate_threshold,
                            'gan_loss_G': print_loss_G_avg,
                            'gan_loss_D': print_loss_D_avg,
                            'gan_D_x': print_D_x_avg,
                            'gan_D_G_z1': print_D_G_z1_avg,
                            'gan_D_G_z2': print_D_G_z2_avg},
                           os.path.join(save_model_path,
                                        'MedRecGAN_{}_{}_{}_{}'.format(
                                            data_loader_medrecGAN.ddi_rate_threshold, epoch, medrec_trained_iterations,
                                            gan_patient_ddi_rate)))

        log_file.close()

    def train(self, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
              encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, encoder_type, decoder_dropout_rate,
              decoder_regular_lambda, decoder_learning_rate, decoder_type, hop, attn_type_kv, attn_type_embedding,
              gan_lr, gan_regular_lambda, gan_single_iteration, discriminator_dropout_rate,
              discriminator_n_hidden_layers, discriminator_dim_B, discriminator_dim_C, medrec_patient_ddi_rate,
              gan_patient_ddi_rate, proprotion_bce, proportion_multi, real_data_file, save_model_path, n_epoch=40,
              print_every_iteration=100, save_every_epoch=5, load_model_name=None, pretrained_embedding_diagnoses=None,
              pretrained_embedding_procedures=None, pretrained_embedding_medications=None):
        print('initializing>>>')
        embedding_diagnoses_np, embedding_procedures_np, embedding_medications_np = None, None, None
        if pretrained_embedding_diagnoses:
            embedding_diagnoses_np = np.load(pretrained_embedding_diagnoses)
        if pretrained_embedding_procedures:
            embedding_procedures_np = np.load(pretrained_embedding_procedures)
        if pretrained_embedding_medications:
            embedding_medications_np = np.load(pretrained_embedding_medications)

        if load_model_name:
            print('load model from checkpoint file: ', load_model_name)
            checkpoint = torch.load(load_model_name)

        print('bulid model...')
        self.encoder_type = encoder_type

        encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                     self.procedures_count, encoder_n_layers, encoder_embedding_dropout_rate,
                                     encoder_gru_dropout_rate, embedding_diagnoses_np, embedding_procedures_np,
                                     bidirectional=encoder_bidirectional)
        decoder = DecoderKeyValueGCNMultiEmbedding(self.device, hidden_size, self.medication_count,
                                                   self.medication_count, hop, dropout_rate=decoder_dropout_rate,
                                                   attn_type_kv=attn_type_kv,
                                                   attn_type_embedding=attn_type_embedding, ehr_adj=self.ehr_matrix)

        discriminator = DiscriminatorMLPPremium(hidden_size, discriminator_dropout_rate, discriminator_n_hidden_layers,
                                                discriminator_dim_B, discriminator_dim_C, params.USE_CUDA)

        if load_model_name:
            encoder_sd = checkpoint['encoder']
            decoder_sd = checkpoint['decoder']
            discriminator_sd = checkpoint['discriminator']
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
            discriminator.load_state_dict(discriminator_sd)

        encoder.to(self.device)
        decoder.to(self.device)
        discriminator.to(self.device)
        encoder.train()
        decoder.train()
        discriminator.train()

        print('build optimizer...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate,
                                       weight_decay=encoder_regular_lambda)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate,
                                       weight_decay=decoder_regular_lambda)
        generator_optimizer = optim.Adam(encoder.parameters(), lr=gan_lr, weight_decay=gan_regular_lambda)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=gan_lr, weight_decay=gan_regular_lambda)
        if load_model_name:
            encoder_optimizer_sd = checkpoint['encoder_optimizer']
            decoder_optimizer_sd = checkpoint['decoder_optimizer']
            generator_optimizer_sd = checkpoint['generator_optimizer']
            discriminator_optimizer_sd = checkpoint['discriminator_optimizer']

            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)
            generator_optimizer.load_state_dict(generator_optimizer_sd)
            discriminator_optimizer.load_state_dict(discriminator_optimizer_sd)

        print('build data loader...')
        data_loader_medrecGAN = DataLoaderGANSingleDistribution(real_data_file, self.patient_records_file_separate,
                                                                medrec_patient_ddi_rate)
        data_loader_medrec_train = DataLoaderMedRec(self.patient_records_file_accumulate, medrec_patient_ddi_rate)

        data_loader_medrec_test = DataLoaderMedRec(self.patient_records_file_accumulate, medrec_patient_ddi_rate,
                                                   'test')

        print('start training>>>')
        trained_epoch = 0
        trained_medrec_iteration = 0
        trained_gan_iteration = 0
        if load_model_name:
            trained_epoch = checkpoint['epoch']
            trained_medrec_iteration = checkpoint['trained_medrec_iterations']
            trained_gan_iteration = checkpoint['trained_gan_iterations']
        # set save_path
        save_model_structure = encoder_type + '_' + decoder_type + '_' + str(encoder_n_layers) + '_' + str(
            input_size) + '_' + str(hidden_size) + '_' + str(encoder_bidirectional) + '_' + str(
            attn_type_kv) + '_' + str(attn_type_embedding) + '_MLPPremium_' + str(discriminator_n_hidden_layers)

        save_model_parameters = str(encoder_embedding_dropout_rate) + '_' + str(
            encoder_gru_dropout_rate) + '_' + str(encoder_regular_lambda) + '_' + str(
            encoder_learning_rate) + '_' + str(decoder_dropout_rate) + '_' + str(
            decoder_regular_lambda) + '_' + str(decoder_learning_rate) + '_' + str(hop) + '_' + str(
            discriminator_dropout_rate) + '_' + str(gan_lr) + '_' + str(gan_regular_lambda) + '_' + str(
            discriminator_dim_B) + '_' + str(discriminator_dim_C)
        save_model_path = os.path.join(save_model_path, save_model_structure, save_model_parameters)

        self.train_iters(encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, generator_optimizer,
                         discriminator_optimizer, data_loader_medrecGAN, data_loader_medrec_train,
                         data_loader_medrec_test, save_model_path, n_epoch, print_every_iteration, save_every_epoch,
                         gan_single_iteration, proprotion_bce, proportion_multi, trained_epoch,
                         trained_medrec_iteration, trained_gan_iteration, gan_patient_ddi_rate)


def MedRecTraining(input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
                   encoder_gru_dropout_rate, encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate,
                   encoder_type, decoder_dropout_rate, decoder_regular_lambda, decoder_learning_rate, decoder_type, hop,
                   attn_type_kv, attn_type_embedding, save_model_dir, proportion_bce, proportion_multi,
                   medrec_patient_ddi_rate, n_epoch=40, print_every_iteration=100, save_every_epoch=5,
                   load_model_name=None, pretrained_embedding_diagnoses=None, pretrained_embedding_procedures=None,
                   pretrained_embedding_medications=None, discriminator_structure=None, discriminator_parameters=None):
    module = TrainMedRec(params.device, params.PATIENT_RECORDS_FILE_ACCUMULATE, params.DDI_MATRIX_FILE,
                         params.CONCEPTID_FILE, params.EHR_MATRIX_FILE)
    module.train(input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
                 encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, encoder_type,
                 decoder_dropout_rate, decoder_regular_lambda, decoder_learning_rate, decoder_type, hop, attn_type_kv,
                 attn_type_embedding, save_model_dir, proportion_bce, proportion_multi, medrec_patient_ddi_rate,
                 n_epoch, print_every_iteration, save_every_epoch, load_model_name, pretrained_embedding_diagnoses,
                 pretrained_embedding_procedures, pretrained_embedding_medications, discriminator_structure,
                 discriminator_parameters)


def MedRecGANTraining(batch_size, input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate,
                      encoder_gru_dropout_rate, encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate,
                      encoder_type, decoder_dropout_rate, decoder_regular_lambda, decoder_learning_rate, decoder_type,
                      hop, attn_type_kv, attn_type_embedding, gan_lr, gan_regular_lambda, gan_single_iteration,
                      discriminator_dropout_rate, discriminator_n_hidden_layers, discriminator_dim_B,
                      discriminator_dim_C, medrec_patient_ddi_rate, gan_patient_ddi_rate, proprotion_bce,
                      proportion_multi, real_data_file, save_model_path, n_epoch=40, print_every_iteration=100,
                      save_every_epoch=5, load_model_name=None, pretrained_embedding_diagnoses=None,
                      pretrained_embedding_procedures=None, pretrained_embedding_medications=None):
    model = TrainMedRecGANPremiumSingleDisNormal(params.device, params.PATIENT_RECORDS_FILE_ACCUMULATE,
                                                 params.PATIENT_RECORDS_FILE_SEPARATE, params.CONCEPTID_FILE,
                                                 params.DDI_MATRIX_FILE, params.EHR_MATRIX_FILE, batch_size)
    model.train(input_size, hidden_size, encoder_n_layers, encoder_embedding_dropout_rate, encoder_gru_dropout_rate,
                encoder_bidirectional, encoder_regular_lambda, encoder_learning_rate, encoder_type,
                decoder_dropout_rate, decoder_regular_lambda, decoder_learning_rate, decoder_type, hop,
                attn_type_kv, attn_type_embedding, gan_lr, gan_regular_lambda, gan_single_iteration,
                discriminator_dropout_rate, discriminator_n_hidden_layers, discriminator_dim_B, discriminator_dim_C,
                medrec_patient_ddi_rate, gan_patient_ddi_rate, proprotion_bce, proportion_multi, real_data_file,
                save_model_path, n_epoch, print_every_iteration, save_every_epoch, load_model_name,
                pretrained_embedding_diagnoses, pretrained_embedding_procedures, pretrained_embedding_medications)


def fit_distribution_from_encoder(input_size, hidden_size, encoder_n_layers,
                                  encoder_bidirectional, encoder_type, load_model_name, medrec_patient_ddi_rate,
                                  real_data_save_path, encoder_structure_str, encoder_parameters_str, decimals=6,
                                  boxcox_transform=False, boxcox_lambda=None, standard=False):
    module = TrainMedRec(params.device, params.PATIENT_RECORDS_FILE_ACCUMULATE, params.DDI_MATRIX_FILE,
                         params.CONCEPTID_FILE, params.EHR_MATRIX_FILE)
    module.fit_gaussian_from_encoder(input_size, hidden_size, encoder_n_layers, encoder_bidirectional, encoder_type,
                                     load_model_name, medrec_patient_ddi_rate, real_data_save_path,
                                     encoder_structure_str, encoder_parameters_str, decimals, boxcox_transform,
                                     boxcox_lambda, standard)


def get_hidden_states(input_size, hidden_size, encoder_n_layers, encoder_bidirectional, encoder_type, load_model_name,
                      medrec_patient_ddi_rate, file_save_path, encoder_structure_str, encoder_parameters_str):
    module = TrainMedRec(params.device, params.PATIENT_RECORDS_FILE_ACCUMULATE, params.DDI_MATRIX_FILE,
                         params.CONCEPTID_FILE, params.EHR_MATRIX_FILE)
    module.get_hidden_states(input_size, hidden_size, encoder_n_layers, encoder_bidirectional, encoder_type,
                             load_model_name, medrec_patient_ddi_rate, file_save_path, encoder_structure_str,
                             encoder_parameters_str)


def get_hidden_states_between_threshold(input_size, hidden_size, encoder_n_layer, encoder_bidirectional,
                                        encoder_type, load_model_name, patient_records_file, data_mode,
                                        ddi_rate_low, ddi_rate_high, save_data_path, encoder_structure_str,
                                        encoder_parameters_str):
    module = TrainMedRec(params.device, params.PATIENT_RECORDS_FILE_ACCUMULATE, params.DDI_MATRIX_FILE,
                         params.CONCEPTID_FILE, params.EHR_MATRIX_FILE)
    module.get_hidden_states_between_thresholds(input_size, hidden_size, encoder_n_layer, encoder_bidirectional,
                                                encoder_type, load_model_name, patient_records_file, data_mode,
                                                ddi_rate_low, ddi_rate_high, save_data_path, encoder_structure_str,
                                                encoder_parameters_str)
