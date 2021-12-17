import dill
import os
import torch
import pickle
import numpy as np

from Networks import EncoderLinearQuery, DecoderKeyValueGCNMultiEmbedding
from Auxiliary import DataLoaderMedRec
from itertools import combinations
from sklearn.metrics import average_precision_score
from Parameters import Params

params = Params()


class EvaluationMedRec:
    def __init__(self, device, concept2id_mapping_file, patient_records_file, ddi_matrix_file,
                 predict_prob_threshold=0.5, ehr_matrix_file=None):
        self.device = device
        concept2id_object = dill.load(open(concept2id_mapping_file, 'rb'))
        self.concept2id_medications = concept2id_object['concept2id_prescriptions']
        self.medication_count = self.concept2id_medications.get_concept_count()
        self.diagnoses_count = concept2id_object['concept2id_diagnoses'].get_concept_count()
        self.procedures_count = concept2id_object['concept2id_procedures'].get_concept_count()
        self.patient_records_file = patient_records_file  # need to specify parameter mode and patient_ddi_rate
        self.predict_prob_threshold = predict_prob_threshold
        # self.embedding_diagnoses = None
        # self.embedding_procedures = None
        # self.embedding_medications = None
        self.data_loader = None
        self.encoder = None
        self.decoder = None
        self.load_model_name = None
        self.evaluate_utils = EvaluationUtil(ddi_matrix_file)
        self.ddi_matrix_file = ddi_matrix_file
        self.ehr_matrix_file = ehr_matrix_file
        self.ehr_matrix = None
        if self.ehr_matrix_file is not None:
            self.ehr_matrix = dill.load(open(self.ehr_matrix_file, 'rb'))

    def get_ddi_rate(self, medications):
        return self.evaluate_utils.get_ddi_rate(medications)

    def metric_jaccard_similarity(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_jaccard_similarity(predict_medications, target_medications)

    def metric_precision(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_precision(predict_medications, target_medications)

    def metric_recall(self, predict_medications, target_medications):
        return self.evaluate_utils.metric_recall(predict_medications, target_medications)

    def metric_f1(self, precision, recall):
        return self.evaluate_utils.metric_f1(precision, recall)

    def metric_prauc(self, predict_prob, target_multi_hot):
        return self.evaluate_utils.precision_auc(predict_prob, target_multi_hot)

    def get_predict_medications(self, medications, diagnoses, procedures):
        query, memory_keys, memory_values = self.encoder(medications, diagnoses, procedures)
        output = self.decoder(query, memory_keys, memory_values)

        predict_prob = torch.sigmoid(output).detach().cpu().numpy()[0]  # predict probability for each medication
        predict_multi_hot = predict_prob.copy()

        index_nan = np.argwhere(np.isnan(predict_multi_hot))
        if index_nan.shape[0] != 0:
            predict_multi_hot = np.zeros_like(predict_multi_hot)

        predict_multi_hot[predict_multi_hot >= self.predict_prob_threshold] = 1
        predict_multi_hot[predict_multi_hot < self.predict_prob_threshold] = 0
        predict_medications = np.where(predict_multi_hot == 1)[0]
        return predict_medications, predict_prob

    def get_predict_medications_attention(self, medications, diagnoses, procedures):
        query, memory_keys, memory_values = self.encoder(medications, diagnoses, procedures)
        output, all_attention = self.decoder(query, memory_keys, memory_values)

        predict_prob = torch.sigmoid(output).detach().cpu().numpy()[0]  # predict probability for each medication
        predict_multi_hot = predict_prob.copy()

        index_nan = np.argwhere(np.isnan(predict_multi_hot))
        if index_nan.shape[0] != 0:
            predict_multi_hot = np.zeros_like(predict_multi_hot)

        predict_multi_hot[predict_multi_hot >= self.predict_prob_threshold] = 1
        predict_multi_hot[predict_multi_hot < self.predict_prob_threshold] = 0
        predict_medications = np.where(predict_multi_hot == 1)[0]
        return predict_medications, predict_prob, all_attention

    def evaluateIters(self, save_predict_result_file):
        patient_count = self.data_loader.patient_count
        total_metric_jaccard = 0.0
        total_metric_precision = 0.0
        total_metric_recall = 0.0
        total_metric_f1 = 0.0
        total_metric_ddi_rate_predict = 0.0
        total_metric_ddi_rate_ground_truth = 0.0
        predict_multi_hot_results = np.zeros(self.medication_count)
        total_prauc = 0.0
        tmp_predict_medications = []

        for _ in range(patient_count):
            medications, diagnoses, procedures = self.data_loader.load_patient_record()
            predict_medications, predict_prob = self.get_predict_medications(medications, diagnoses, procedures)
            target_medications = medications[-1]
            target_multi_hot = np.zeros(self.medication_count)
            target_multi_hot[target_medications] = 1

            tmp_predict_medications.append(predict_medications)

            jaccard = self.metric_jaccard_similarity(predict_medications, target_medications)
            precision = self.metric_precision(predict_medications, target_medications)
            recall = self.metric_recall(predict_medications, target_medications)
            f1 = self.metric_f1(precision, recall)
            prauc = self.metric_prauc(predict_prob, target_multi_hot)
            total_metric_jaccard += jaccard
            total_metric_precision += precision
            total_metric_recall += recall
            total_metric_f1 += f1
            total_prauc += prauc
            predict_multi_hot_results[predict_medications] = 1
            total_metric_ddi_rate_predict += self.get_ddi_rate(predict_medications)
            total_metric_ddi_rate_ground_truth += self.get_ddi_rate(target_medications)

        nonzero_count = np.count_nonzero(predict_multi_hot_results)
        coverage = float(nonzero_count) / self.medication_count

        print('  jaccard:', total_metric_jaccard / patient_count)
        print('precision:', total_metric_precision / patient_count)
        print('   recall:', total_metric_recall / patient_count)
        print('       f1:', total_metric_f1 / patient_count)
        print(' ddi rate:', total_metric_ddi_rate_ground_truth / patient_count)
        print('predict ddi:', total_metric_ddi_rate_predict / patient_count)
        print('delta ddi:',
              (total_metric_ddi_rate_predict - total_metric_ddi_rate_ground_truth) / total_metric_ddi_rate_ground_truth)
        print(' coverage:', coverage)
        print('    prauc:', total_prauc / patient_count)

        if not os.path.exists(save_predict_result_file):
            os.makedirs(save_predict_result_file)
        pickle.dump(tmp_predict_medications,
                    open(os.path.join(save_predict_result_file, 'predict_medications.pkl'), 'wb'))

    def evaluate(self, load_model_name, save_predict_result_file, patient_ddi_rate, input_size, hidden_size,
                 encoder_n_layers, encoder_bidirectional, hop, attn_type_kv='dot', attn_type_embedding='dot',
                 data_mode='validation'):
        print('load model from: ', load_model_name)
        self.load_model_name = load_model_name
        checkpoint = torch.load(self.load_model_name)
        # checkpoint=torch.load(self.load_model_name,map_location='cpu)
        encoder_sd = checkpoint['encoder']
        decoder_sd = checkpoint['decoder']
        print('building models >>>')
        self.encoder = EncoderLinearQuery(self.device, input_size, hidden_size, self.diagnoses_count,
                                          self.procedures_count, encoder_n_layers,
                                          bidirectional=encoder_bidirectional)
        if self.ehr_matrix is None:
            raise Exception('ehr matrix is required, which is None now')
        self.decoder = DecoderKeyValueGCNMultiEmbedding(self.device, hidden_size, self.medication_count,
                                                        self.medication_count, hop, attn_type_kv=attn_type_kv,
                                                        attn_type_embedding=attn_type_embedding,
                                                        ehr_adj=self.ehr_matrix)

        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

        self.data_loader = DataLoaderMedRec(self.patient_records_file, patient_ddi_rate, data_mode=data_mode)
        print('evaluating >>>')
        self.evaluateIters(save_predict_result_file)


class EvaluationUtil:
    def __init__(self, ddi_matrix_file):
        # self.ddi_matrix = np.load(ddi_matrix_file)['ddi_matrix']
        self.ddi_matrix = dill.load(open(ddi_matrix_file, 'rb'))['ddi_matrix']

    def precision_auc(self, predict_prob, target_prescriptions):
        # all_micro = []
        # for b in range(len(target_prescriptions)):
        #     all_micro.append(average_precision_score(target_prescriptions[b], predict_prob[b], average='macro'))

        return average_precision_score(target_prescriptions, predict_prob, average='macro')

    def get_ddi_rate(self, prescriptions):
        med_pair_count = 0.0
        ddi_count = 0.0
        ddi_rate = 0.0
        for med_i, med_j in combinations(prescriptions, 2):
            med_pair_count += 1
            if self.ddi_matrix[med_i][med_j] == 1:
                ddi_count += 1
        if med_pair_count != 0:
            ddi_rate = ddi_count / med_pair_count
        return ddi_rate

    def metric_jaccard_similarity(self, predict_prescriptions, target_prescriptions):
        union = list(set(predict_prescriptions) | set(target_prescriptions))
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        jaccard = float(len(intersection)) / len(union)
        return jaccard

    def metric_precision(self, predict_prescriptions, target_prescriptions):
        if len(set(predict_prescriptions)) == 0:
            return 0
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        precision = float(len(intersection)) / len(set(predict_prescriptions))
        return precision

    def metric_recall(self, predict_prescriptions, target_prescriptions):
        intersection = list(set(predict_prescriptions) & set(target_prescriptions))
        recall = float(len(intersection)) / len(set(target_prescriptions))
        return recall

    def metric_f1(self, precision, recall):
        if precision + recall == 0:
            return 0
        f1 = 2.0 * precision * recall / (precision + recall)
        return f1


def evaluate(load_model_name, save_predict_result_file, patient_ddi_rate, input_size, hidden_size, encoder_n_layers,
             encoder_bidirectional, hop, attn_type_kv, attn_type_embedding):
    module = EvaluationMedRec(params.device, params.CONCEPTID_FILE, params.PATIENT_RECORDS_FILE_ACCUMULATE,
                              params.DDI_MATRIX_FILE, 0.5, params.EHR_MATRIX_FILE)
    module.evaluate(load_model_name, save_predict_result_file, patient_ddi_rate, input_size, hidden_size,
                    encoder_n_layers, encoder_bidirectional, hop, attn_type_kv, attn_type_embedding)
