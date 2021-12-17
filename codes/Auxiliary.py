import pickle

import dill
import numpy as np


# dataloader for training medication recommendation model
class DataLoaderMedRec:
    # patient_record_file_path: data/split_patient_records
    # ddi_rate_threshold: only patient records with a ddi rate less than or equal to ddi_rate_threshold will be sampled
    # data_mode: tarin, test, or validation, will sample data from corresponding dataset
    # data_split: True or False, True=patient records are stored separately according to their ddi rate
    def __init__(self, patient_records_file_name, ddi_rate_threshold, data_mode='train', data_split=False):
        self.patient_records_file_name = patient_records_file_name
        self.ddi_rate_threshold = ddi_rate_threshold
        self.data_mode = data_mode
        self.data_split = data_split
        self.patient_records = None
        self.patient_count = 0
        self.patient_count_split = {}
        self.read_index = None

        if self.data_split:
            patient_records = dill.load(open(self.patient_records_file_name, 'rb'))[data_mode]
            self.patient_records = {}
            self.read_index = {}
            self.patient_count = 0
            for rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
                self.patient_records[round(rate, 1)] = patient_records[round(rate, 1)]
                self.read_index[round(rate, 1)] = 0
                self.patient_count_split[round(rate, 1)] = len(self.patient_records[round(rate, 1)])
                self.patient_count = self.patient_count + self.patient_count_split[round(rate, 1)]
        else:
            self.patient_records = dill.load(open(self.patient_records_file_name, 'rb'))[data_mode][ddi_rate_threshold]
            self.patient_count = len(self.patient_records)
            self.read_index = 0

    # shullef the patient records
    def shuffle(self, ddi_rate=-0.1):
        if ddi_rate < 0:
            np.random.shuffle(self.patient_records)
            self.read_index = 0
        else:
            np.random.shuffle(self.patient_records[ddi_rate])
            self.read_index[ddi_rate] = 0

    def load_patient_record(self, ddi_rate=-0.1):
        if ddi_rate < 0:
            return self.load_patient_record_split_false()
        else:
            return self.load_patient_record_split_true(ddi_rate)

    def load_patient_record_split_false(self):
        if self.read_index >= self.patient_count:
            # print('index out of range, shuffle patient records')
            self.shuffle()
        picked_patient = self.patient_records[self.read_index]  # pick a patient
        medications = [admission[0] for admission in picked_patient]
        diagnoses = [admission[1] for admission in picked_patient]
        procedures = [admission[2] for admission in picked_patient]
        self.read_index += 1
        return medications, diagnoses, procedures

    def load_patient_record_split_true(self, ddi_rate):
        if self.read_index[ddi_rate] >= self.patient_count_split[ddi_rate]:
            # print('index out of range, shuffle patient records')
            self.shuffle(ddi_rate)
        picked_patient = self.patient_records[ddi_rate][self.read_index[ddi_rate]]
        medications = [admission[0] for admission in picked_patient]
        diagnoses = [admission[1] for admission in picked_patient]
        procedures = [admission[2] for admission in picked_patient]
        self.read_index[ddi_rate] += 1
        return medications, diagnoses, procedures


class DataLoaderGANSingleDistribution:
    def __init__(self, fitted_distribution_file_name, patient_records_file_name, ddi_rate_threshold, data_mode='train'):
        self.fitted_distribution_file_name = fitted_distribution_file_name
        self.patient_records_file_name = patient_records_file_name
        self.ddi_rate_threshold = ddi_rate_threshold
        self.data_mode = data_mode

        # self.distribution_data = np.load(
        #     self.fitted_distribution_file_name)['real_data']  # np.array,dim=(#data point, data dimension)
        self.distribution_data = dill.load(open(self.fitted_distribution_file_name, 'rb'))['real_data']
        self.dataloader_medrec = DataLoaderMedRec(self.patient_records_file_name, self.ddi_rate_threshold,
                                                  self.data_mode, True)

        self.patient_count = self.dataloader_medrec.patient_count
        self.patient_count_split = self.dataloader_medrec.patient_count_split  # dict, key: ddi rate, value: #patients with the corresponding ddi rate
        self.distribution_n = len(self.distribution_data)
        self.read_index = 0

    def shullfe_all(self):
        for ddi_rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
            self.dataloader_medrec.shuffle(round(ddi_rate, 1))
        self.shuffle_distribution()

    def shuffle_distribution(self):
        np.random.shuffle(self.distribution_data)
        self.read_index = 0

    def load_data(self, ddi_rate):
        if self.read_index >= self.distribution_n:
            self.shuffle_distribution()
        medication, diagnoses, procedures = self.dataloader_medrec.load_patient_record(ddi_rate)
        sampled_distribution = self.distribution_data[self.read_index]
        self.read_index += 1
        return medication, diagnoses, procedures, sampled_distribution

