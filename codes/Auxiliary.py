import pickle
import numpy as np
import csv
import math
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split


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
            patient_records = np.load(self.patient_records_file_name)[data_mode]
            self.patient_records = {}
            self.read_index = {}
            self.patient_count = 0
            for rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
                self.patient_records[round(rate, 1)] = patient_records[round(rate, 1)]
                self.read_index[round(rate, 1)] = 0
                self.patient_count_split[round(rate, 1)] = len(self.patient_records[round(rate, 1)])
                self.patient_count = self.patient_count + self.patient_count_split[round(rate, 1)]
        else:
            self.patient_records = np.load(self.patient_records_file_name)[data_mode][ddi_rate_threshold]
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

        self.distribution_data = np.load(
            self.fitted_distribution_file_name)['real_data']  # np.array,dim=(#data point, data dimension)
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


# dataloader for training GAN model
# class DataLoaderGAN:
#     def __init__(self, fitted_distribution_file_name, patient_records_file_name, ddi_rate_threshold, data_mode='train'):
#         self.fitted_distribution_file_name = fitted_distribution_file_name
#         self.patient_records_file_name = patient_records_file_name
#         self.ddi_rate_threshold = ddi_rate_threshold
#         self.data_mode = data_mode
#         # self.distribution_data = np.load(
#         #     self.fitted_distribution_file_name)  # dict, key:cluster number, value=np.array,dim=[#samples in the same cluster, embedding_dim]
#
#         self.distribution_data = pickle.load(open(self.fitted_distribution_file_name, 'rb'), encoding='latin1')
#         self.dataloader_medrec = DataLoaderMedRec(self.patient_records_file_name, self.ddi_rate_threshold,
#                                                   self.data_mode, True)
#
#         self.patient_count = self.dataloader_medrec.patient_count
#         self.patient_count_split = self.dataloader_medrec.patient_count_split  # dict, key: ddi rate, value: #patients with the corresponding ddi rate
#         self.distribution_n = len(self.distribution_data.keys())
#         self.distribution_count_split = {}
#         self.read_index = {}
#         for key, value in self.distribution_data.items():
#             self.read_index[key] = 0
#             self.distribution_count_split[key] = len(value)
#         self.distribution_count_label = len(self.read_index.keys())
#
#     def shullfe_all(self):
#         for ddi_rate in np.arange(0, self.ddi_rate_threshold + 0.1, 0.1):
#             self.dataloader_medrec.shuffle(round(ddi_rate, 1))
#         for n in np.arange(0, self.distribution_n):
#             self.shuffle_distribution(n)
#
#     def shuffle_distribution(self, distribution_n):
#         # self.dataloader_medrec.shuffle()
#         # np.random.shuffle(self.distribution_data)
#         # self.read_index = 0
#         np.random.shuffle(self.distribution_data[distribution_n])
#         self.read_index[distribution_n] = 0
#
#     def load_data(self, distribution_n, ddi_rate):
#         if self.read_index[distribution_n] >= self.distribution_count_split[distribution_n]:
#             # print('index out of range, shuffle patient records')
#             self.shuffle_distribution(distribution_n)
#         medication, diagnoses, procedures = self.dataloader_medrec.load_patient_record(ddi_rate)
#         sampled_distribution = self.distribution_data[distribution_n][self.read_index[distribution_n]]
#         self.read_index[distribution_n] += 1
#         return medication, diagnoses, procedures, sampled_distribution
#
#
# class DataLoaderInception:
#     def __init__(self, fitted_distribution_file_name):
#         self.fitted_distribution_file_name = fitted_distribution_file_name
#         self.count_sample = 0  # #samples
#         self.count_sample_split = {}  # #samples of each cluster
#         self.count_label = 0  # #labels
#         self.read_index = 0
#
#         # distribution_data = np.load(
#         #     self.fitted_distribution_file_name)  # dict, key:cluster number, value=np.array,dim=[#samples in the same cluster, embedding_dim]
#         distribution_data = pickle.load(open(self.fitted_distribution_file_name, 'rb'), encoding='latin1')
#         self.distribution_data = self.pack_data(
#             distribution_data)  # np.array,dim=(#sample,embedding_dim+1),data[:,-1]=label
#         self.shuffle()
#
#     def pack_data(self, distribution_data):
#         data = []
#         for key, value in distribution_data.items():
#
#             self.count_sample_split[key] = len(value)
#             self.count_label += 1
#             self.count_sample += self.count_sample_split[key]
#             for sample in value:
#                 sample = list(sample)
#                 sample.append(key)
#                 data.append(sample)
#
#         data = np.array(data)
#         return data
#
#     def shuffle(self):
#         np.random.shuffle(self.distribution_data)
#         self.read_index = 0
#
#     def load_data(self, batch_size):
#         data = []
#         label = []
#         for _ in range(batch_size):
#             if self.read_index >= self.count_sample:
#                 self.shuffle()
#             sample = self.distribution_data[self.read_index]
#             self.read_index += 1
#             current_data = sample[:-1]
#             current_label = sample[-1]
#             data.append(current_data)
#             label.append(current_label)
#         data = np.array(data)
#         label = np.array(label)
#         return data, label


class Concept2Id(object):
    def __init__(self):
        self.concept2id = {}
        self.id2concept = {}

    # given a sequence of medical concepts, obtain their ids and store the mapping
    def add_concepts(self, concepts):
        for item in concepts:
            if item not in self.concept2id.keys():
                # self.id2concept[len(self.concept2id)] = item
                self.concept2id[item] = len(self.concept2id)
                self.id2concept[self.concept2id.get(item)] = item

    def get_concept_count(self):
        return len(self.concept2id)


# read patient information and get their ids
# patient_info_file_path: patient information, each line for a admission
# file format:subject_id,hadm_id,admittime,prescriptions_set(ATC3 code seperated by ;), diagnoses_set, procedures_set
# concept2id_output_file_path: output file for results,
# contains a dic with three keys: concept2id_prescriptions, concept2id_diagnoses,concept2id_procedures

def map_concepts2id(patient_info_file_path, concept2id_output_file_path):
    concept2id_prescriptions = Concept2Id()
    concept2id_diagnoses = Concept2Id()
    concept2id_procedures = Concept2Id()

    patient_info_file = open(patient_info_file_path, 'r')
    for line in patient_info_file:
        line = line.rstrip('\n')
        patient_info = line.split(',')
        prescriptions = patient_info[3].split(';')
        diagnoses = patient_info[4].split(';')
        procedures = patient_info[5].split(';')
        concept2id_prescriptions.add_concepts(prescriptions)
        concept2id_diagnoses.add_concepts(diagnoses)
        concept2id_procedures.add_concepts(procedures)
    patient_info_file.close()
    dump_objects = {'concept2id_prescriptions': concept2id_prescriptions, 'concept2id_diagnoses': concept2id_diagnoses,
                    'concept2id_procedures': concept2id_procedures}
    pickle.dump(dump_objects, open(concept2id_output_file_path, 'wb'))


# get the ddi drug pair indicated by stitch_id
# drug_ddi_file: ddi type source file
# mode: the most frequently ddi type (head) or the least frequently ddi type (tail)
# top_N: the number of ddi types that will be considered
# ddi_info_output_file_pre: the file prefix the information will be written to
def get_ddi_information(drug_ddi_file, mode, top_N, ddi_info_output_file_pre):
    drug_ddi_df = pd.read_csv(drug_ddi_file)
    ddi_most_pd = drug_ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    if mode == 'tail':
        ddi_most_pd = ddi_most_pd.iloc[-top_N:, :]
    else:
        ddi_most_pd = ddi_most_pd.iloc[0:top_N, :]

    fliter_ddi_df = drug_ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)
    ddi_info_output_file = ddi_info_output_file_pre + "_" + mode + "_top" + str(top_N)
    pickle.dump({'ddi_info': ddi_df}, open(ddi_info_output_file, 'wb'))

    # read test
    # test_ddi_info = np.load(ddi_info_output_file)['ddi_info']
    # for index, row in test_ddi_info.iterrows():
    #     print(row['STITCH 1'], row['STITCH 2'])


# construct the ddi matrix
# ddi_file_path: ddi information from the TWOSIDES dataset
# stitch2atc_file_path: map between stitch_id and atc
# concept2id_file_path: map between atc to id
# ddi_matrix_output_file: output file of the ddi matrix
# two stitch_ids from ddi_file_path -> get the atcs of these two stitch_ids from stitch2atc_file_path
# -> get the id of these atcs as coordinates in the matrix
def construct_ddi_matrix(ddi_file_path, stitch2atc_file_path, concept2id_file_path, ddi_matrix_output_file):
    concept2id_prescriptions = np.load(concept2id_file_path).get('concept2id_prescriptions')

    stitch2atc_dict = {}
    with open(stitch2atc_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            stitch_id = line[0]
            atc_set = line[1:]
            stitch2atc_dict[stitch_id] = atc_set

    prescriptions_size = concept2id_prescriptions.get_concept_count()
    ddi_matrix = np.zeros((prescriptions_size, prescriptions_size))
    # ddi_file = open(ddi_file_path)
    ddi_info = np.load(ddi_file_path)['ddi_info']

    for index, row in ddi_info.iterrows():
        stitch_id1 = row['STITCH 1']
        stitch_id2 = row['STITCH 2']

        if stitch_id1 in stitch2atc_dict.keys() and stitch_id2 in stitch2atc_dict.keys():
            for atc_i in stitch2atc_dict[stitch_id1]:
                for atc_j in stitch2atc_dict[stitch_id2]:
                    atc_i = atc_i[:4]
                    atc_j = atc_j[:4]
                    if atc_i in concept2id_prescriptions.concept2id.keys() and atc_j in concept2id_prescriptions.concept2id.keys() and atc_i != atc_j:
                        ddi_matrix[
                            concept2id_prescriptions.concept2id.get(atc_i), concept2id_prescriptions.concept2id.get(
                                atc_j)] = 1
                        ddi_matrix[
                            concept2id_prescriptions.concept2id.get(atc_j), concept2id_prescriptions.concept2id.get(
                                atc_i)] = 1

    ddi_matrix_object = {'ddi_matrix': ddi_matrix}
    ddi_matrix_output_file = ddi_matrix_output_file + '_' + ddi_file_path.split('_')[-2] + '_' + \
                             ddi_file_path.split('_')[-1]
    pickle.dump(ddi_matrix_object, open(ddi_matrix_output_file, 'wb'))

    # statistic information
    unique, counts = np.unique(ddi_matrix, return_counts=True)
    print(dict(zip(unique, counts)))


def construct_cooccurance_matrix(patient_records_file, concept2id_file, matrix_output_file, patient_ddi_rate):
    concetp2id_medications = np.load(concept2id_file).get('concept2id_prescriptions')
    medication_count = concetp2id_medications.get_concept_count()
    matrix = np.zeros((medication_count, medication_count))
    patient_records = np.load(patient_records_file)['train'][1.0]
    count = 0
    for patient in patient_records:
        for admission in patient:
            if admission[-1][0] <= patient_ddi_rate:
                medications = admission[0]
                for med_i, med_j in combinations(medications, 2):
                    count += 1
                    matrix[med_i][med_j] = 1
                    matrix[med_j][med_i] = 1
    pickle.dump(matrix, open(matrix_output_file + '_' + str(patient_ddi_rate), 'wb'))

    unique, counts = np.unique(matrix, return_counts=True)
    print(dict(zip(unique, counts)))


# transform patient records that consist of medical concept codes to records that consist of correponding ids
# patient_info_file_path:the file that contains patient information,
# file format: subject_id,hadm_id,admittime,prescriptions_set(ATC3 code seperated by ;), diagnoses_set, procedures_set
# concept2id_mapping_file_path: map between concepts and ids
# ddi_matrix_file_path: ddi matrix, for computing ddi rate for each admission
# patient_record_output_file_path: output file for the result
def construct_patient_records(patient_info_file_path, concept2id_mapping_file_path, ddi_matrix_file_path,
                              patient_records_output_file_path):
    ddi_matrix = np.load(ddi_matrix_file_path)['ddi_matrix']

    def get_ddi_rate(medications):
        med_pair_count = 0.0
        ddi_count = 0.0
        ddi_rate = 0
        for med_i, med_j in combinations(medications, 2):
            med_pair_count += 1
            if ddi_matrix[med_i][med_j] == 1:
                ddi_count += 1
        if med_pair_count != 0:
            ddi_rate = ddi_count / med_pair_count
        return ddi_rate

    concept2id_object = np.load(concept2id_mapping_file_path)
    concept2id_prescriptions = concept2id_object.get('concept2id_prescriptions')
    concept2id_diagnoses = concept2id_object.get('concept2id_diagnoses')
    concept2id_procedures = concept2id_object.get('concept2id_procedures')

    tmp_ddi_rate = []

    patient_records = []
    patient = []
    last_subject_id = ''
    patient_info_file = open(patient_info_file_path)
    for line in patient_info_file:
        admission = []
        line = line.rstrip('\n').split(',')
        current_subject_id = line[0]
        prescriptions = line[3].split(';')
        diagnoses = line[4].split(';')
        procedures = line[5].split(';')
        admission.append([concept2id_prescriptions.concept2id.get(item) for item in prescriptions])
        admission.append([concept2id_diagnoses.concept2id.get(item) for item in diagnoses])
        admission.append([concept2id_procedures.concept2id.get(item) for item in procedures])
        ddi_rate = get_ddi_rate(admission[0])
        # admission.append([get_ddi_rate(admission[0])])
        admission.append([ddi_rate])
        tmp_ddi_rate.append(round(ddi_rate, 1))
        if current_subject_id == last_subject_id:
            patient.append(admission)
        else:
            # if len(patient) != 0 and filter_patient_records(patient):
            if len(patient) != 0:
                patient_records.append(patient)
            patient = []
            patient.append(admission)
        last_subject_id = current_subject_id
    patient_records.append(patient)
    patient_info_file.close()
    dump_object = {'patient_records': patient_records}
    pickle.dump(dump_object, open(patient_records_output_file_path, 'wb'))

    # statistic infomation
    unique, counts = np.unique(tmp_ddi_rate, return_counts=True)
    print(dict(zip(unique, counts)))
    print(sum(tmp_ddi_rate) / float(len(tmp_ddi_rate)))


# split data into traning set, test set, and validation set
# patient_records_file_path: patient records generated by function constructe_patient_records
# sampling_data_output_file_path: output file for the results, contains a dic with keys 'trian','test', and 'validation'
def data_sampling(patient_records_file_path, sampling_data_seperate_output_file_path,
                  sampling_data_accumulate_output_file_path):
    ddi_rate_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    patient_records_split_by_ddi_rate = {}
    for ddi_rate in ddi_rate_bins:
        patient_records_split_by_ddi_rate[ddi_rate] = []
    patient_records = np.load(patient_records_file_path)['patient_records']
    for patient in patient_records:
        for idx, admission in enumerate(patient):
            ddi_rate = admission[3][0]
            current_patient_record = patient[:idx + 1]
            patient_records_split_by_ddi_rate[math.ceil(ddi_rate * 10.0) / 10].append(current_patient_record)
            # if len(admission[0]) <= MAX_sequence_length:
            #     patient_records_split_by_ddi_rate[math.ceil(ddi_rate * 10.0) / 10].append(current_patient_record)
            # if ddi_rate <= 0.5:
            #     patient_records_split_by_ddi_rate[0.5].append(current_patient_record)
            # else:
            #     patient_records_split_by_ddi_rate[math.ceil(ddi_rate * 10.0) / 10].append(current_patient_record)

    train, test, validation = {}, {}, {}
    for ddi_rate, patients in patient_records_split_by_ddi_rate.items():
        train_patients, test_patients = train_test_split(patients, test_size=0.1)
        train_patients, validation_patients = train_test_split(train_patients, test_size=0.1)
        train[ddi_rate], test[ddi_rate], validation[ddi_rate] = train_patients, test_patients, validation_patients
    pickle.dump({'train': train, 'test': test, 'validation': validation},
                open(sampling_data_seperate_output_file_path, 'wb'))

    print('patient records information stored seperately by ddi rate')
    print('training dataset:')
    for key, value in train.items():
        print(key, len(value), end=';')
    print()
    print('test dataset')
    for key, value in test.items():
        print(key, len(value), end=';')
    print()
    print('validation dataset')
    for key, value in validation.items():
        print(key, len(value), end=';')
    print()

    for ddi_rate in ddi_rate_bins[1:]:
        train[ddi_rate] = train[ddi_rate] + train[round(ddi_rate - 0.1, 1)]
        test[ddi_rate] = test[ddi_rate] + test[round(ddi_rate - 0.1, 1)]
        validation[ddi_rate] = validation[ddi_rate] + validation[round(ddi_rate - 0.1, 1)]
    pickle.dump({'train': train, 'test': test, 'validation': validation},
                open(sampling_data_accumulate_output_file_path, 'wb'))

    print('patient records information stored accumulately by ddi rate')
    print('training dataset:')
    for key, value in train.items():
        print(key, len(value), end=';')
    print()
    print('test dataset')
    for key, value in test.items():
        print(key, len(value), end=';')
    print()
    print('validation dataset')
    for key, value in validation.items():
        print(key, len(value), end=';')
    print()

