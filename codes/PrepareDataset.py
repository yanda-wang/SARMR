import csv
import dill
import itertools
import math

import pandas as pd
import numpy as np

from itertools import combinations
from sklearn.model_selection import train_test_split
from tqdm import tqdm

med_file = 'data/PRESCRIPTIONS.csv'
diag_file = 'data/DIAGNOSES_ICD.csv'
procedure_file = 'data/PROCEDURES_ICD.csv'

ndc2atc_file = 'data/ndc2atc_level4.csv'
cid_atc = 'data/drug-atc.csv'
ndc2rxnorm_file = 'data/ndc2rxnorm_mapping.txt'
drug_ddi_file = 'data/drug-DDI.csv'
drug_stitch2atc_file = 'data/drug_stitch2atc.csv'
DDI_MATRIX_FILE = 'data/ddi_matrix_tail_top100.pkl'
EHR_MATRIX_FILE = 'data/ehr_matrix_1.0.pkl'

PATIENT_RECORDS_FILE = 'data/patient_records.pkl'
PATIENT_RECORDS_FINAL_FILE = 'data/patient_records_final.pkl'
PATIENT_RECORDS_FILE_ACCUMULATE = 'data/patient_records_accumulate_tail_top100.pkl'
PATIENT_RECORDS_FILE_SEPARATE = 'data/patient_records_separate_tail_top100.pkl'

CONCEPTID_FILE = 'data/concepts2id_mapping.pkl'

# DIAGNOSES_INDEX = 0
# PROCEDURES_INDEX = 1
# MEDICATIONS_INDEX = 2

VOC_FILE = 'data/voc.pkl'
GRAPH_FILE = 'data/graph.pkl'


# ===================处理原始EHR数据，选取对应记录================
# we borrow part of the codes from https://github.com/sjy1203/GAMENet

def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    #     pro_pd = pro_pd[pro_pd['SEQ_NUM']<5]
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     pro_pd['ICD9_CODE'] = pro_pd['ICD9_CODE'].map(icd9_tree)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd = filter_first24hour_med(med_pd)
    #     med_pd = med_pd.drop(columns=['STARTDATE'])

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)


def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)


def process_ehr():
    # get med and diag (visit>=2)
    med_pd = process_med()
    med_pd = ndc2atc4(med_pd)
    #     med_pd = filter_300_most_med(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()
    #     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))

    patient_records = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_df = data[data['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([item for item in row['NDC']])  # medications
            admission.append([item for item in row['ICD9_CODE']])  # diagnoses
            admission.append([item for item in row['PRO_CODE']])  # procedures
            patient.append(admission)
        patient_records.append(patient)

    dill.dump(patient_records, open(PATIENT_RECORDS_FILE, 'wb'))


# ======================

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


def map_concepts2id():
    concept2id_prescriptions = Concept2Id()
    concept2id_diagnoses = Concept2Id()
    concept2id_procedures = Concept2Id()

    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))
    for patient in patient_records:
        for adm in patient:
            medications, diagnoses, procedures = adm[0], adm[1], adm[2]
            concept2id_prescriptions.add_concepts(medications)
            concept2id_diagnoses.add_concepts(diagnoses)
            concept2id_procedures.add_concepts(procedures)

    dill.dump({'concept2id_prescriptions': concept2id_prescriptions, 'concept2id_diagnoses': concept2id_diagnoses,
               'concept2id_procedures': concept2id_procedures}, open(CONCEPTID_FILE, 'wb'))


def build_ddi_matrix():
    topN = 100
    drug_ddi_df = pd.read_csv(drug_ddi_file)
    ddi_most_pd = drug_ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-topN:, :]
    fliter_ddi_df = drug_ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)

    concept2id_prescriptions = dill.load(open(CONCEPTID_FILE, 'rb')).get('concept2id_prescriptions')
    stitch2atc_dict = {}
    with open(drug_stitch2atc_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            stitch_id = line[0]
            atc_set = line[1:]
            stitch2atc_dict[stitch_id] = atc_set

    prescriptions_size = concept2id_prescriptions.get_concept_count()
    ddi_matrix = np.zeros((prescriptions_size, prescriptions_size))
    for index, row in ddi_df.iterrows():
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

    dill.dump({'ddi_matrix': ddi_matrix}, open(DDI_MATRIX_FILE, 'wb'))


def build_patient_records():
    ddi_matrix = dill.load(open(DDI_MATRIX_FILE, 'rb'))['ddi_matrix']

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

    concept2id_object = dill.load(open(CONCEPTID_FILE, 'rb'))
    concept2id_prescriptions = concept2id_object.get('concept2id_prescriptions')
    concept2id_diagnoses = concept2id_object.get('concept2id_diagnoses')
    concept2id_procedures = concept2id_object.get('concept2id_procedures')

    patient_records_idx = []
    patient_records = dill.load(open(PATIENT_RECORDS_FILE, 'rb'))
    for patient in patient_records:
        current_patient = []
        for adm in patient:
            medications, diagnoses, procedures = adm[0], adm[1], adm[2]
            admission = []
            admission.append([concept2id_prescriptions.concept2id.get(item) for item in medications])
            admission.append([concept2id_diagnoses.concept2id.get(item) for item in diagnoses])
            admission.append([concept2id_procedures.concept2id.get(item) for item in procedures])
            ddi_rate = get_ddi_rate(admission[0])
            admission.append([ddi_rate])
            current_patient.append(admission)
        patient_records_idx.append(current_patient)

    dill.dump({'patient_records': patient_records_idx}, open(PATIENT_RECORDS_FINAL_FILE, 'wb'))


def data_sampling():
    ddi_rate_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    patient_records_split_by_ddi_rate = {}
    for ddi_rate in ddi_rate_bins:
        patient_records_split_by_ddi_rate[ddi_rate] = []

    patient_records = dill.load(open(PATIENT_RECORDS_FINAL_FILE, 'rb'))['patient_records']
    for patient in patient_records:
        for idx, admission in enumerate(patient):
            ddi_rate = admission[3][0]
            current_patient_record = patient[:idx + 1]
            patient_records_split_by_ddi_rate[math.ceil(ddi_rate * 10.0) / 10].append(current_patient_record)

    train, test, validation = {}, {}, {}
    for ddi_rate, patients in patient_records_split_by_ddi_rate.items():
        train_patients, test_patients = train_test_split(patients, test_size=0.1)
        train_patients, validation_patients = train_test_split(train_patients, test_size=0.1)
        train[ddi_rate], test[ddi_rate], validation[ddi_rate] = train_patients, test_patients, validation_patients
    dill.dump({'train': train, 'test': test, 'validation': validation}, open(PATIENT_RECORDS_FILE_SEPARATE, 'wb'))

    print('patient records information stored separately by ddi rate')
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
    dill.dump({'train': train, 'test': test, 'validation': validation}, open(PATIENT_RECORDS_FILE_ACCUMULATE, 'wb'))

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


def build_co_occurrence_matrix():
    patient_ddi_rate = 1.0
    concept2id_object = dill.load(open(CONCEPTID_FILE, 'rb'))
    concept2id_medication = concept2id_object.get('concept2id_prescriptions')
    medication_count = concept2id_medication.get_concept_count()
    matrix = np.zeros((medication_count, medication_count))
    patient_records = dill.load(open(PATIENT_RECORDS_FILE_ACCUMULATE, 'rb'))['train'][1.0]
    count = 0
    for patient in patient_records:
        for admission in patient:
            if admission[-1][0] <= patient_ddi_rate:
                medications = admission[0]
                for med_i, med_j in combinations(medications, 2):
                    count += 1
                    matrix[med_i][med_j] = 1
                    matrix[med_j][med_i] = 1
    dill.dump(matrix, open(EHR_MATRIX_FILE, 'wb'))

    unique, counts = np.unique(matrix, return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == '__main__':
    process_ehr()
    map_concepts2id()
    build_ddi_matrix()
    build_patient_records()
    data_sampling()
    build_co_occurrence_matrix()
