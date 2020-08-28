import torch


class Params:
    def __init__(self):
        self.PATIENT_RECORDS_FILE_ACCUMULATE = 'data/patient_records_accumulate_tail_top100'
        self.PATIENT_RECORDS_FILE_SEPARATE = 'data/patient_records_separate_tail_top100'
        self.DDI_MATRIX_FILE = 'data/ddi_matrix_tail_top100'
        self.CONCEPTID_FILE = 'data/concepts2id_mapping'
        self.EHR_MATRIX_FILE = 'data/ehr_matrix_1.0'
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.USE_CUDA else "cpu")
        self.MEDICATION_COUNT = 153
        self.DIAGNOSES_COUNT = 1960
        self.PROCEDURES_COUNT = 1432

        self.OPT_SPLIT_TAG_ADMISSION = -1
        self.OPT_SPLIT_TAG_VARIABLE = -2
        self.OPT_MODEL_MAX_EPOCH = 2
        self.OPT_PATIENT_DDI_RATE = 0.1
        self.OPT_METRIC_TYPE = 'f1'  # accuracy or f1
        self.OPT_DECODER_TYPE = 'KVGCNMultiEmb'

        self.PRETRAINED_EMBEDDING_MEDICATION = None
        self.PRETRAINED_EMBEDDING_DIAGNOSES = None
        self.PRETRAINED_EMBEDDING_PROCEDURES = None

        self.LOSS_PROPORTION_BCE = 0.9
        self.LOSS_PROPORTION_Multi_Margin = 0.1

        self.INPUT_SIZE = 200
        self.HIDDEN_SIZE = 200

        self.REAL_label = 1
        self.FAKE_label = 0

        self.fitted_distribution_label_num = 5
        self.fitted_distribution_file_name = 'data/fitted_data/real_data_6_False_None_ddi_rate_0.4_standard_False'
        self.fitted_distribution_file_name_train = 'data/fitted_data/real_data_6_False_None_ddi_rate_0.4_standard_False_train.npy'
        self.fitted_distribution_file_name_test = 'data/fitted_data/real_data_6_False_None_ddi_rate_0.4_standard_False_test.npy'

        self.ENCODER_OUTPUT_TRAIN_FILE_NAME = 'data/fitted_data/encoder_output_train_0.4_1.npy'
        self.ENCODER_OUTPUT_TEST_FILE_NAME = 'data/fitted_data/encoder_output_test_0.4_1.npy'


def print_params(self):
    print('current parameters:')
