import os, sys
import random
import numpy as np

ENABLE_LOG = False
DEFAULT_STDOUT = sys.stdout
cwd = os.getcwd()
#RANDOM_SEED = random.randint(0,10000)
RANDOM_SEED = 3475

'''
Filter data parameters examples:

    SOURCE:
        SOURCE = 'wgs'
        SOURCE = 'wes'
        SOURCE = 'wgs_wes'
        
    CANCER_TYPE: 
        CANCER_TYPE = ['Breast', 'Liver-HCC', 'Skin-Melanoma']
        input list of cancer types
        also support cancer subtypes 
            eg. enter 'Liver' to filter cancer type or 'Liver-HCC' for further filtering
'''
CANCER_DATA_SOURCE = ['wgs', 'wes']
CANCER_TYPE = ['Breast']

VIRUS_DATA_SOURCE = ['alphamatrix']
VIRUS_TYPE = ['covid']

ROUND_TO_DECIMAL_PLACE = 3
MIN_MUTATION_COUNT = 0 

GRIDSEARCH_DIR = 'gridsearch'
TUNED_TRAIN_DIR = 'tuned_train'
ENSEMBLE_DIR = 'ensemble_train'

LOG_DIR = 'log'
TUNED_MODEL_DIR = 'tuned_model'
GRIDSEARCH_NUM_TOPICS_DIR = 'gridsearch_num_topics'
GRIDSEARCH_ITERATIONS_DIR = 'gridsearch_iterations'
COMBINED_SIGNATURE_DIR = 'combined_signatures'

# ldamulticore params
CHUNK_SIZE = 2000
WORKERS = 4

# Baseline params
BASELINE_ITERATIONS = 2
BASELINE_PASSES = 2
BASELINE_NUM_TOPICS = 5

# Grid search params
GRIDSEARCH_ITERATIONS = 3
GRIDSEARCH_PASSES = 3

GRIDSEARCH_NUM_TOPICS_RANGE = range(2, 5, 1)
GRIDSEARCH_ITERATIONS_RANGE = [1, 2, 5, 10, 30, 50]

GRIDSEARCH_ALPHA_RANGE = list(np.arange(0.01, 1, 0.3))
GRIDSEARCH_BETA_RANGE = list(np.arange(0.01, 1, 0.3))

GRID_RESULTS_COLUMNS = ['model_name', 'num_topics', 'alpha', 'beta', 'random_seed', 'iterations', 'passes', 'num topics', 'doc_len', 'metric', 'value']

# Tuned train params
TUNED_ITERATIONS = 3
TUNED_PASSES = 3
TUNED_TRANINGS = 4
TUNED_NUM_TOPICS = 2
TUNED_ALPHA = 0.1
TUNED_BETA = 0.1
TUNED_PARAMS_FROM_GRIDSEARCH = True

# Ensemble train params
ENSEMBLE_ITERATIONS = 3
ENSEMBLE_PASSES = 3
ENSEMBLE_NUM_MODELS = 8
LOAD_ELDA_MODEL_FROM_DISK = False


COMBINE_SIGNATURES_COSINE_SIMILARITY = 0.9
COMBINE_SIGNATURES_MAX_ITERATIONS = 30

COSINE_SIM_TO_COMPARE_COSMIC = 0.8
# Evaluation metrics
'''
https://datascience.oneoffcoder.com/topic-modeling-gensim.html
'''
'''
Supported metrics(string):
    'Variational Lower Bound'
    'Perplexity'
    'Mean Cosine Similarity'
    'Median Cosine Similarity'
    'Coherence cv'
    'Coherence umass'

['Variational Lower Bound', 'Perplexity', 'Mean Cosine Similarity','Median Cosine Similarity', 'Coherence cv', 'Coherence umass']
'''
EVAL_METRICS_GRIDSEARCH_K = ['Mean Cosine Similarity','Median Cosine Similarity', 'Variational Lower Bound']
SELECTED_EVAL_METRIC_GRIDSEARCH_K = 'Variational Lower Bound'
EVAL_METRICS_GRIDSEARCH_AB = ['Mean Cosine Similarity', 'Median Cosine Similarity', 'Variational Lower Bound']
SELECTED_EVAL_METRIC_GRIDSEARCH_AB = 'Variational Lower Bound'


#******** Type: Cancer ********#

# Synapse credentials to download COSMIC mutations data
SYNAPSE_USERID = 'bipinsteephenmodoor@gmail.com'
SYNAPSE_PW = 'pdsynapsebip'

'''
COSMIC v3.0 data stored in https://www.synapse.org/#!Synapse:syn11726616

WES_Other - syn11726617
WES_TCGA - syn11726618
WGS_Other - syn11726619
WGS_PCAWG - syn11726620
'''
SYNAPSE_ENTITY_LIST = ['syn11726617', 'syn11726618', 'syn11726619', 'syn11726620']

'''
set
'*.96.csv' for single base substitutions in unstranded trinucleotide context
'*.192.csv' for stranded trinucleotide context
'*.1536.csv' for unstranded pentanucleotide context
'*.dinucs.csv' for doublet base substitutions
'*.indels.csv' for insertion and deletion mutations
'*.csv' for all files
Note: This project support only '*.96.csv'
'''
MUTATION_GROUP = '*.96.csv'

# Set the location to download data
DATA_DIR_CANCER = cwd+'/data_cancer'
RESULTS_DIR_CANCER = cwd+'/results_cancer'
COMBINED_CANCER_DATA_NAME = 'combined_preprocessed_cancer_data'
COMBINED_VIRUS_DATA_NAME = 'combined_preprocessed_virus_data'


#******** Type: Virus ********#

DATA_DIR_VIRUS = cwd+'/data_virus'
RESULTS_DIR_VIRUS = cwd+'/results_virus'

COVID_DATA_URL = "https://www.dropbox.com/s/bi2gi9ag61hoqbm/Alpha_Matrix.csv?dl=1"

SUPPORT_TYPE_1_PRESENCE = 'support_presence'
SUPPORT_TYPE_2_PROBABILITY = 'support_probability'