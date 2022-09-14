import MutationalSignaturesPy.config as c
from MutationalSignaturesPy.functions.cancer_functions import *
from MutationalSignaturesPy.functions.mutation_sig import *

import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')

'''
Information on input data available at https://cancer.sanger.ac.uk/signatures/sbs/
COSMIC v3.0 data used in this project is available in https://www.synapse.org/#!Synapse:syn11726616
Related research papaer by Alexandrov, L.B. et al., 2020 available at https://www.nature.com/articles/s41586-020-1943-3
Reference signature available to download at: https://cancer.sanger.ac.uk/signatures/downloads/
'''

def run_cancer_download():

    # Download mutation catalog/s
    download_cosmic_mutation_data(c.SYNAPSE_ENTITY_LIST, c.SYNAPSE_USERID, c.SYNAPSE_PW, c.MUTATION_GROUP)

    # Download cosmic signatures as reference
    download_reference_cosmic_signatures()


def run_cancer_preprocess():

    '''
    WES - Exomes
    WGS - Whole Genomes
    PCAWG - Pan-Cancer Analysis of Whole Genomes
    TCGA - The Cancer Genome Atlas

    Input data - Catalogs of mutation spectra (counts of each type of mutation in each tumour) from four sources
    '''

    print("Loading..")
    data_dir = c.DATA_DIR_CANCER
    combined_data_name = c.COMBINED_CANCER_DATA_NAME 

    # Load input data from csv
    WES_Other_Mutation_Catalog = pd.read_csv(data_dir+'/WES_Other.96.csv', sep=",", header=0)
    WES_TCGA_Mutation_Catalog = pd.read_csv(data_dir+'/WES_TCGA.96.csv', sep=",", header=0)
    WGS_Other_Mutation_Catalog = pd.read_csv(data_dir+'/WGS_Other.96.csv', sep=",", header=0)
    WGS_PCAWG_Mutation_Catalog = pd.read_csv(data_dir+'/WGS_PCAWG.96.csv', sep=",", header=0)

    print("Preprocessing..")

    # Call preprocess_mutation_catalog function
    WES_Other_Preprocessed = preprocess_mutation_catalog(WES_Other_Mutation_Catalog, 'wes_other')
    WES_TCGA_Preprocessed = preprocess_mutation_catalog(WES_TCGA_Mutation_Catalog, 'wes_tcga')
    WGS_Other_Preprocessed = preprocess_mutation_catalog(WGS_Other_Mutation_Catalog, 'wgs_other')
    WGS_PCAWG_Preprocessed = preprocess_mutation_catalog(WGS_PCAWG_Mutation_Catalog, 'wgs_pcawg')

    print("Combining..")

    # Combined preprocessed data from all four sources
    combined_data = pd.concat([WES_Other_Preprocessed, WES_TCGA_Preprocessed, WGS_Other_Preprocessed, WGS_PCAWG_Preprocessed])

    print("Storing in CSV..")

    # tokenize.tokenize_combined_data_spacy(combined_data)
    # print("Tokenized..")

    # Store data combined from all four sources in a csv file locally
    combined_data_path = data_dir+'/'+combined_data_name+'.csv'
    combined_data.to_csv (combined_data_path, index = False, header=True)

    print("Stored at: {}".format(combined_data_path))


# Data for visualisation of input data
def run_cancer_preprocess_to_viz():

    '''
    WES - Exomes
    WGS - Whole Genomes
    PCAWG - Pan-Cancer Analysis of Whole Genomes
    TCGA - The Cancer Genome Atlas

    Input data - Catalogs of mutation spectra (counts of each type of mutation in each tumour) from four sources
    '''

    print("Loading..")
    data_dir = c.DATA_DIR_CANCER
    combined_data_name = c.COMBINED_CANCER_DATA_NAME+'_to_viz'

    # Load input data from csv
    WES_Other_Mutation_Catalog = pd.read_csv(data_dir+'/WES_Other.96.csv', sep=",", header=0)
    WES_TCGA_Mutation_Catalog = pd.read_csv(data_dir+'/WES_TCGA.96.csv', sep=",", header=0)
    WGS_Other_Mutation_Catalog = pd.read_csv(data_dir+'/WGS_Other.96.csv', sep=",", header=0)
    WGS_PCAWG_Mutation_Catalog = pd.read_csv(data_dir+'/WGS_PCAWG.96.csv', sep=",", header=0)

    print("Preprocessing..")

    # Call preprocess_mutation_catalog function
    WES_Other_Preprocessed = preprocess_mutation_catalog_to_viz(WES_Other_Mutation_Catalog, 'wes_other')
    WES_TCGA_Preprocessed = preprocess_mutation_catalog_to_viz(WES_TCGA_Mutation_Catalog, 'wes_tcga')
    WGS_Other_Preprocessed = preprocess_mutation_catalog_to_viz(WGS_Other_Mutation_Catalog, 'wgs_other')
    WGS_PCAWG_Preprocessed = preprocess_mutation_catalog_to_viz(WGS_PCAWG_Mutation_Catalog, 'wgs_pcawg')

    print("Combining..")

    # Combined preprocessed data from all four sources
    combined_data = pd.concat([WES_Other_Preprocessed, WES_TCGA_Preprocessed, WGS_Other_Preprocessed, WGS_PCAWG_Preprocessed])

    print("Storing in CSV..")

    # tokenize.tokenize_combined_data_spacy(combined_data)
    # print("Tokenized..")

    # Store data combined from all four sources in a csv file locally
    combined_data_path = data_dir+'/'+combined_data_name+'.csv'
    combined_data.to_csv (combined_data_path, index = False, header=True)

    print("Completed..") 


def run_cancer_gridsearch(
        sources = c.CANCER_DATA_SOURCE, 
        cancer_types = c.CANCER_TYPE, 
        ignore_mutations_less_than = c.MIN_MUTATION_COUNT,
        write_log_file = c.ENABLE_LOG,
        passes = c.GRIDSEARCH_PASSES, 
        iterations = c.GRIDSEARCH_ITERATIONS, 
        num_topics_range = c.GRIDSEARCH_NUM_TOPICS_RANGE,
        alpha_range = c.GRIDSEARCH_ALPHA_RANGE,
        beta_range = c.GRIDSEARCH_BETA_RANGE,
        random_seed = c.RANDOM_SEED,
        results_dir = c.RESULTS_DIR_CANCER):

    # Load combined data generated by run_preprocess
    combined_data = load_cancer_data()

    # Gridsearch 
    for cancer_type in cancer_types:

        run_gridsearch_cancer_type(
            combined_data,
            sources, 
            cancer_type, 
            ignore_mutations_less_than, 
            write_log_file,
            passes, 
            iterations, 
            num_topics_range, 
            alpha_range, 
            beta_range, 
            random_seed,
            results_dir)


def run_gridsearch_cancer_type(
        combined_data, 
        sources, 
        cancer_type, 
        ignore_mutations_less_than, 
        write_log_file,
        passes, 
        iterations, 
        num_topics_range, 
        alpha_range, 
        beta_range, 
        random_seed,
        results_dir):

    #Recommended format: source <doubleunderscore> data(Eg. source__data)
    model_name = '_'.join(sources)+'__'+cancer_type
    # model_name = 'custom__name'

    # Filter input data for each cancer type
    filtered_df = filter_cancer_data(combined_data, sources, cancer_type, ignore_mutations_less_than)

    # Gridsearch and get tuned hyperparameters(k, a, b).
    gridsearch_result = gridsearch(
        filtered_df, 
        model_name, 
        results_dir, 
        write_log_file, 
        passes, 
        iterations, 
        num_topics_range, 
        alpha_range, 
        beta_range, 
        random_seed)


def run_cancer_ensemble(        
        sources = c.CANCER_DATA_SOURCE, 
        cancer_types = c.CANCER_TYPE, 
        ignore_mutations_less_than = c.MIN_MUTATION_COUNT,
        write_log_file = c.ENABLE_LOG,
        passes = c.ENSEMBLE_PASSES, 
        iterations = c.ENSEMBLE_ITERATIONS, 
        num_models = c.ENSEMBLE_NUM_MODELS,
        num_topics = c.TUNED_NUM_TOPICS,
        alpha = c.TUNED_ALPHA,
        beta = c.TUNED_BETA,
        tuned_params_from_gridsearch = c.TUNED_PARAMS_FROM_GRIDSEARCH,
        load_elda_model_from_disk = c.LOAD_ELDA_MODEL_FROM_DISK,
        elda_model_file_path = '',
        random_seed = c.RANDOM_SEED,
        results_dir = c.RESULTS_DIR_CANCER):

    # Load combined data generated by run_preprocess
    combined_data = load_cancer_data()
    
    # Ensemble with tuned parameters
    for cancer_type in cancer_types:
        run_ensemble_cancer_type(
            combined_data,
            sources, 
            cancer_type, 
            ignore_mutations_less_than, 
            write_log_file,
            passes, 
            iterations, 
            num_models,
            num_topics,
            alpha,
            beta,
            tuned_params_from_gridsearch,
            load_elda_model_from_disk,
            elda_model_file_path,
            random_seed,
            results_dir)


def run_ensemble_cancer_type(
        combined_data,
        sources, 
        cancer_type, 
        ignore_mutations_less_than, 
        write_log_file,
        passes, 
        iterations, 
        num_models,
        num_topics,
        alpha,
        beta,
        tuned_params_from_gridsearch,
        load_elda_model_from_disk,
        elda_model_file_path,
        random_seed,
        results_dir):

    #Recommended format: source <doubleunderscore> data(Eg. source__data)
    model_name = '_'.join(sources)+'__'+cancer_type
    # model_name = 'custom__name'

    # Filter input data for each cancer type
    filtered_df = filter_cancer_data(combined_data, sources, cancer_type, ignore_mutations_less_than)

    # Ensemble models
    ensemble(
        filtered_df, 
        model_name, 
        results_dir, 
        write_log_file,
        passes, 
        iterations, 
        num_models,
        num_topics,
        alpha,
        beta,
        tuned_params_from_gridsearch,
        load_elda_model_from_disk,
        elda_model_file_path,
        random_seed)


def run_cancer_tuned_train(
    sources = c.CANCER_DATA_SOURCE, 
    cancer_types = c.CANCER_TYPE, 
    ignore_mutations_less_than = c.MIN_MUTATION_COUNT,
    write_log_file = c.ENABLE_LOG,
    passes = c.TUNED_PASSES, 
    iterations = c.TUNED_ITERATIONS, 
    num_trainings = c.TUNED_TRANINGS,
    num_topics = c.TUNED_NUM_TOPICS,
    alpha = c.TUNED_ALPHA,
    beta = c.TUNED_BETA,
    tuned_params_from_gridsearch = c.TUNED_PARAMS_FROM_GRIDSEARCH,
    random_seed = c.RANDOM_SEED,
    results_dir = c.RESULTS_DIR_CANCER):

    # Load combined data generated by run_preprocess
    combined_data = load_cancer_data()

    for cancer_type in cancer_types:
        run_tuned_train_cancer_type(
            combined_data, 
            sources, 
            cancer_type,
            ignore_mutations_less_than,
            write_log_file,
            passes,
            iterations,
            num_topics,
            alpha,
            beta,
            tuned_params_from_gridsearch,
            random_seed,
            results_dir)
    
    # Training with tuned parameters
    for i in range(num_trainings-1):
        random_seed = random.randint(0,10000)
        for cancer_type in cancer_types:
            run_tuned_train_cancer_type(
                combined_data, 
                sources, 
                cancer_type,
                ignore_mutations_less_than,
                write_log_file,
                passes,
                iterations,
                num_topics,
                alpha,
                beta,
                tuned_params_from_gridsearch,
                random_seed,
                results_dir)


def run_tuned_train_cancer_type(
    combined_data, 
    sources, 
    cancer_type,
    ignore_mutations_less_than,
    write_log_file,
    passes,
    iterations,
    num_topics,
    alpha,
    beta,
    tuned_params_from_gridsearch,
    random_seed,
    results_dir):

    #Recommended format: source <doubleunderscore> data(Eg. source__data)
    model_name = '_'.join(sources)+'__'+cancer_type
    # model_name = 'custom__name'

    # Filter input data for each cancer type
    filtered_df = filter_cancer_data(combined_data, sources, cancer_type, ignore_mutations_less_than)

    # Apply tuned hyperparameters to get tuned model
    tuned_train(
        filtered_df, 
        model_name, 
        results_dir, 
        write_log_file,
        passes,
        iterations,
        num_topics,
        alpha,
        beta,
        tuned_params_from_gridsearch,
        random_seed)


def run_cancer_combine_signatures(    
    sources = c.CANCER_DATA_SOURCE, 
    cancer_types = c.CANCER_TYPE, 
    write_log_file = c.ENABLE_LOG,
    cosine_sim_greater_than = c.COMBINE_SIGNATURES_COSINE_SIMILARITY,
    results_dir = c.RESULTS_DIR_CANCER
    ):
    
    # Ensemble with tuned parameters
    for cancer_type in cancer_types:
        run_combine_signatures_cancer_type(sources, cancer_type, write_log_file, cosine_sim_greater_than, results_dir)


def run_combine_signatures_cancer_type(sources, cancer_type, write_log_file, cosine_sim_greater_than, results_dir):

    #Recommended format: source <doubleunderscore> data(Eg. source__type)
    model_name = '_'.join(sources)+'__'+cancer_type
    # model_name = 'custom__name'

    signature_path = results_dir+'/'+c.TUNED_TRAIN_DIR+'/'+model_name+'/'

    # Combine signatures using cosine similarity
    combined_signature = combine_signatures(
        signature_path, 
        model_name, 
        write_log_file, 
        cosine_sim_greater_than, 
        )


def run_gridsearch_iterations_cancer(
    sources = c.CANCER_DATA_SOURCE, 
    cancer_types = c.CANCER_TYPE, 
    ignore_mutations_less_than = c.MIN_MUTATION_COUNT,
    write_log_file = c.ENABLE_LOG,
    iterations_range = c.GRIDSEARCH_ITERATIONS_RANGE, 
    num_topics = c.TUNED_NUM_TOPICS,
    random_seed = c.RANDOM_SEED,
    results_dir = c.RESULTS_DIR_CANCER
    ):

    # Load combined data generated by run_preprocess
    combined_data = load_cancer_data()

    for cancer_type in cancer_types:
        run_gridsearch_iterations_cancer_type(
            combined_data, 
            sources, 
            cancer_type,
            ignore_mutations_less_than,
            write_log_file,
            iterations_range,
            num_topics,
            random_seed,
            results_dir)

def run_gridsearch_iterations_cancer_type(
    combined_data, 
    sources, 
    cancer_type,
    ignore_mutations_less_than,
    write_log_file,
    iterations_range,
    num_topics,
    random_seed,
    results_dir):

    #Recommended format: source <doubleunderscore> data(Eg. source__data)
    model_name = '_'.join(sources)+'__'+cancer_type
    # model_name = 'custom__name'

    # Filter input data for each cancer type
    filtered_df = filter_cancer_data(combined_data, sources, cancer_type, ignore_mutations_less_than)

    # Apply tuned hyperparameters to get tuned model
    gridsearch_iterations(
        filtered_df, 
        model_name, 
        results_dir,
        write_log_file,
        iterations_range,
        num_topics,
        random_seed)