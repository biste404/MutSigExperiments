from MutationalSignaturesPy.cancer_signatures import *

cwd = os.getcwd()

def run_download():

  run_cancer_download()

def run_preprocess():

  run_cancer_preprocess()

def run_gridsearch():

  run_cancer_gridsearch(
    sources = sources, 
    cancer_types = cancer_types,
    ignore_mutations_less_than = ignore_mutations_less_than,
    write_log_file = write_log_file,
    passes = passes,
    iterations = iterations,
    num_topics_range = num_topics_range,
    alpha_range = list(np.arange(0.01, 1, 0.3)),
    beta_range = list(np.arange(0.01, 1, 0.3)),
    random_seed = random_seed,
    results_dir = cwd+'/'+experiment_name
  )

def run_ensemble():

  run_cancer_ensemble(        
    sources = sources, 
    cancer_types = cancer_types,
    ignore_mutations_less_than = ignore_mutations_less_than,
    write_log_file = write_log_file,
    passes = passes, 
    iterations = iterations, 
    num_models = num_models,
    num_topics= k,
    alpha = a,
    beta = b,
    tuned_params_from_gridsearch = tuned_params_from_gridsearch,
    load_elda_model_from_disk = False,
    elda_model_file_path = '',
    random_seed = random_seed,
    results_dir = cwd+'/'+experiment_name
    )



def run_tuned_train():

  run_cancer_tuned_train(   
    sources = sources, 
    cancer_types = cancer_types,
    ignore_mutations_less_than = ignore_mutations_less_than,
    write_log_file = write_log_file,
    passes = passes, 
    iterations = iterations, 
    num_trainings = num_models,
    num_topics = k,
    alpha = a,
    beta = b,
    tuned_params_from_gridsearch = tuned_params_from_gridsearch,
    random_seed = random_seed,
    results_dir = cwd+'/'+experiment_name
    )

def run_combine_signatures():

  run_cancer_combine_signatures(    
    sources = sources, 
    cancer_types = cancer_types,
    write_log_file = write_log_file,
    cosine_sim_greater_than = 0.9,
    results_dir = cwd+'/'+experiment_name
    )


def run_gridsearch_iterations():

  run_gridsearch_iterations_cancer(
    sources = sources, 
    cancer_types = cancer_types,
    ignore_mutations_less_than = ignore_mutations_less_than,
    write_log_file = write_log_file,
    iterations_range = [2, 5, 10, 15, 20], 
    num_topics = k,
    random_seed = random_seed,
    results_dir = cwd+'/'+experiment_name
    )

if __name__ == "__main__":
    # execute only if run as a script

    # run_download()
    run_preprocess()

    sources = ['wgs', 'wes']
    cancer_types = ['Eso', 'Head-SCC', 'Kidney-RCC']
    iterations = 1 # 1 for a test run, greater than 100 for experiments
    passes = 1 # 1 for a test run, greater than 100 for experiments
    random_seed = 3474
    num_topics_range = range(2, 30, 2)
    num_models = 16
    ignore_mutations_less_than = 100
    write_log_file = True
    
    tuned_params_from_gridsearch = True
    
    # If tuned_params_from_gridsearch = False, then set hyperparameter values manually
    k = 14
    a = 0.31
    b = 0.31

    #All the results will be stored in a folder with this name
    experiment_name = 'Results_multiple_cancers_iterations1' 

    run_gridsearch()
    run_ensemble()
    run_tuned_train()
    run_combine_signatures()
    #run_gridsearch_iterations()