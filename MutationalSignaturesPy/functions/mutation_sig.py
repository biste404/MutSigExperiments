import MutationalSignaturesPy.config as c

import os, sys, csv, glob, tqdm
from os import path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel, LdaMulticore, EnsembleLda
from gensim.corpora import Dictionary
from gensim.test.utils import datapath

#****** Gridsearch *******#

def gridsearch(input_df, model_name, results_dir, write_log_file, passes, iterations, num_topics_range, alpha_range, beta_range, random_seed):

    current_model_dir, log_dir = create_gridsearch_dirs(
        model_name, 
        results_dir,
        random_seed)

    if write_log_file:
        log_filename = log_dir+'/'+model_name+'_'+str(random_seed)
        print("See logs at: {}".format(log_filename+'.log'))
        sys.stdout = open(log_filename+'.log', 'w')

    print("Start time: {}".format(datetime.now()))
    print("Model_name: ", model_name)
    print("Number of records to train: ", len(input_df))
    print("Model directory: "+current_model_dir+'/')

    corpus, dictionary, tokens = generate_dictionary_corpus_tokens(input_df)

    print("\n****************************************** Baseline LDA ******************************************")

    print("Baseline model parameters: Passes = {}, Iterations = {}, Num topics = {}".format(
        c.BASELINE_PASSES, c.BASELINE_ITERATIONS, c.BASELINE_NUM_TOPICS))

    # Train LDA on baseline parameters to calcuate baseline metric
    baseline_metric = train_lda_and_get_eval_metric(
        corpus, dictionary, tokens,
        iterations=c.BASELINE_ITERATIONS,
        num_topics=c.BASELINE_NUM_TOPICS,
        passes=c.BASELINE_PASSES,
        random_seed=random_seed)

    print("\n****************************************** GRID SEARCH - NUM TOPICS ******************************************")

    print("Grid search parameters: Passes = {}, Iterations = {}, Num topics = {}, Metric = {}".format(
        passes, iterations, num_topics_range, c.SELECTED_EVAL_METRIC_GRIDSEARCH_K))

    # Train LDAs and evaluate different metrics by varying number of topics
    eval_metrics_for_k = grid_search_num_topics(
        corpus=corpus,
        dictionary=dictionary,
        tokens=tokens,
        num_topics_range=num_topics_range,
        iterations=iterations, 
        passes=passes, 
        random_seed=random_seed)


    eval_metrics_for_k.to_csv(
        current_model_dir+'/'+c.GRIDSEARCH_NUM_TOPICS_DIR+'/'+model_name+'__'+c.GRIDSEARCH_NUM_TOPICS_DIR+'.csv')

    print(pd.DataFrame(eval_metrics_for_k))

    # Get number of topics (k) at which the selected evaluation metric is best
    # Cosine similarity -> Lower is better
    # Coherence -> Higher is better
    tuned_model_params = identify_best_record(
        df=eval_metrics_for_k, 
        column=c.SELECTED_EVAL_METRIC_GRIDSEARCH_K)

    tuned_num_topics = int(tuned_model_params['Num Topics'])

    print("Best {} at Num Topics = {}".format(
        c.SELECTED_EVAL_METRIC_GRIDSEARCH_K, tuned_num_topics))

    plot_eval_metric_vs_n_topics(
        df=eval_metrics_for_k,
        column=c.SELECTED_EVAL_METRIC_GRIDSEARCH_K,
        path=current_model_dir+'/'+model_name,
        num_topics_range=num_topics_range)

    for column in eval_metrics_for_k.columns[1:-1]:

        plot_eval_metric_vs_n_topics(df=eval_metrics_for_k,
                                     path=current_model_dir+'/'+c.GRIDSEARCH_NUM_TOPICS_DIR+'/'+model_name,
                                     column=column,num_topics_range=num_topics_range)

    print("\n****************************************** GRID SEARCH - ALPHA, BETA ******************************************")

    print("Grid search parameters: Passes = {}, Iterations = {}, Num topics = {}, Metric = {}".format(
        passes, iterations, tuned_num_topics, c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB))

    # Train LDA by varying alpha, beta and optionally number of topics

    topics_range = range(tuned_num_topics, tuned_num_topics + 1, 1)

    grid_results_ab = grid_search_ab(
        corpus, 
        dictionary, 
        model_name, 
        topics_range,
        alpha_range, 
        beta_range, 
        iterations=iterations, 
        passes=passes, 
        random_seed=random_seed)

    grid_results_ab.to_csv(current_model_dir+'/' +
                           model_name+'__gridsearch_ab.csv', index=False)

    print(grid_results_ab)

    tuned_model_params = identify_best_record(
        df=grid_results_ab, 
        column=c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB)

    #tuned_model_params = grid_results_ab.iloc[grid_results_ab[c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB].idxmin()]

    # Train LDA with tuned parameters
    tuned_num_topics = int(tuned_model_params['topics'])
    tuned_alpha = round(float(tuned_model_params['alpha']), 2)
    tuned_beta = round(float(tuned_model_params['beta']), 2)
    tuned_metric = float(tuned_model_params[c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB])

    print("Best {} at Alpha = {}, Tuned Beta = {}".format(
        c.SELECTED_EVAL_METRIC_GRIDSEARCH_K, tuned_alpha, tuned_beta))

    print("\n****************************************** TUNED HYPERPARAMETERS ******************************************")

    print("Tuned model parameters: Num topics = {}, Alpha = {}, Beta = {}".format(
        tuned_num_topics, tuned_alpha, tuned_beta))

    print("Finish time: {}".format(datetime.now()))

    result_list = [model_name, tuned_num_topics, tuned_alpha, tuned_beta, random_seed, iterations,
                   passes, num_topics_range, len(input_df), c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB, tuned_metric]

    with open(results_dir+'/grid_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)
        f.close()

    #print(result_list)
    if write_log_file:
        sys.stdout.close()
        sys.stdout = c.DEFAULT_STDOUT

    return result_list


def create_gridsearch_dirs(model_name, results_dir, random_seed):

    gridsearch_dir = results_dir+'/'+c.GRIDSEARCH_DIR
    model_name_dir = gridsearch_dir+'/'+model_name
    random_seed_dir = model_name_dir+'/'+str(random_seed)
    gridsearch_num_topics_dir = random_seed_dir+'/'+c.GRIDSEARCH_NUM_TOPICS_DIR
    log_dir = random_seed_dir+'/'+c.LOG_DIR

    if path.exists(results_dir) == False:
        os.mkdir(results_dir)

    if path.exists(gridsearch_dir) == False:
        os.mkdir(gridsearch_dir)

    if path.exists(model_name_dir) == False:
        os.mkdir(model_name_dir)

    if path.exists(random_seed_dir) == False:
        os.mkdir(random_seed_dir)

    if path.exists(gridsearch_num_topics_dir) == False:
        os.mkdir(gridsearch_num_topics_dir)

    if path.exists(log_dir) == False:
        os.mkdir(log_dir)

    return random_seed_dir, log_dir


def generate_dictionary_corpus_tokens(filtered_df):

    # Split doc before feed into Dictionary
    tokens = [doc.replace("'", '').replace(",", '').split()
              for doc in filtered_df['text']]

    dictionary = Dictionary(tokens)
    # print(dictionary.token2id)

    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    return corpus, dictionary, tokens


def train_lda_and_get_eval_metric(corpus, dictionary, tokens, iterations, num_topics, passes, random_seed):

    num_words = len(dictionary.token2id)

    lda_model = LdaMulticore(
        corpus=corpus, 
        id2word=dictionary, 
        iterations=iterations, 
        random_state=random_seed,
        num_topics=num_topics, 
        workers=c.WORKERS, 
        chunksize=c.CHUNK_SIZE, 
        passes=passes)

    lda_signature = get_signatures_from_model(
        lda_model, num_words, c.BASELINE_NUM_TOPICS)

    median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound = calculate_eval_metrics(
        lda_signature, lda_model, corpus, dictionary, tokens)

    print('Variational Lower Bound: ', variational_lower_bound)
    print('Mean Cosine Similarity: ', mean_cosine_similarity)
    print('Median Cosine Similarity: ', median_cosine_similarity)

    if c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Median Cosine Similarity':
        eval_metric = median_cosine_similarity
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Mean Cosine Similarity':
        eval_metric = mean_cosine_similarity
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Coherence cv':
        eval_metric = coherence_cv
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Coherence umass':
        eval_metric = coherence_umass
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Perplexity':
        eval_metric = perplexity
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Variational Lower Bound':
        eval_metric = variational_lower_bound
    else:
        print("Evaluation metric is not defined..")
        eval_metric = 0

    return eval_metric


def grid_search_num_topics(corpus, dictionary, tokens, num_topics_range, iterations, passes, random_seed):

    results = []

    num_words = len(dictionary.token2id)

    for num_topics in num_topics_range:
        print("num_topics: ", num_topics)
        start_time = datetime.now()
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=iterations, random_state=random_seed,
                                 num_topics=num_topics,  chunksize=c.CHUNK_SIZE, workers=c.WORKERS, passes=passes)

        lda_signature = get_signatures_from_model(
            lda_model, num_words, num_topics)

        end_time = datetime.now()
        time_taken = (end_time-start_time).total_seconds()

        median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound = calculate_eval_metrics(
            lda_signature, lda_model, corpus, dictionary, tokens)

        print("Variational Lower Bound: {}".format(variational_lower_bound))
        print("Time taken: {}".format(time_taken))

        tup = num_topics, median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound, time_taken

        results.append(tup)

    results = pd.DataFrame(results, columns=['Num Topics', 'Mean Cosine Similarity', 'Median Cosine Similarity',
                           'Coherence cv', 'Coherence umass', 'Perplexity', 'Variational Lower Bound', 'Time(s)'])

    return results


def identify_best_record(df, column):

    if column in ['Mean Cosine Similarity', 'Median Cosine Similarity']:
        best_record = df.iloc[df[[column]].idxmin()]

    elif column in ['Variational Lower Bound', 'Perplexity', 'Coherence cv', 'Coherence umass']:
        best_record = df.iloc[df[[column]].idxmax()]

    return best_record


def plot_eval_metric_vs_n_topics(df, column, path, num_topics_range):

    tuned_model_params = identify_best_record(df=df, column=column)

    best_num_topics = int(tuned_model_params['Num Topics'])

    # Plot cosine similarity vs number of topics
    plt.plot(num_topics_range, df[column])
    plt.xlabel("Num Topics")
    plt.ylabel(column)
    plt.title("{} for different num topics\nBest value at k = {}".format(
        column, best_num_topics))
    plt.xticks(num_topics_range)
    plt.savefig(path+'__gridsearch_num_topics__'+column+'.png')
    plt.close()


def grid_search_ab(
        corpus, 
        dictionary, 
        model_name, 
        topics_range, 
        alpha_range, 
        beta_range, 
        iterations, 
        passes, 
        random_seed):

    grid_results = {
        'topics': [],
        'alpha': [],
        'beta': [],
        'Variational Lower Bound': [],
        'Mean Cosine Similarity': [],
        'Median Cosine Similarity': []
    }

    num_runs = len(topics_range) * len(alpha_range) * len(beta_range)

    progress_bar = tqdm.tqdm(total=num_runs)

    # iterate through number of topics
    for k in topics_range:
        # iterate through alpha values
        for a in alpha_range:
            # iterare through beta values
            for b in beta_range:
                # get the cosine similarity score for the given parameters
                ev, mean_cs, median_cs, variational_lower_bound = compute_eval_metric_ab(
                    corpus=corpus, 
                    dictionary=dictionary,
                    k=k, 
                    a=a, 
                    b=b, 
                    iterations=iterations, 
                    passes=passes, 
                    random_seed=random_seed)

                # Save the model results
                grid_results['topics'].append(k)
                grid_results['alpha'].append(a)
                grid_results['beta'].append(b)
                grid_results['Mean Cosine Similarity'].append(mean_cs)
                grid_results['Median Cosine Similarity'].append(median_cs)
                grid_results['Variational Lower Bound'].append(variational_lower_bound)
                #grid_results[c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB].append(ev)
                grid_results_df = pd.DataFrame(grid_results)

                progress_bar.update(1)

    progress_bar.close()

    return grid_results_df


def compute_eval_metric_ab(corpus, dictionary, k, a, b, iterations, passes, random_seed):

    num_words = len(dictionary.token2id)

    lda_model = LdaMulticore(corpus=corpus, 
        id2word=dictionary, 
        iterations=iterations, 
        random_state=random_seed,
        num_topics=k,  
        chunksize=c.CHUNK_SIZE, 
        workers=c.WORKERS, 
        passes=passes, 
        alpha=a, 
        eta=b)
    lda_signature = get_signatures_from_model(lda_model, num_words, k)

    median_cosine_similarity = calculate_median_cosine_similarity(lda_signature)

    mean_cosine_similarity = calculate_mean_cosine_similarity(lda_signature)

    variational_lower_bound = calculate_variational_lower_bound(
        lda_model, 
        corpus)

    if c.SELECTED_EVAL_METRIC_GRIDSEARCH_AB == 'Median Cosine Similarity':
        eval_metric = median_cosine_similarity
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Mean Cosine Similarity':
        eval_metric = mean_cosine_similarity
    elif c.SELECTED_EVAL_METRIC_GRIDSEARCH_K == 'Variational Lower Bound':
        eval_metric = variational_lower_bound
    else:
        print("Evaluation metric is not defined..")
        eval_metric = 0

    return eval_metric, mean_cosine_similarity, median_cosine_similarity, variational_lower_bound


def calculate_variational_lower_bound(lda_model, corpus):

    vlb = lda_model.bound(corpus=corpus)
    return vlb


def calculate_median_cosine_similarity(df):

    df = df[df != 1]
    median_cosine_similarity = pd.DataFrame(
        cosine_similarity(df.T)).median(axis=1).median()
    return round(median_cosine_similarity, c.ROUND_TO_DECIMAL_PLACE)


def calculate_mean_cosine_similarity(df):

    df = df[df != 1]
    #print(df)
    mean_cosine_similarity = pd.DataFrame(
        cosine_similarity(df.T)).mean(axis=1).mean()
    return round(mean_cosine_similarity, c.ROUND_TO_DECIMAL_PLACE)


def calculate_eval_metrics(lda_signature, lda_model, corpus, dictionary, tokens):

    median_cosine_similarity = 0
    mean_cosine_similarity = 0
    coherence_cv = 0
    coherence_umass = 0
    perplexity = 0
    variational_lower_bound = 0

    if 'Median Cosine Similarity' in c.EVAL_METRICS_GRIDSEARCH_K:
        median_cosine_similarity = calculate_median_cosine_similarity(
            lda_signature)
        #print('median_cosine_similarity: ', median_cosine_similarity)

    if 'Mean Cosine Similarity' in c.EVAL_METRICS_GRIDSEARCH_K:
        mean_cosine_similarity = calculate_mean_cosine_similarity(
            lda_signature)
        #print('mean_cosine_similarity: ', mean_cosine_similarity)

    if 'Coherence cv' in c.EVAL_METRICS_GRIDSEARCH_K:
        coherence_cv = CoherenceModel(
            model=lda_model, corpus=corpus, dictionary=dictionary, texts=tokens, coherence='c_v').get_coherence()
        #print('coherence_c_v: ', coherence_cv)

    if 'Coherence umass' in c.EVAL_METRICS_GRIDSEARCH_K:
        coherence_umass = CoherenceModel(
            model=lda_model, corpus=corpus, coherence='u_mass').get_coherence()
        #print('coherence_umass: ', coherence_umass)

    if 'Perplexity' in c.EVAL_METRICS_GRIDSEARCH_K:
        perplexity = lda_model.log_perplexity(corpus)
        #print('Perplexity: ', perplexity)

    if 'Variational Lower Bound' in c.EVAL_METRICS_GRIDSEARCH_K:
        variational_lower_bound = calculate_variational_lower_bound(
            lda_model, corpus)
        #print('variational_lower_bound: ', variational_lower_bound)

    return median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound


#****** Ensemble train *******#

def ensemble(
        input_df, 
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
        random_seed):

    # Default parameter values
    tuned_num_topics = num_topics
    tuned_alpha = alpha
    tuned_beta = beta

    if tuned_params_from_gridsearch:
        # Get the best parameter values from grid results
        grid = pd.read_csv(results_dir+'/grid_results.csv', header=None)

        grid.columns = c.GRID_RESULTS_COLUMNS

        if len(grid[grid.model_name == model_name]) != 0:
            grid_best = grid.iloc[grid[grid.model_name ==
                                    model_name]['value'].idxmax()]

            tuned_num_topics = grid_best['num_topics']
            tuned_alpha = grid_best['alpha']
            tuned_beta = grid_best['beta']

    current_model_dir, log_dir = create_ensemble_dirs(model_name, results_dir, random_seed)

    if write_log_file:
        sys.stdout = sys.__stdout__
        log_filename = log_dir+'/'+model_name+'_'+str(random_seed)
        print("See logs at: {}".format(log_filename+'.log'))
        sys.stdout = open(log_filename+'.log', 'w')

    print("Start time: {}".format(datetime.now()))
    print("Model_name: ", model_name)
    print("Number of records to train: ", len(input_df))
    print("Model directory: "+current_model_dir+'/')

    print("\n****************************************** ENSEMBLE TRAINING ******************************************")

    print("Tuned model parameters: Num topics = {}, Alpha = {}, Beta = {}".format(
        tuned_num_topics, tuned_alpha, tuned_beta))

    print("Num models = {} Passes = {}, Iterations = {}".format(
        num_models, passes, iterations))

    corpus, dictionary, tokens = generate_dictionary_corpus_tokens(input_df)

    # import logging
    # logging.basicConfig(
    #     format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if load_elda_model_from_disk:

        try:
            # Load a potentially pretrained model from disk.
            elda = EnsembleLda.load(elda_model_file_path)
            print('Pretrained EnsembleLDA model exists at path: {}'.format(elda_model_file_path))
            print("Loaded pretrained model from disk.")
        except Exception:
            print('EnsembleLDA file does not exists at path: {}'.format(elda_model_file_path))

    else:
    # Train ensemble lda
        print("Training EnsembleLDA (may take some time depending on the parameters): ")
        elda = EnsembleLda(
            corpus=corpus,
            id2word=dictionary,
            num_topics=tuned_num_topics,
            passes=passes,
            iterations=iterations,
            num_models=num_models,
            alpha=tuned_alpha,
            eta=tuned_beta,
            topic_model_class=LdaMulticore,
            ensemble_workers=4,
            distance_workers=4
        )

        print("\n****************************************** SAVE TO DISK ******************************************")

        # Saving model
        elda_model_path = datapath(current_model_dir+'/'+c.TUNED_MODEL_DIR+'/'+model_name)
        print(current_model_dir)
        print(elda_model_file_path)
        print(elda_model_path)
        print("Training complete. Saving trained model locally at: {}".format(elda_model_path))
        elda.save(elda_model_path)

    print("Ensemble num topics: {}".format(len(elda.get_topics())))

    # shape = elda.asymmetric_distance_matrix.shape
    # without_diagonal = elda.asymmetric_distance_matrix[~np.eye(
    #     shape[0], dtype=bool)].reshape(shape[0], -1)
    # print("eps in range(min, mean, max): ")
    # print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())

    # print("Recluster with eps = {}".format(without_diagonal.mean()))
    # elda.recluster(eps=without_diagonal.mean(), min_samples=2, min_cores=2)
    # print("Recluster num topics: {}".format(len(elda.get_topics())))
    # print("Topic term distribution array length: {}".format(len(elda.ttda)))

    ensemble_model = elda.generate_gensim_representation()
    # print(type(elda))
    # print(type(ensemble_model))



    lda_signature = get_signatures_from_model(
        ensemble_model, len(dictionary), len(ensemble_model.get_topics()))

    lda_signature = pd.DataFrame(lda_signature).rename_axis(
        'mutation_type').reset_index()

    # print("lda_signature:")
    # print(lda_signature)

    topic_probabilities = get_topic_probabilities_from_model(
        ensemble_model, corpus)

    lda_signature = calculate_signature_support(lda_signature, topic_probabilities)

    print("LDA Signatures (with support): ")
    print(lda_signature)

    lda_signature_path = current_model_dir + '/'+model_name + \
        '__ensemble_signatures'+'_'+str(random_seed)+'.csv'
    lda_signature.to_csv(lda_signature_path, index=False)

    print("topic_probabilities: \n", topic_probabilities)
    topic_probabilities_path = current_model_dir + '/'+model_name + \
        '__document_signature_probabilities'+'_'+str(random_seed)+'.csv'
    topic_probabilities.to_csv(topic_probabilities_path, index=False)

    # Saving topic probabilities plot
    topic_probabilities_plot = topic_probabilities.head(10)
    topic_probabilities_plot.plot(x="document", kind="bar", stacked=True, xlabel='Document',
                                    ylabel='Percentage of signature',
                                    title='Probabilities of signatures in documents'
                                  )
    topic_probabilities_plot_path = current_model_dir+'/'+model_name + \
        '__document_signature_probabilities'+'_'+str(random_seed)+'.png'
    
    plt.savefig(topic_probabilities_plot_path)
    plt.close()

    if model_name != 'alphamatrix__covid':

        sig = lda_signature.iloc[:-2,:]
        sim = compare_with_cosmic(sig, current_model_dir)


    sig_viz = signature_to_viz(lda_signature)
    sig_viz_path = current_model_dir+'/sig_viz.csv'
    sig_viz.to_csv(sig_viz_path)
    print(sig_viz_path)

    plot_signatures(sig_viz, current_model_dir)


    print(topic_probabilities_path)
    print(lda_signature_path)
    print("Finish time: {}".format(datetime.now()))

    if write_log_file:
        sys.stdout.close()
        sys.stdout = c.DEFAULT_STDOUT

def plot_signatures(df, dir):    
    # Create an array with the colors you want to use
    colors = ["#06bceb", "#0a0a0a", "#e32d26", "#9f9f9f", "#9ece5c", "#ecc6c5"]
    # Set your custom color palette
    customPalette = sns.set_palette(sns.color_palette(colors))

    g = sns.catplot(data=df, x='mutation_type', y='percentage', hue = 'substitution', row='signature',
                kind='bar',  palette=customPalette, aspect=4, dodge=False)

    (g.set_axis_labels("Mutation type", "Percentage of Single Base Substitutions")
    .set_xticklabels( 
        rotation=90, 
        horizontalalignment='center',
        fontweight='light',
        fontsize=7))
    # .despine(left=True))  

    signatures_path = dir+'/signatures.png'
    g.figure.savefig(signatures_path)

def create_ensemble_dirs(model_name, results_dir, random_seed):

    ensemble_dir = results_dir+'/'+c.ENSEMBLE_DIR
    model_name_dir = ensemble_dir+'/'+model_name
    random_seed_dir = model_name_dir+'/'+str(random_seed)
    tuned_model_dir = random_seed_dir+'/'+c.TUNED_MODEL_DIR
    log_dir = random_seed_dir+'/'+c.LOG_DIR

    if path.exists(results_dir) == False:
        os.mkdir(results_dir)

    if path.exists(ensemble_dir) == False:
        os.mkdir(ensemble_dir)

    if path.exists(model_name_dir) == False:
        os.mkdir(model_name_dir)

    if path.exists(random_seed_dir) == False:
        os.mkdir(random_seed_dir)

    if path.exists(tuned_model_dir) == False:
        os.mkdir(tuned_model_dir)

    if path.exists(log_dir) == False:
        os.mkdir(log_dir)

    return random_seed_dir, log_dir


def get_signatures_from_model(lda_model, num_words, num_topics):

    topics = lda_model.show_topics(
        num_words=num_words, num_topics=num_topics, formatted=False)

    # print('topics: ',topics)

    lda_signature = pd.DataFrame(t[1][i]
                                 for t in topics for i in range(num_words))

    # print(lda_signature)

    t = [n for n in range(num_words) for n in range(num_topics)]
    #print(len(sorted(t)))
    lda_signature['topic'] = sorted(t)

    lda_signature = lda_signature.rename(
        columns={lda_signature.columns[0]: "type", lda_signature.columns[1]: "value"})
    # lda_model[corpus][0]

    lda_signature_pivot = lda_signature.pivot(
        index='type', columns='topic')['value']

    #print(lda_signature_pivot)

    return lda_signature_pivot

    # lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    # pyLDAvis.display(lda_display)


def get_topic_probabilities_from_model(lda_model, corpus):
    topic_probabilities = [
        lda_model.get_document_topics(item) for item in corpus]

    topic_probabilities_df = pd.DataFrame(
        topic_probabilities).reset_index().melt(id_vars='index')

    topic_probabilities_df = topic_probabilities_df[~topic_probabilities_df['value'].isnull(
    )]

    #topic_probabilities_df['topic'], topic_probabilities_df['probability'] = topic_probabilities_df.value.str
    topic_probabilities_df['topic'], topic_probabilities_df['probability'] = topic_probabilities_df.value.str[0], topic_probabilities_df.value.str[1]

    topic_probabilities_df = topic_probabilities_df[['index', 'topic', 'probability']].pivot(
        index='index', columns='topic', values='probability')

    topic_probabilities_df.reset_index(inplace=True)
    topic_probabilities_df = topic_probabilities_df.rename(
        columns={'index': 'document'})
    return topic_probabilities_df




#****** Tuned train *******#

def tuned_train(
        input_df, 
        model_name, 
        results_dir,
        write_log_file,
        passes,
        iterations,
        num_topics,
        alpha,
        beta,
        tuned_params_from_gridsearch,
        random_seed):

    # Default parameter values
    tuned_num_topics = num_topics
    tuned_alpha = alpha
    tuned_beta = beta

    if tuned_params_from_gridsearch:

        # Get the best parameter values from grid results
        grid = pd.read_csv(results_dir+'/grid_results.csv', header=None)

        grid.columns = c.GRID_RESULTS_COLUMNS

        if len(grid[grid.model_name == model_name]) != 0:
            grid_best = grid.iloc[grid[grid.model_name ==
                                    model_name]['value'].idxmax()]

            tuned_num_topics = grid_best['num_topics']
            tuned_alpha = grid_best['alpha']
            tuned_beta = grid_best['beta']

    current_model_dir, log_dir = create_tuned_train_dirs(
        model_name, results_dir, random_seed)

    if write_log_file:
        #sys.stdout = sys.__stdout__
        log_filename = log_dir+'/'+model_name+'_'+str(random_seed)
        print("See logs at: {}".format(log_filename+'.log'))
        sys.stdout = open(log_filename+'.log', 'w')

    print("Start time: {}".format(datetime.now()))
    print("Model_name: ", model_name)
    print("Number of records to train: ", len(input_df))
    print("Model directory: "+current_model_dir+'/')

    print("\n****************************************** LDA TRAINING ******************************************")

    print("Tuned model parameters: Num topics = {}, Alpha = {}, Beta = {}".format(
        tuned_num_topics, tuned_alpha, tuned_beta))

    print("Passes = {}, Iterations = {}".format(
        passes, iterations))

    corpus, dictionary, tokens = generate_dictionary_corpus_tokens(input_df)

    lda_model, topic_probabilities, corpus, dictionary, lda_signature, mean_cosine_similarity, median_cosine_similarity, variational_lower_bound = train_lda_tuned(
        corpus=corpus,
        dictionary=dictionary,
        tokens=tokens,
        num_topics=tuned_num_topics,
        iterations=iterations,
        passes=passes,
        random_seed=random_seed,
        alpha=tuned_alpha,
        beta=tuned_beta)

    print("\n****************************************** SAVE TO DISK ******************************************")

    # Saving model
    lda_model.save(datapath(current_model_dir+'/' +
                   c.TUNED_MODEL_DIR+'/'+model_name))

    #lda_signature = lda_signature.append([topic_probabilities.count(axis = 0)], ignore_index = True)
    #lda_signature = pd.concat([ lda_signature, pd.DataFrame([topic_probabilities.count(axis = 0)]) ], ignore_index=True, axis=0)

    # Saving signatures
    #lda_signature = rename_topics(df = lda_signature, column_pos = 1, suffix = model_name.split('__')[1])
    lda_signature_path = current_model_dir + '/'+model_name + \
        '__signatures'+'_'+str(random_seed)+'.csv'
    lda_signature.to_csv(lda_signature_path, index=False)

    # Saving topic probabilities
    #topic_probabilities = rename_topics(df = topic_probabilities, column_pos = 1, suffix = model_name.split('__')[1])
    topic_probabilities_path = current_model_dir + '/'+model_name + \
        '__document_signature_probabilities'+'_'+str(random_seed)+'.csv'
    topic_probabilities.to_csv(topic_probabilities_path, index=False)

    # Saving topic probabilities plot
    topic_probabilities_plot = topic_probabilities.head(10)
    topic_probabilities_plot.plot(x="document", kind="bar", stacked=True, xlabel='Document',
                                    ylabel='Percentage of signature',
                                    title='Probabilities of signatures in documents'
                                  )
    topic_probabilities_plot_path = current_model_dir+'/'+model_name + \
        '__document_signature_probabilities'+'_'+str(random_seed)+'.png'
    plt.savefig(topic_probabilities_plot_path)
    plt.close()

    if model_name != 'alphamatrix__covid':

        sig = lda_signature.iloc[:-2,:]
        
        sim = compare_with_cosmic(sig, current_model_dir)

    sig_viz = signature_to_viz(lda_signature)
    sig_viz_path = current_model_dir+'/sig_viz.csv'
    sig_viz.to_csv(sig_viz_path)
    print(sig_viz_path)

    plot_signatures(sig_viz, current_model_dir)

    print(topic_probabilities_path)
    print(lda_signature_path)
    print(topic_probabilities_plot_path)
    print("Finish time: {}".format(datetime.now()))

    if write_log_file:
        sys.stdout.close()
        sys.stdout = c.DEFAULT_STDOUT


def create_tuned_train_dirs(model_name, results_dir, random_seed):

    tuned_train_dir = results_dir+'/'+c.TUNED_TRAIN_DIR
    model_name_dir = tuned_train_dir+'/'+model_name
    random_seed_dir = model_name_dir+'/'+str(random_seed)
    tuned_model_dir = random_seed_dir+'/'+c.TUNED_MODEL_DIR
    log_dir = random_seed_dir+'/'+c.LOG_DIR

    if path.exists(results_dir) == False:
        os.mkdir(results_dir)

    if path.exists(tuned_train_dir) == False:
        os.mkdir(tuned_train_dir)

    if path.exists(model_name_dir) == False:
        os.mkdir(model_name_dir)

    if path.exists(random_seed_dir) == False:
        os.mkdir(random_seed_dir)

    if path.exists(tuned_model_dir) == False:
        os.mkdir(tuned_model_dir)

    if path.exists(log_dir) == False:
        os.mkdir(log_dir)

    return random_seed_dir, log_dir


def train_lda_tuned(corpus, dictionary, tokens, num_topics, iterations, passes, random_seed, alpha='symmetric', beta='auto'):

    num_words = len(dictionary.token2id)

    lda_model = LdaMulticore(
        corpus=corpus, 
        id2word=dictionary, 
        iterations=iterations, 
        random_state=random_seed,
        num_topics=num_topics, 
        workers=c.WORKERS, 
        chunksize=c.CHUNK_SIZE,  
        passes=passes, 
        alpha=alpha, 
        eta=beta)

    topic_probabilities_df = get_topic_probabilities_from_model(
        lda_model, corpus)

    lda_signature = get_signatures_from_model(lda_model, num_words, num_topics)
    lda_signature_df = pd.DataFrame(lda_signature).rename_axis(
        'mutation_type').reset_index()

    lda_signature_df = calculate_signature_support(lda_signature_df, topic_probabilities_df)

    print("LDA Signatures (with support): ")
    print(lda_signature_df)

    mean_cosine_similarity = calculate_mean_cosine_similarity(lda_signature)
    median_cosine_similarity = calculate_median_cosine_similarity(lda_signature)
    variational_lower_bound = calculate_variational_lower_bound(lda_model, corpus)
    print('Mean Cosine Similarity: ', mean_cosine_similarity)
    print('Median Cosine Similarity: ', median_cosine_similarity)
    print("Variational Lower Bound: ", variational_lower_bound)

    # coherence_c_v = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, texts=tokens, coherence='c_v').get_coherence()
    # coherence_u_mass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence()
    # print('coherence_c_v: ', coherence_c_v)
    # print('coherence_u_mass: ', coherence_u_mass)

    return lda_model, topic_probabilities_df, corpus, dictionary, lda_signature_df, mean_cosine_similarity, median_cosine_similarity, variational_lower_bound

def calculate_signature_support(lda_signature, topic_probabilities):

    doc_length = len(topic_probabilities)

    signature_support_presence = pd.DataFrame([topic_probabilities.count(axis=0)/doc_length]).drop('document', axis=1)
    signature_support_probability = pd.DataFrame([topic_probabilities.sum(axis=0)/doc_length]).drop('document', axis=1)

    signature_support_presence['mutation_type'] = c.SUPPORT_TYPE_1_PRESENCE
    signature_support_probability['mutation_type'] = c.SUPPORT_TYPE_2_PROBABILITY

    lda_signature = pd.concat([lda_signature, signature_support_presence, signature_support_probability], ignore_index=True, axis=0)

    return lda_signature


#****** Combine signatures *******#

def combine_signatures(signature_path, model_name, write_log_file, cosine_sim_greater_than):

    current_model_dir, log_dir = create_combine_signatures_dirs(
        signature_path)

    if write_log_file:
        #sys.stdout = sys.__stdout__
        log_filename = log_dir+'/'+model_name
        print("See logs at: {}".format(log_filename+'.log'))
        sys.stdout = open(log_filename+'.log', 'w')

    print("Start time: {}".format(datetime.now()))
    print("Model_name: ", model_name)
    print("Model directory: "+current_model_dir+'/')

    print("\n****************************************** Combine signatures ******************************************")

    signatures = {}

    for file in glob.glob(signature_path+'*signatures*.csv'):
        print(file)
        signatures[file[:-4].split('/')[-1]] = pd.read_csv(file,sep=",", header=0).set_index('mutation_type')


    if list(signatures.keys()) == []:

        for file in glob.glob(signature_path+'*/*signatures*.csv'):
            print(file)
            signatures[file[:-4].split('/')[-1]] = pd.read_csv(file,sep=",", header=0).set_index('mutation_type')

    combined_signature = pd.concat(list(signatures.values()), axis=1, join="inner")

    combined_signature_path = current_model_dir+'/combined_sig.csv'
    combined_signature.to_csv(combined_signature_path)
    print(combined_signature_path)

    df = pd.read_csv(current_model_dir+'/combined_sig.csv').set_index('mutation_type')

    print("Extracting similar signatures using cosine similarity: ")
    for i in range(c.COMBINE_SIGNATURES_MAX_ITERATIONS):
        prev_count_df = df.mean().mean()
        df, cos_sim = extract_similar_signatures(df, cosine_sim_greater_than)
        current_count_df = df.mean().mean()
        # print(current_count_df)
        # print(prev_count_df)
        if current_count_df == prev_count_df:
            #print("inside break")
            break
        #print(df)

    print("Pairwise Cosine Similarity of extracted signature: ")
    print(cos_sim)
    print("Mean Cosine Similarity of extracted signature: ")
    #print(calculate_mean_cosine_similarity(cos_sim))
    print(cos_sim.mean().mean())

    df.rename(index={c.SUPPORT_TYPE_1_PRESENCE:c.SUPPORT_TYPE_1_PRESENCE+'_sum'},inplace=True)
    df.rename(index={c.SUPPORT_TYPE_2_PROBABILITY:c.SUPPORT_TYPE_2_PROBABILITY+'_sum'},inplace=True)

    extracted_signature_path = current_model_dir+'/extracted_sig__bootstrap_by_cossim.csv'

    print("Extracted signatures: ")
    print(df)
    df.to_csv(extracted_signature_path)
    print(extracted_signature_path)

    df = pd.read_csv(current_model_dir+'/extracted_sig__bootstrap_by_cossim.csv')

    sig_viz = signature_to_viz(df)
    sig_viz_path = current_model_dir+'/sig_viz.csv'
    sig_viz.to_csv(sig_viz_path)
    print(sig_viz_path)

    plot_signatures(sig_viz, current_model_dir)

    extracted_sig = pd.read_csv(current_model_dir+'/extracted_sig__bootstrap_by_cossim.csv',sep=",", header=0)

    #print(extracted_sig)

    if model_name != 'alphamatrix__covid':

        sig = extracted_sig.iloc[:-2,:]
        
        sim = compare_with_cosmic(sig, current_model_dir)

        
        
    print("Finish time: {}".format(datetime.now()))

    if write_log_file:
        sys.stdout.close()
        sys.stdout = c.DEFAULT_STDOUT

    return combined_signature

def compare_with_cosmic(sig, current_model_dir):

    cosmic = pd.read_csv(os.getcwd()+'/data_cancer/cosmic_signature.csv',sep=",", header=0)

    sim = 0

    # print(len(sig))
    # print(len(cosmic))

    if (len(sig)) == len(cosmic):

        #print("inside len")

        sig = sig.set_index('mutation_type')

        cosmic['Type'] = cosmic['Type'].str.replace('>', '')
        cosmic = cosmic.set_index('Type')

        # print(sig)
        # print(cosmic)

        sim = pd.DataFrame(cosine_similarity(sig.T, cosmic.T))


        # print(sim)

        sim.columns=cosmic.columns
        sim.index=sig.columns

        m = sim
        sim = sim.where(m> c.COSINE_SIM_TO_COMPARE_COSMIC).dropna(how = 'all', axis = 1)

        if(isinstance(sim, int) or sim.empty):
            print("Unable to calculate similarity with COSMIC Signatures.")

        else:
            #print("inside else")
            sim_to_cosmic_path = current_model_dir+'/similarity_to_cosmic.csv'
            sim.to_csv(sim_to_cosmic_path)
            print(sim_to_cosmic_path)

        print("Mean cosine similarity with matching COSMIC signatures: ")
        print(sim.mean().mean())
        
    return sim

def create_combine_signatures_dirs(signature_path):
    combined_sig_dir = signature_path+c.COMBINED_SIGNATURE_DIR
    log_dir = combined_sig_dir+'/'+c.LOG_DIR

    if path.exists(combined_sig_dir) == False:
        os.mkdir(combined_sig_dir)

    if path.exists(log_dir) == False:
        os.mkdir(log_dir)

    return combined_sig_dir, log_dir


def extract_similar_signatures(df, cosine_sim_greater_than):

    # Remove support row and get signatures
    sig = df.iloc[:-2, :]
    # Store support row
    support_presence = df.iloc[-2:-1, :]
    support_probability = df.iloc[-1:, :]

    # Calculate pairwise cosine similarity of signatures
    cos_sim = pd.DataFrame(cosine_similarity(sig.T))

    # print("cos_sim")
    # print(cos_sim)

    sim_list = []

    # For each column of cosine similarity matrix
    for i in range(len(cos_sim)):
        # When cosine similarity greater than given value, store the idexes of signatures into a list
        ith_list = cos_sim.index[~cos_sim[cos_sim > cosine_sim_greater_than].isna()[i]]
        sim_list.append(ith_list)

    # Multiple columns could be having same list of signatures 
    # So remove duplicates
    sim_list = [list(x) for x in set(tuple(x) for x in sim_list)]
    # print("sim_list")
    print(sim_list)

    extracted_signatures = []
    signature_support_presence = []
    signature_support_probability = []

    # For each item in similarity list (list of signatures with cosine similarity > a given value)
    for item in sim_list:
        # print(item)
        # print(sig.iloc[:,item])
        # Extract signature as mean of all signatures for each mutation type
        extracted_signatures.append(sig.iloc[:,item].mean(axis=1))
        # Calculate support as the sum of supports
        signature_support_presence.append(support_presence.iloc[:,item].sum(axis=1))
        signature_support_probability.append(support_probability.iloc[:,item].sum(axis=1))

    # print("extracted_signatures")
    # print(extracted_signatures)
    extracted_signatures_df = pd.DataFrame(extracted_signatures).T
    signature_support_presence_df = pd.DataFrame(signature_support_presence).T
    signature_support_probability_df = pd.DataFrame(signature_support_probability).T

    # print(extracted_signatures_df)
    # print(signature_support_presence_df)
    # print(signature_support_probability_df)
    # #signature_support_presence_df.index = sim_list
    # Join signature and support

    frames = [extracted_signatures_df, signature_support_presence_df, signature_support_probability_df]

    extracted_signatures_df = pd.concat(frames)
    #print(extracted_signatures_df)

    return extracted_signatures_df, cos_sim


def signature_to_viz(df):

    # Remove two support rows
    df = df.iloc[:-2,:]

    df = pd.melt(df, id_vars=['mutation_type'], var_name = 'signature')

    #df['substitution'] = df['mutation_type'].str[2]+df['mutation_type'].str[3]
    df['substitution'] = df['mutation_type'].str[2:5]

    #print('substitution')
    #print(df)

    df['trinucleotide'] = df['mutation_type'].str[0]+df['mutation_type'].str[2]+df['mutation_type'].str[6]

    #print('trinucleotide')

    #print(df)

    # df['mutation_type'] = df['mutation_type'].str[2]+'['+df['mutation_type'].str[0]+df['mutation_type'].str[1]+']'+df['mutation_type'].str[4] 

    #print('mutation_type')
    #print(df)
    df['percentage'] = round(df['value']*100,2)

    df = df.sort_values(by = ['substitution', 'trinucleotide'])

    return df


#****** Misc *******#


def create_gridsearch_iterations_dirs(model_name, results_dir, random_seed):

    gridsearch_dir = results_dir+'/'+c.GRIDSEARCH_ITERATIONS_DIR
    model_name_dir = gridsearch_dir+'/'+model_name
    random_seed_dir = model_name_dir+'/'+str(random_seed)
    gridsearch_num_topics_dir = random_seed_dir+'/'+c.GRIDSEARCH_ITERATIONS_DIR
    log_dir = random_seed_dir+'/'+c.LOG_DIR

    if path.exists(results_dir) == False:
        os.mkdir(results_dir)

    if path.exists(gridsearch_dir) == False:
        os.mkdir(gridsearch_dir)

    if path.exists(model_name_dir) == False:
        os.mkdir(model_name_dir)

    if path.exists(random_seed_dir) == False:
        os.mkdir(random_seed_dir)

    if path.exists(gridsearch_num_topics_dir) == False:
        os.mkdir(gridsearch_num_topics_dir)

    if path.exists(log_dir) == False:
        os.mkdir(log_dir)

    return random_seed_dir, log_dir


def rename_topics(df, column_pos, suffix):
    for col in df.columns[column_pos:]:
        newname = suffix+'_'+str(col)
        df = df.rename(columns={col: newname})
    return df


def get_groupby_counts(df, column):
    result_df = pd.DataFrame(df.groupby([column])[column].count())
    result_df = result_df.rename(columns={'cancer_type': 'count'})
    result_df = result_df.reset_index(drop=False)
    result_df = result_df.sort_values(by=['count'], ascending=False)
    return result_df



def gridsearch_iterations(
        input_df, 
        model_name, 
        results_dir, 
        write_log_file,
        iterations_range,
        num_topics,
        random_seed):

    current_model_dir, log_dir = create_gridsearch_iterations_dirs(model_name, results_dir, random_seed)

    if write_log_file:
        log_filename = log_dir+'/'+model_name+'_'+str(random_seed)
        print("See logs at: {}".format(log_filename+'.log'))
        sys.stdout = open(log_filename+'.log', 'w')

    print("Start time: {}".format(datetime.now()))
    print("Model_name: ", model_name)
    print("Number of records to train: ", len(input_df))
    print("Model directory: "+current_model_dir+'/')

    corpus, dictionary, tokens = generate_dictionary_corpus_tokens(input_df)

    print("\n****************************************** GRID SEARCH - ITERATIONS ******************************************")

    print("Grid search parameters: Passes = {}, Iterations = {}, Num topics = {}, Metric = {}".format(
        iterations_range, iterations_range, num_topics, c.SELECTED_EVAL_METRIC_GRIDSEARCH_K))

    # Train LDAs and evaluate different metrics by varying number of topics
    eval_metrics_for_iterations = gridsearch_iterations_train(
        corpus=corpus,
        dictionary=dictionary,
        tokens=tokens,
        iterations_range=iterations_range, 
        num_topics=num_topics,
        current_model_dir=current_model_dir, 
        model_name=model_name,
        random_seed=random_seed)

    print(eval_metrics_for_iterations)

    #print(eval_metrics_for_iterations[c.SELECTED_EVAL_METRIC_GRIDSEARCH_K])

    plot_eval_metric_vs_iterations(
        df = eval_metrics_for_iterations, 
        column = c.SELECTED_EVAL_METRIC_GRIDSEARCH_K,
        path = current_model_dir+'/'+model_name,
        iterations_range=iterations_range)

    tuned_model_params = identify_best_record(df = eval_metrics_for_iterations, column = c.SELECTED_EVAL_METRIC_GRIDSEARCH_K)

    tuned_iterations = int(tuned_model_params['Iterations'])
    
    print("Best {} at Iteration = {}".format(c.SELECTED_EVAL_METRIC_GRIDSEARCH_K, tuned_iterations))

    print("Finish time: {}".format(datetime.now()))

    if write_log_file:
        sys.stdout.close()
        sys.stdout = c.DEFAULT_STDOUT

def plot_eval_metric_vs_iterations(df, column, path, iterations_range):

    tuned_model_params = identify_best_record(df = df, column = column)

    best_iterations = int(tuned_model_params['Iterations'])
    
    # Plot cosine similarity vs number of topics
    plt.plot(iterations_range, df[column])
    plt.xlabel("Iterations")
    plt.ylabel(column)
    plt.title("{} for different iterations\nBest value at iterations = {}".format(column, best_iterations))
    plt.xticks(iterations_range)
    plt.savefig(path+'__gridsearch_iterations__'+column+'.png')
    plt.close()


def gridsearch_iterations_train(corpus, dictionary, tokens, iterations_range, num_topics, current_model_dir, model_name, random_seed):

    results = []

    num_words = len(dictionary.token2id)

    print("Iterations range: {}".format(iterations_range))

    for iteration in iterations_range:
        #this_iteration = 2**iteration
        this_iteration = iteration
        print("iterations: ",this_iteration)
        start_time = datetime.now()
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=this_iteration, random_state=random_seed,
                                 num_topics=num_topics,  chunksize=c.CHUNK_SIZE, workers=c.WORKERS, passes=this_iteration)

        lda_signature = get_signatures_from_model(
            lda_model, num_words, num_topics)
        
        #print(lda_signature)

        median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound = calculate_eval_metrics(
            lda_signature, lda_model, corpus, dictionary, tokens)

        end_time = datetime.now()

        time_taken = (end_time-start_time).total_seconds()

        print("Variational Lower Bound: {}".format(variational_lower_bound))
        print("Time taken: {}".format(time_taken))

        tup = num_topics, this_iteration, median_cosine_similarity, mean_cosine_similarity, coherence_cv, coherence_umass, perplexity, variational_lower_bound, time_taken

        results.append(tup)

        results_df = pd.DataFrame(results, columns=['Num Topics', 'Iterations', 'Mean Cosine Similarity', 'Median Cosine Similarity', 'Coherence cv', 'Coherence umass', 'Perplexity', 'Variational Lower Bound', 'Time(s)'])

        results_df.to_csv(
            current_model_dir+'/'+c.GRIDSEARCH_ITERATIONS_DIR+'/'+model_name+'__'+c.GRIDSEARCH_ITERATIONS_DIR+'.csv')

    return results_df