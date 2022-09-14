import MutationalSignaturesPy.config as c

import pandas as pd

import os
from os import path

def download_virus_mutation_data():

    data_dir = c.DATA_DIR_VIRUS

    # Create /data directory to store required files 
    if path.exists(data_dir) == False:
        os.makedirs(data_dir)

    print("Downloading mutation catalog.")

    # Download covid mutation data from from dropbox
    data_path = c.DATA_DIR_VIRUS+'/alphamatrix__covid.csv'
    pd.read_csv(c.COVID_DATA_URL, sep=",", header=0).to_csv(data_path, index=False)

    print('Stored at: {}'.format(data_path))

    # # Download flu mutation data from from dropbox
    # data_path = c.DATA_DIR_VIRUS+'/alphamatrix__flu.csv'
    # pd.read_csv(c.COVID_DATA_URL, sep=",", header=0).to_csv(data_path, index=False)

    # print('Stored at: {}'.format(data_path))

def split_cols(df, split_col, new_col_list, sep):

    for i in range(len(new_col_list)):
        df[new_col_list[i]] = df[split_col].str.split(sep).str[i]

    return df


def preprocess_virus_mutation_catalog(source, source_name):

    # Columns of mutation types(192) transformed into rows
    source = source.melt(id_vars=["sequence_name"],
                         var_name='mutation_type', value_name='value')

    source.value = source.value.astype(int)

    # Filter only records where count of mutation type is greater than 0
    source = source[source['value'] > 0]

    # Format mutation type
    # Eg. AC-AAG is transformed to A[A>C]G
    # Substitution type: A>C (Adenine to Cytosine)
    # Trinucleotide: AAG
    source['mutation_type'] = source['mutation_type'].str[3]+'['+source['mutation_type'].str[0]+'>'+source['mutation_type'].str[1]+']'+source['mutation_type'].str[5] 
    #print(source['mutation_type'])

    # Repeat mutation type number of times from input
    source['text'] = (source.mutation_type+' ') * source.value

    # Remove multiple spaces from text
    source.text = source.text.replace(r'\s+', ' ', regex=True)
    source['text'] = source['text'].str.replace('  ', " ")
    source['text'] = source['text'].str.replace('-', "")

    source['week'] = source['sequence_name'].str.split('|').str[8].astype(int)

    # source['virus_strain_name'] = source['sequence_name'].str.split('|').str[0]
    # source['accession_id'] = source['sequence_name'].str.split('|').str[1]
    # source['host'] = source['sequence_name'].str.split('|').str[2]
    # source['clade'] = source['sequence_name'].str.split('|').str[3]
    # source['lineage'] = source['sequence_name'].str.split('|').str[4]
    # source['sublineage'] = source['sequence_name'].str.split('|').str[5]
    # source['sequence_length'] = source['sequence_name'].str.split('|').str[6]
    # source['date'] = source['sequence_name'].str.split('|').str[7]
    # source['week'] = source['sequence_name'].str.split('|').str[8]
    # source['country'] = source['sequence_name'].str.split('|').str[9]
    # source['location'] = source['sequence_name'].str.split('|').str[10]

    # Group by sample id and join words with spaces between to construct the document
    source = source.groupby(['week'], as_index=False).agg({'text': ''.join})

    source['mutation_count'] = source['text'].str.count('\s+')

    # Store name of source in a column
    source['source'] = source_name

    # Return preprocessed source data
    return source


def load_virus_data():

    virus_data_path = c.DATA_DIR_VIRUS+'/'+c.COMBINED_VIRUS_DATA_NAME+'.csv'
    virus_data_preprocessed = pd.read_csv(
        virus_data_path, sep=",", header=0, lineterminator='\n')

    print(virus_data_preprocessed)

    return virus_data_preprocessed


def filter_virus_data(df, source, virus_type, min_mutation_count):

    query_str = "source == @virus_type & mutation_count >= @min_mutation_count"
    filtered_df = df.query(query_str).reset_index()

    return filtered_df