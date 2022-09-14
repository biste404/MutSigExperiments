import MutationalSignaturesPy.config as c

import shutil, glob, os
from os import path
from zipfile import ZipFile

import synapseclient 
import pandas as pd
#import spacy

# Get current working directory
cwd = os.getcwd()

# Set the location to download data
download_dir = cwd+r'/download'
unzip_dir = cwd+r'/unzip'
data_dir = c.DATA_DIR_CANCER

# Define function to download cosmic mutation data
def download_cosmic_mutation_data(synapse_entity_list, synapse_userid, synapse_password, mutation_group):

  # Delete /data directory if exists (comment if dont want to delete /data dir each time)
  # shutil.rmtree(data_dir, ignore_errors=True)

  # Create /data directory to store required files 
  if path.exists(data_dir) == False:
    os.makedirs(data_dir)

  # Delete not required directories
  shutil.rmtree(download_dir, ignore_errors=True)
  shutil.rmtree(unzip_dir, ignore_errors=True)

  # synapse connection
  syn = synapseclient.Synapse() 
  syn.login(synapse_userid,synapse_password)  
    
  # Iterating the synapse entity list
  for i in range(len(synapse_entity_list)):
      # print(synapse_entity_list[i])

      # Obtain a pointer and download the data 
      syn_pointer = syn.get(entity = synapse_entity_list[i], downloadLocation = download_dir )  

      # Get the path to the local copy of the data file 
      filepath = syn_pointer.path 
      # print(filepath)

      # Unzip mutation catalog files
      ZipFile(filepath).extractall(unzip_dir)

  # Move only required file/s to /data directory
  for file in glob.glob(unzip_dir+'/'+mutation_group):
      # print(file)
      shutil.copy(file, data_dir)

  # Delete not required directories
  shutil.rmtree(download_dir, ignore_errors=True)
  shutil.rmtree(unzip_dir, ignore_errors=True)

  # Display message on screen
  print(r"Downloaded files: ")
  # Display the downloaded file/s in the /data directory
  for file in glob.glob(data_dir+'/'+mutation_group):
      print(file)


def download_reference_cosmic_signatures():

  '''
  Available to download at: https://cancer.sanger.ac.uk/signatures/downloads/
    v3.0 - May 2019
      GRCh38 - SBS
  '''

  target_url = "https://cancer.sanger.ac.uk/signatures/documents/431/COSMIC_v3_SBS_GRCh38.txt"

  cosmic_signature = pd.read_csv(target_url, sep="\t", header=0)

  cosmic_signature_path = c.DATA_DIR_CANCER+'/cosmic_signature.csv'
  cosmic_signature.to_csv(cosmic_signature_path, index = False)

  print(cosmic_signature_path)


'''
Define function to preprocess each input from four sources
Count of each mutation type is there in input
Consider mutation type as word in document. There are 96 mutation types considered (if single base substitutions in unstranded trinucleotide context (*.96.csv)).
Consider sample id(tumour sample from a patient) as a document in a corpus
Just like documents containing words, many mutation types will be there in same sample id
From count of mutation types, document containing words as mutation types reconstructed
'''
# Preprocess mutation catalog and prepare documents
def preprocess_mutation_catalog(source, source_name):

  # Delete samples where total number of mutations is less than a value defined in constants
  #print(len([col for col, val in source_int.sum().iteritems() if val < 400]))
  # source.drop([col for col, val in source_int.sum().iteritems() if val < c.DROP_RECORD_WHEN_NUM_MUTATIONS_LESS_THAN], 
  #   axis=1, inplace=True)

  #print(source)

  # Rename column 'Mutation type' to 'Mut_type' 
  source = source.rename(columns={ source.columns[0]: "mut_type" })

  # Columns of sample ids transformed into rows
  source = source.melt(id_vars=["mut_type", "Trinucleotide"], var_name = 'sample_id', value_name = 'value')

  # Filter only records where count of mutation type is greater than 0
  source = source[source['value'] > 0]

  # Construct column 'Mutation_type' from other columns
  #source['mutation_type'] = source['Trinucleotide'].str[0] + '[' + source['mut_type'].str.replace('>', '') + ']' + source['Trinucleotide'].str[2]
  source['mutation_type'] = source['Trinucleotide'].str[0] + '[' + source['mut_type'] + ']' + source['Trinucleotide'].str[2]

  # Repeat mutation type number of times from input
  source['text'] = (source.mutation_type+' ') * source.value

  # Remove multiple spaces from text
  source.text = source.text.replace(r'\s+', ' ', regex=True)
  source['text'] = source['text'].str.replace('  '," ")

  # Group by sample id and join words with spaces between to construct the document
  source = source.groupby(['sample_id'], as_index = False).agg({'text': ''.join})
  #source = source[[source.columns[1]]]

  # Extract cancer type and sample id from sample id field
  source['cancer_subtype'] = source['sample_id'].str.split('::').str[0]
  source['cancer_type'] = source['cancer_subtype'].str.split('-').str[0]
  source['sample_id'] = source['sample_id'].str.split('::').str[1]
  source['mutation_count'] = source['text'].str.count('\s+')

  # Store name of source in a column
  source['source'] = source_name

  # Return preprocessed source data
  return source

# Preprocess for visalisation
def preprocess_mutation_catalog_to_viz(source, source_name):

  source['mutation_type'] = source['Trinucleotide'].str[0]+'['+source['Mutation type'].str.replace('>', '')+']'+ source['Trinucleotide'].str[2]
  source['substitution_type'] = source['Mutation type'].str.replace('>', '')
  source['trinucleotide'] = source['Trinucleotide']
  source = source.drop(columns = ['Mutation type', 'Trinucleotide'])

  source = source.melt(id_vars = ['mutation_type', 'substitution_type', 'trinucleotide'], var_name = 'sample_id', value_name = 'value')
  source = source[source['value'] > 0]

  source['cancer_subtype'] = source['sample_id'].str.split('::').str[0]
  source['cancer_type'] = source['cancer_subtype'].str.split('-').str[0]
  source['sample_id'] = source['sample_id'].str.split('::').str[1]

  source['source'] = source_name

  return source

# Tokenize using spacy. Not currently used
# def tokenize_combined_df_spacy(combined_df):

#   nlp = spacy.load("en_core_web_sm")
#   nlp.max_length = 20000000

#   tokens = []

#   for txt in nlp.pipe(combined_df['text'] , disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'] ):
#     this_token = [str(token) for token in txt ]
#     tokens.append(this_token)
  
#   combined_df["tokens"] = tokens

# Tokenize without using spacy. Not currently used
def tokenize_combined_df(combined_df):
    tokens = []

    for txt in combined_df['text']:
        this_token = [str(token) for token in txt ]
    tokens.append(this_token)

    combined_df["tokens"] = tokens


def load_cancer_data():

    combined_data_path = c.DATA_DIR_CANCER+'/'+c.COMBINED_CANCER_DATA_NAME+'.csv'

    combined_data = pd.read_csv(
        combined_data_path, sep=",", header=0, lineterminator='\n')

    print("Loaded "+combined_data_path)

    return combined_data


def filter_cancer_data(df, sources, cancer_type, min_mutation_count):

    source_list = []
    for source in sources:
      if source == 'wgs':
        source_list.append('wgs_pcawg')
        source_list.append('wgs_other')
      if source == 'wes':
        source_list.append('wes_tcga')
        source_list.append('wes_other')

    query_str = "source == @source_list & cancer_type ==  @cancer_type & mutation_count >= @min_mutation_count"
    filtered_df = df.query(query_str).reset_index()
    if(len(filtered_df)) == 0:
      query_str = "source == @source_list & cancer_subtype ==  @cancer_type & mutation_count >= @min_mutation_count"
    filtered_df = df.query(query_str).reset_index()
    return filtered_df

