# MutSigExperiments

Paper: 

Topic Modelling for identifying signatures of mutational processes in cancer and virus genomes using Latent Dirichlet Allocation
[Paper](Paper.pdf)

#TopicModelling #Python #NLP #Gensim #LDA #LatentDirichletAllocation #ExperimentFramework #GridSearch #Ensemble #Visualisations #GenomeMutations #MutationalSignatures #CancerMutations #CovidMutations

## Table of contents
* [Abstract](#abstract)
* [Setup](#setup)

## Abstract
The genomes inside somatic cells of human body are constantly exposed to different intrinsic and extrinsic mutagenic processes. Contributions from each of these mutagenic processes are different, but over the course of time they lead to increased variations in the genetic code and often leads to cancers. Analysis of different mutational signatures in genomes from different cancer samples allow one to examine how mutational processes such as aging, exposure to sunlight and smoking work. In this project, we have developed a novel framework to perform experiments to identify, quantify and evaluate common mutational processes and their activities using Latent Dirichlet Allocation Topic Modelling technique using cancer genomes datasets. Results shows that our method confirm many expected results, provide solutions to some signature extraction challenges, and provide an easy to use, scalable platform to conduct experiments with reproducible results.

Viruses inside hosts body are also subject to mutations. However, some mutagenic process such as aging are not relevant to virus genome mutations since duration of infection is comparatively shorter. Study of mutations in virus genomes inside human body could shed light to the signatures of attack and defence of human immune system. Promising performance of the developed framework evaluated with cancer genome data, together with the availability of publicly available genomes of different human viruses motivated the extraction and study of signatures of virus genome mutations in humans.

The study uses Latent Dirichlet Allocation, which is a probabilistic unsupervised machine learning method for Topic Modelling. Topic Modelling is often used to extract topics from documents in Natural Language Processing (NLP). In this project, Topic Modelling is employed to extract signatures of mutational processes from genome samples, which is a non-NLP task.
	
## Setup
To run this project, download code to local using :

```
$ cd ../<yourfolder>
$ git clone https://github.com/biste404/MutSigExperiments

# or 

$ git clone https://github.com/biste404/MutSigExperiments
```