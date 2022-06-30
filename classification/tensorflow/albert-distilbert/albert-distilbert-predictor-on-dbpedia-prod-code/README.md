### Dbpedia Multi-class Dataset with ALBERT & DistilBERT: ###

This repository covers the code for performing multi class text classification on the Dbpedia dataset 
using state-of-the-art Transformer (Auto Encoder) models ALBERT & DistilBERT.
The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014.
ALBERT is a state-of-the-art transformer model used to perform complex NLP tasks like NLU, NLG, Sentiment Analysis,
Text Classification... and optimized on top of BERT to help in memory and space limitations.
DistilBERT is a distilled version of the original BERT model obtained by performing Knowledge Distillation, thereby
making it lighter anf faster.

Below are the steps to be followed:

1. Install the required packages stated in requirements.txt file 
   Packages can be installed on an Anaconda environment or on normal python interpreter.
   For Anaconda:
   conda create --name <youenvname>
   conda activate <yourenvname>
   pip install -r requirements.txt
   For Python Interpreter:
   pip install -r requirements.txt
   
2. The entire repository is modularised in to individual sections which performs specific task.
   First, go to src folder.
   Under src, there are 2 primary packages:
   a> ML_Pipeline:
   This contains individual modules with different function declarations to perform specific Machine Learning task.
   b> engine.py:
   This is the heart of the project, as all the function calls are done here.
   
3. Run/Debug the engine.py file and all the steps will be automatically taken care as per the logic.

4. All input datasets are stored in the input folder.

5. All predictions and models are stored in the output folder.