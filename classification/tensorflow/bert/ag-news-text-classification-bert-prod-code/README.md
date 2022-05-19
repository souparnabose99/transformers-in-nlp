### AG News Text Classification with BERT ###

This repository covers the code for performing text classification on the AG News dataset using state-of-the-art transformer model BERT.
AG News (AG’s News Corpus) is a sub dataset of AG's corpus of news articles 
constructed by assembling titles and description fields of articles, from the 4 largest classes (“World”, “Sports”, “Business”, “Sci/Tech”)
of AG’s Corpus.
BERT is a state-of-the-art transformer model use to perform complex NLP tasks like NLU, NLG, Sentiment Analysis,
Text Classification...., and it gives very high performance metrics with respect to these, sometimes even surpassing human capabilities.

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