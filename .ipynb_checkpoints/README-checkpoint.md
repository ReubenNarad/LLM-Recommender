# Product Recommender Model Based on LLM

## Overview
This repository contains work for a research project exploring product recommender models using fine-tuned LLMs.

## Directory Structure
- `data/`: Directory storing formatted data used by the model.
    - `video/`: Movie and TV review datasets.
        - `preprocessed_movies.json`: 
        - `embed_movies.py`: Python script for embedding movies.
        - `test_query.ipynb`: Jupyter notebook for testing queries against the movie database.
        - `chroma_db/`: Contains the sqlite3 database for Chroma.
      
- `src/`: Python files and notebooks for conducting experiments and building models.
    - `preprocess_data.ipynb`: Jupyter notebook for pre-processing the data. It combines metadata and reviews into a single JSON file.
    - `local_call.py`: Python script for evaluating local models.
    - `openai_call.py`: Python script for evaluating OpenAI models.
    - `request_test.ipynb`: Jupyter notebook for testing LLM queries and chains
    - `tools/`: Directory containing utility scripts.
        - `local_llm_chains.py`: Python script defining chains for local models.
        - `openai_chains.py`: Python script defining chains for OpenAI models
        - `utils.py`: Python script with various utility functions.
    
- `results/`: Directory where experiment results are saved.
    - `Mixtral-8x7B-Instruct-v0.1/`: Contains outputs and evaluations for the Mixtral-8x7B-Instruct-v0.1 model.
    - `gpt-3.5-turbo/`: Contains outputs and evaluations for the gpt-3.5-turbo model.
    - `gpt-4-1106-preview/`: Contains outputs for the gpt-4-1106-preview model.
 
- `figures/`: Directory for generating and saving figures and plots. 
    - `plot_evals.py: Script for creating bar plots of eval files`
