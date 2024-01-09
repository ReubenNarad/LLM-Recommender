from langchain.chat_models import ChatOpenAI
import sys
sys.path.append('tools')
from utils import *
import openai_chains
import concurrent.futures
import csv
import os
from tqdm import tqdm
from time import sleep

# Configure run
MODEL_NAME = 'gpt-3.5-turbo'
CHAIN_NAME = 'keyphrases_explain'
DATA = './../data/video/preprocessed_movies.json'

SAMPLE_SIZE = 2000
MAX_THREADS = 4

output_path = f"../results/{MODEL_NAME}/output"
metrics_path = f"../results/{MODEL_NAME}/eval"

opai_api_key = "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"

# Retrieve user data
users = extract_rows(SAMPLE_SIZE, DATA)
print(f"Extracted {len(users)} rows ...")

# Generate chain
model = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=opai_api_key, temperature=0)
chain = openai_chains.get_chain(CHAIN_NAME, model)

# Initialize empty lists to store results
pred = []
truth = []
title = []

# Define a function to be run in each thread
def process_user(user):
    sleep(.1)
    try:
        prefs = format_preferences(user)

        # This is where the model is actually called
        response = chain.invoke(prefs)
        prediction = json.loads(response.additional_kwargs['function_call']['arguments'])
        sleep(.2)
        return {
            'pred': prediction['reccomend'],
            'truth': prefs['truth'],
            'title': prefs['target'],
            'explanation': response.additional_kwargs['function_call']['arguments'],
        }
    except Exception as e:
        print(f"Error: {e}")
        print(f"User: {list(user.keys())[0]}")
        sleep(3)
        return None

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    # Use list() to make sure all threads are finished
    print("Making requests...")
    results = list(tqdm(executor.map(process_user, users), total=len(users)))
    results = [i for i in results if i is not None]
    print("Done!")
    
# Log the results into a csv file

# Create directories if they don't exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)

csv_path = os.path.join(output_path, f"{CHAIN_NAME}.csv")
metrics_csv_path = os.path.join(metrics_path, f"{CHAIN_NAME}_eval.csv")

# Save results
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['pred', 'truth', 'title', 'explanation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

        
# Evaluate results
f1, recall, precision, roc_auc = evaluate(results)        

# Save metrics to a separate CSV file
with open(metrics_csv_path, 'w', newline='') as metrics_csvfile:
    fieldnames_metrics = ['Metric', 'Value']
    writer_metrics = csv.DictWriter(metrics_csvfile, fieldnames=fieldnames_metrics)
    writer_metrics.writeheader()
    writer_metrics.writerow({'Metric': 'F1 Score', 'Value': f1})
    writer_metrics.writerow({'Metric': 'Recall', 'Value': recall})
    writer_metrics.writerow({'Metric': 'Precision', 'Value': precision})
    writer_metrics.writerow({'Metric': 'ROC AUC', 'Value': roc_auc})