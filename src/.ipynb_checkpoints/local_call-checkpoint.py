import sys
sys.path.append('tools')
from utils import *
import local_llm_chains

import concurrent.futures, torch, csv, os, warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from time import sleep

# Configure run
MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
CHAIN_NAME = 'base_explain_probs'
DATA = './../data/video/preprocessed_movies.json'
CACHE = '/mnt/sdb1/LLM_RecSys/models'

SAMPLE_SIZE = 500
MAX_THREADS = 4

output_path = f"../results/{MODEL_NAME.split('/')[-1]}/output"
metrics_path = f"../results/{MODEL_NAME.split('/')[-1]}/eval"

# Retrieve user data
users = extract_rows(SAMPLE_SIZE, DATA)
print(f"Extracted {len(users)} rows ...")

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Load model, tokenizer, and chain
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             load_in_4bit=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             cache_dir=CACHE,
                                             do_sample=True,
                                             temperature=.1
                                             )
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          padding_side='left')

# Retrieve chain
chain = local_llm_chains.get_chain(CHAIN_NAME, model, tokenizer)

# Define inference calling for user
def process_user(user):
    try:
        prefs = format_preferences(user)

        # This is where the model is actually called
        response = chain.invoke(prefs)
        print(response)
        return {
            'pred': response['recommend'],
            'truth': prefs['truth'],
            'title': prefs['target'],
            'explanation': response['explanation'],
        }
    except Exception as e:
        print(f"Error: {e}")
        print(f"User: {list(user.keys())[0]}")
        return None

# Create threading and execute
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    # Use list() to make sure all threads are finished
    print("Generating results...")
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

if 'probs' not in CHAIN_NAME:

    # Evaluate results
    preds = [result['pred'] for result in results]
    truth = [result['truth'] for result in results]
    f1, recall, precision, roc_auc = evaluate(pred, truth)        

    # Save metrics to a separate CSV file
    with open(metrics_csv_path, 'w', newline='') as metrics_csvfile:
        fieldnames_metrics = ['Metric', 'Value']
        writer_metrics = csv.DictWriter(metrics_csvfile, fieldnames=fieldnames_metrics)
        writer_metrics.writeheader()
        writer_metrics.writerow({'Metric': 'F1 Score', 'Value': f1})
        writer_metrics.writerow({'Metric': 'Recall', 'Value': recall})
        writer_metrics.writerow({'Metric': 'Precision', 'Value': precision})
        writer_metrics.writerow({'Metric': 'ROC AUC', 'Value': roc_auc})
