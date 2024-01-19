import matplotlib.pyplot as plt
import csv
import numpy as np

# List to store data from CSV files
data = []

# File names of the CSV files
file_names = ['./../results/Mixtral-8x7B-Instruct-v0.1/eval/base_eval.csv',
              './../results/Mixtral-8x7B-Instruct-v0.1/eval/base_explain_eval.csv',
              './../results/Mixtral-8x7B-Instruct-v0.1/eval/base_probs_eval.csv',
              './../results/Mixtral-8x7B-Instruct-v0.1/eval/base_explain_probs_eval.csv',
              # './../results/Mixtral-8x7B-Instruct-v0.1/eval/reviews_eval.csv',
              # './../results/Mixtral-8x7B-Instruct-v0.1/eval/reviews_explain_eval.csv',
              # './../results/gpt-3.5-turbo/eval/base_eval.csv',
              # './../results/gpt-3.5-turbo/eval/base_explain_eval.csv',
              # './../results/gpt-3.5-turbo/eval/reviews_explain_eval.csv',
              # './../results/gpt-3.5-turbo/eval/reviews_eval.csv',
              ]

# Read data from each file and store in a list of dictionaries
for file_name in file_names:
    with open(file_name, 'r') as file:
        csv_reader = csv.DictReader(file)
        metrics_data = {row['Metric']: float(row['Value']) for row in csv_reader}
        data.append(metrics_data)

# List of metrics
metrics = ['F1 Score', 'Recall', 'Precision', 'ROC AUC']
num_metrics = len(metrics)
num_files = len(file_names)

bar_width = 0.1
index = np.arange(num_metrics)

# Plotting the metrics for comparison
plt.figure(figsize=(10, 6))

for i, file_data in enumerate(data):
    label = '_'.join([file_names[i].split('/')[-3].split('-')[0]] + [file_names[i].split('/')[-1].split('_eval')[0]])
    values = [file_data[metric] for metric in metrics]
    plt.bar(index + i * bar_width, values, bar_width, alpha=0.7, label=label)

plt.xlabel('Metrics')
plt.ylabel('Metric Value')
plt.title('Results')
plt.xticks(index + bar_width * (num_files - 1) / 2, metrics)
plt.ylim(0, 1.0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Save the plot as evals.png
plt.savefig('probs_mixtral.png')

# Show the plot
plt.show()

