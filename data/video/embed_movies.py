# Adds movie product descriptions to Chroma vectorstore, for similarity search

import json
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Document
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
import os
import threading
from queue import Queue

# Load api key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Movie data path
metadata_file = 'meta_Movies_and_TV.json'     

def worker(input_queue, db_lock):
    while True:
        line = input_queue.get()
        if line is None:
            break  # None is the signal to stop the thread

        try:
            entry = json.loads(line.strip())
            asin = entry.get('asin')
            if not asin:
                continue

            title = entry.get('title')
            description = entry.get('description')
            category = entry.get('main_cat')

            if description:  # Making sure the description is not empty
                combined_description = f"Title: {title}, Description: {description}"
                document = Document(page_content=combined_description, metadata={"title": title})

                with db_lock:  # Assuming `db` is not thread-safe
                    db.add_documents(documents=[document], ids=[asin])

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            input_queue.task_done()

# Create a queue and a lock
input_queue = Queue()
db_lock = threading.Lock()

# Define and start worker threads
num_threads = 5
threads = []
for _ in range(num_threads):
    thread = threading.Thread(target=worker, args=(input_queue, db_lock))
    thread.daemon = True  # Ensure the thread will exit if the main thread ends
    thread.start()
    threads.append(thread)

total = 0
try:
    with open(metadata_file, 'r') as metadata_file:
        for line in metadata_file:
            total += 1
            if total % 1000 == 0:
                print(total)
            input_queue.put(line)

finally:
    # Signal the worker threads to stop
    for _ in range(num_threads):
        input_queue.put(None)

    # Wait for all items in the queue to be processed
    input_queue.join()

print(total)