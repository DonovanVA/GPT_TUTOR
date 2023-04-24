# > pip install -r requirements.txt
# install gpt4all binaries here:https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/
# Import necessary packages
from load_embeddings import load_embeddings
# Define the path to the GPT4ALL model file
MODEL_PATH = "ggml-model-q4_0.bin"

# Define the directories and file names for the text documents and vector database
PERSIST_DIRECTORY = './.chroma'
COLLECTION_NAME = 'data'
DOCUMENT_NAME= './embeddings.txt'

def askllama(question):
    qa = load_embeddings(MODEL_PATH,PERSIST_DIRECTORY,DOCUMENT_NAME,COLLECTION_NAME)
    print(qa.run(question))
    
askllama("hello who is this")
