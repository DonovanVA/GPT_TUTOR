
import os
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
## loads the model and the embeddings to be used
def load_embeddings (model_path,persist_directory,document_name,collection_name):
        
    # Initialize the LlamaCppEmbeddings object using the GPT4ALL model
    llama_embeddings = LlamaCppEmbeddings(model_path=model_path)
    
    # If the vector database does not exist, create it and persist it to disk
    if not os.path.isdir(persist_directory):
        # Load the text documents
        print('Parsing ' + document_name)
        loader = TextLoader(document_name)
        documents = loader.load()

        # Split the text into chunks and create a Chroma vector store from them
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents=texts, embedding=llama_embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vectordb.persist()
        print(vectordb)
        print('Saved to ' + persist_directory)

    # If the vector database already exists, load it from disk
    else:
        print('Loading ' + persist_directory)
        vectordb = Chroma(persist_directory=persist_directory,
                        embedding_function=llama_embeddings, collection_name=collection_name)
        print(vectordb)

    # Initialize the LlamaCpp model object
    llm = LlamaCpp(model_path=model_path)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 1}))
