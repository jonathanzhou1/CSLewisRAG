from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

#Load OpenAI API key from .env file
load_dotenv()

#Embedding model
embedding = OpenAIEmbeddings()

#Flag to indicate whether to reuse disk saved vector db or re-embed
newDB = True

if newDB:
    #If not reusing vector db: 
    # Load documents
    pdf_folder_path = "books"
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]
    all_documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_documents.extend(docs)

    # Chunk split after loading the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)

    # Embed and store
    vectordb = FAISS.from_documents(chunks, embedding)

    # Optional step: store the vector db into disk for reuse
    vectordb.save_local("vector_db")
else:
    vectordb = FAISS.load_local("vector_db", embeddings=embedding)

# Query
query = "faithfulness"
docs = vectordb.similarity_search(query, k=10) #modify the top k value here

print(docs)

# # RAG w/ LLM
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# llm = ChatOpenAI(model_name="gpt-4")
# chain = load_qa_with_sources_chain(llm, chain_type="stuff")
# response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
