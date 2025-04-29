from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

#Load OpenAI API key from .env file
load_dotenv()

#Error if no API key is set
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to the .env file")

#Embedding model
embedding = OpenAIEmbeddings()

#Flag to indicate whether to reuse disk saved vector db or re-embed
newDB = False

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
    vectordb = FAISS.load_local("vector_db", embeddings=embedding, allow_dangerous_deserialization=True) #Only run if you trust the vec db files

# Query ----------------------------- 

query = "What are some quotes related to pride in The Great Divorce?"
#Retrieve similar chunks
docs = vectordb.similarity_search(query, k=10) #modify the top k value here

#Prepare context
context = "\n\n".join(doc.page_content for doc in docs)
#Create prompt
prompt = ChatPromptTemplate.from_template(
    '''Use the following context to answer the query:\n\n{context}\n\nQuery: {question}. Please print the query at the top of the response. 
    Please provide your reasoning for whatever answer you give.
    For each quote, you must provide context from the book for the quote. Thank you!'''
)

# RAG w/ LLM -- set up the model
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
#Format the input
input_vars = {"context": context, "question": query}
#call the LLM model
response = llm.invoke(prompt.format_messages(**input_vars))

print(response.content)