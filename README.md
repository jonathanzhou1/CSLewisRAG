# UNIV 0400: Beyond Narnia: The Literature of C.S. Lewis Final Project
This is a Retrieval Augmented Generation (RAG) system built as a final project for UNIV 0400. This is a tool meant purely for internal use for students taking the course.

## How does it work?
Use FAISS as a vector database and OpenAI's API for Embeddings and for querying. First chunk the given book pdf in a books repository and embed those chunks. Store those embeddings in the vector database. Upon being given a query, find the 'k' (cofigurable) most similar chunks to the query. Then, pass this context along with the query to an LLM model with a slightly prompt-engineered prompt to receive a formatted answer.

By default, running the script will store your vector database (and by extension, embeddings) onto disk in a vector_db folder. To re-use these embeddings instead of re-embedding the book, set the newDB flag to False. If you want to re-embed, such as in the case where you want to query on a new book, set the newDB flag to True.

## Instructions
1. Clone this repository and make sure you have an up to date python version and package installer (pip). <br>
2. Install all necessary dependencies for the project: ```pip install langchain langchain-community openai faiss-cpu python-dotenv pypdf``` <br>
3. Set up your OpenAI API key in a .env file: ```OPENAI_API_KEY=sk-...```. (Alternatively, email me and I can provide my API key with some remaining money from testing still on it). <br>
4. Set up the /books folder and /vector_db folder. /vector_db should be empty, and /books should contain a pdf of the book you want to query on.
4. Modify the query to be your desired question, and run the project!
