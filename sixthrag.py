import ollama
import bs4
import chromadb
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader

#accounting test = db7
#physics test = db8
#old phys test = db9
#new test using csv -> narratives2 txt: db13
#acct test with custom narrartives - db22
#acct and fil studies test witrh custom narratives - db23
persist_directory= 'db24'
#reading csvs
loader = TextLoader("narratives6.txt", encoding = 'utf-8')
#print("loaded csv as txt")
#reading docs
#loader = Docx2txtLoader(r"C:\Users\h703158224\Downloads\2-Computer-Science-Programs.docx")

#loader = TextLoader("sample.txt")
#loader.load()

# 1. Load the data
#loader = WebBaseLoader(
#    web_paths=("https://en.wikipedia.org/wiki/Large_language_model",),
#   bs_kwargs=dict(
    #    parse_only=bs4.SoupStrainer(
    #        class_=("post-content", "post-title", "post-header")
    #    )
   # ),
#)
docs = loader.load()
print(docs)
#changed chunk size from 1000 to 3000
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")

print("embeddings...", embeddings)
print("splits")

if os.path.exists(persist_directory):
    vectorstore = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
else:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings , persist_directory= persist_directory)
#vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("right before ollama")
# 3. Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    #print("formatted prompt: ", formatted_prompt)
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    #print('response: ', response)
    return response['message']['content']


# 4. RAG Setup
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    #print("retrived docs:", retrieved_docs)
    formatted_context = combine_docs(retrieved_docs)
    #print("formatted context:", retrieved_docs)
    return ollama_llm(question, formatted_context)

# 5. Use the RAG App
#result = rag_chain("What is a LLM?")
#print(result)

print("done. time to query")
#querying
while True:
    question = input("Ask a question or type 'exit' to quit: ")
    if question == 'exit':
        print("Exiting...")
        break
    else:
        answer = rag_chain(question)
        print("Answer:", answer)
