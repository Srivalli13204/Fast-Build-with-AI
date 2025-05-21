import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# --------------------- Step 1: API Key Setup ---------------------
# Replace with your actual OpenAI API key or use environment variable
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # <-- Replace this

# --------------------- Step 2: Load Data ---------------------
print("Loading data from faq.csv...")
loader = CSVLoader(file_path='faq.csv', source_column="answer")
documents = loader.load()

# --------------------- Step 3: Text Splitting ---------------------
print("Splitting and processing documents...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# --------------------- Step 4: Vector Embedding & Indexing ---------------------
print("Generating embeddings and creating vector store...")
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# --------------------- Step 5: Create Retrieval Chain ---------------------
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --------------------- Step 6: Sample Q&A & Save ---------------------
print("Saving sample questions and responses...")
sample_qna = [
    {"Question": "What is LangChain?", "Answer": qa_chain({"query": "What is LangChain?"})['result']},
    {"Question": "What is RAG?", "Answer": qa_chain({"query": "What is RAG?"})['result']},
    {"Question": "Who created LangChain?", "Answer": qa_chain({"query": "Who created LangChain?"})['result']}
]

df = pd.DataFrame(sample_qna)
df.to_excel("responses.xlsx", index=False)
print("Sample responses saved to responses.xlsx")

# --------------------- Step 7: Chatbot Loop ---------------------
print("\n🔹 Chatbot is ready! Type your questions below (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    result = qa_chain({"query": user_input})
    print("Bot:", result['result'])
