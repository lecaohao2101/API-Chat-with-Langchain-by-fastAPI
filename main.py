# #choose
# import os
# import dotenv
# import uvicorn
# from pymongo import MongoClient
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["*"],
# )
#
# @app.on_event("startup")
# async def startup_event():
#     dotenv.load_dotenv()
#     MONGO_URI = os.getenv("MONGO_URI")
#     client = MongoClient(MONGO_URI)
#     db = client.transcripts
#     global transcript_collection
#     transcript_collection = db.transcripts
#
# def process_transcript_files(all_text):
#     # split into chunks
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(all_text)
#     return chunks
#
# class Question(BaseModel):
#     question: str
#
# @app.post("/ask")
# async def ask_question(question: Question):
#     user_question = question.question
#
#
#     mongo_transcripts = list(transcript_collection.find({}, {"_id": 0, "text": 1}))
#     all_text = "\n".join([transcript["text"] for transcript in mongo_transcripts])
#
#
#     chunks = process_transcript_files(all_text)
#
#
#     embeddings = OpenAIEmbeddings()
#     knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)
#
#     docs = knowledge_base.similarity_search(user_question)
#
#     llm = OpenAI(model_name="gpt-3.5-turbo")
#     chain = load_qa_chain(llm, chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=user_question)
#
#     return {"response": response}
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001)
#

import os
import dotenv
import uvicorn
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from functools import lru_cache  # Sử dụng lru_cache để caching
# from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    dotenv.load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client.transcripts
    global transcript_collection
    transcript_collection = db.transcripts


    global embeddings, knowledge_base
    # embeddings = HuggingFaceInstructEmbeddings()
    # embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = OpenAIEmbeddings()
    mongo_transcripts = list(transcript_collection.find({}, {"_id": 0, "text": 1}))
    all_text = "\n".join([transcript["text"] for transcript in mongo_transcripts])
    chunks = process_transcript_files(all_text)
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

def process_transcript_files(all_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
    return chunks

class Question(BaseModel):
    question: str

@lru_cache(maxsize=128)
def answer_question(user_question):
    docs = knowledge_base.similarity_search(user_question, k=5)

    # llm = OpenAI(model_name="gpt-3.5-turbo")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)

    return response

@app.post("/ask")
async def ask_question(question: Question):
    user_question = question.question
    response = answer_question(user_question)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
