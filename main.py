import os
import dotenv
import uvicorn
from pymongo import MongoClient
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    dotenv.load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client.transcripts
    global transcript_collection
    transcript_collection = db.transcripts


def process_transcript_files(all_text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
    return chunks


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(question: Question):
    user_question = question.question

    # Get all transcript files from MongoDB
    mongo_transcripts = list(transcript_collection.find({}, {"_id": 0, "text": 1}))
    all_text = "\n".join([transcript["text"] for transcript in mongo_transcripts])

    # Process transcript files
    chunks = process_transcript_files(all_text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)

    docs = knowledge_base.similarity_search(user_question)

    llm = OpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)

    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)