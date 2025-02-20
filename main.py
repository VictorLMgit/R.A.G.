from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from services.tirdhPartyIA import getAns
from utilities import Docs


app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

docs_embeddings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in Docs.documents
}


class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def read_root(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    match_document = max(Docs.documents, key=lambda doc: util.pytorch_cos_sim(query_embedding, docs_embeddings[doc["id"]]))
    best_score_dodument = util.pytorch_cos_sim(query_embedding, docs_embeddings[match_document["id"]])


    prompt = f"You are a AI assistant. You are helping a user with a question. Your response must to be based only in this document: '{match_document["text"]}' \n\nUser: " + request.query + ". Assistant:"
    response = getAns(prompt)
    return {"response": response, "doc_based": match_document["text"] , "best_score_dodument": best_score_dodument.item()}


