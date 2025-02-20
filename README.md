# 🤖 Retrieval-Augmented Generation (RAG) 🔬
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
## 📌 Overview
Retrieval-Augmented Generation (RAG) is an AI technique that combines **retrieval-based** and **generation-based** approaches to improve the quality and accuracy of generated text. This method enhances language models by allowing them to access external knowledge documents during the generation process, leading to more informative and context-aware responses.

## 🔍 How RAG Works
1. **Retrieval Phase**: The model first searches an external knowledge base (e.g., a document repository, a vector database) to find relevant information based on the input query.
2. **Augmentation Phase**: The retrieved data is then incorporated into the model’s context, enriching its understanding.
3. **Generation Phase**: The language model generates a response using both the retrieved data and its pre-trained knowledge.

## 📌 Why Use RAG?
✅ **Enhanced Accuracy** – Reduces hallucinations by grounding responses in real data and can retrieve the latest information without retraining the model.

⚾ ***Example:*** Imagine that you're building a Sports AI and need to implement a service that connects to an LLM(Large Language Models) to interact with your subscribers. When a user asks the LLM—without RAG—about the score of Liverpool’s last game, it will likely provide an incorrect answer, which means the model is hallucinating. The RAG technique can significantly reduce these hallucinations.
🤡 Please, don't put clue on your pizza
 
✅ **Domain-Specific Adaptation** – Enables customization by retrieving data from specialized sources.  

📲 ***Example:*** Imagine that you're building an AI to contact your customers via messages. If a customer asks when their payment is due, the LLM will probably not be able to answer. In that case, you need to provide this information to it, and the "R.A.G" can help you!

✅ **Efficient Use of Resources** – Combines the benefits of retrieval models and generative AI without excessive computational costs.

## 🚀 Getting Started

### First of all, you'll need to retrieve external knowledge from a database or any static data source.
Let's suppose that our data is:

```python
documents = [
    {
        "id": 1,
        "text": "Our company closes at 9 P.M. every day."
    },
    {
        "id": 2,
        "text": "Our payment is due on the 1st of each month"
    },
    {
        "id": 3,
        "text": "Our company is located in the heart of downtown. The address is 123 Main St."
    },
    {
        "id": 4,
        "text": "Our company is open 7 days a week."
    },
    {
        "id": 5,
        "text": "The last match Liverpool played was against Plymouth Argyle, and Liverpool lost 1-0."
    },
    
]
```
## 🐍 Implementing a simple RAG model using Python:

▶️ Let's choose our embedding model. There are many options, such as all-MPNet-base-v2, distilbert-base-nli-stsb-mean-tokens, sentence-t5-base, and all-MiniLM-L6-v2, etc.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("<Your embedding model>")
```

🛑 Wait! But what is a embedding model? 
An **embedding model** is a type of machine learning model that converts text into **dense numerical vectors** in a high-dimensional space. These vectors capture the **semantic meaning** of the text, enabling various NLP tasks.
 **In other words, we can apply mathematical operations to compare texts!** 🧮

🌐 *Step Two:* Just embed our external knowledge from the docs for future comparisons.

```python
from utilities import Docs
docs_embeddings = {
    doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in Docs.documents
}
```

⚡ Using FastAPI or any other method to receive user input, we embed the input, compare it with our documents, and retrieve the external knowledge that is closest to the user's query.
```python
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI

app = FastAPI()

@app.post("/query")
def read_root(request: QueryRequest):
 
    query_embedding = model.encode(request.query, convert_to_tensor=True)

    match_document = max(Docs.documents, key=lambda doc: util.pytorch_cos_sim(query_embedding, docs_embeddings[doc["id"]]))

    best_score_dodument = util.pytorch_cos_sim(query_embedding, docs_embeddings[match_document["id"]])
```
📓 Note that each embedding generates a vector, and comparing two embeddings using pytorch_cos_sim produces a float value called a "score." This coefficient represents the similarity between the given data embedding and each previously embedded document. So, we must get the document with the maximum score.

⏫ and then, we can create our prompt like: 

```python
prompt = f"You are a AI assistant. You are helping a user with a question. Your response must be based only on this document: '{match_document["text"]}' \nUser: " + request.query
```
🔚 Finally, you can send this prompt for any model of LLM, in this case, I have used **gpt-4o-mini**.

```python
from openai import OpenAI
def getAns(prompt):

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = 500
    )

    return completion.choices[0].message.content

```

## 💠 Conclusion
🥳 Now, the answers from your AI assistant are based on your context. If the user asks about the payment due, the assistant will be able to help. If the user talks about the last Liverpool game, the assistant will also be able to assist.

---
