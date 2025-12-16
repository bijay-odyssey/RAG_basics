from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

COLLECTION_NAME = "documents"
TOP_K = 3

client = Groq()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant_client = QdrantClient(host="localhost", port=6333)


def retrieve_context(query, top_k=TOP_K):
    query_vector = embedding_model.encode(query).tolist()

    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [point.payload["text"] for point in search_result.points]


def build_prompt(query, contexts):
    context_text = "\n".join(f"- {c}" for c in contexts)

    return f"""You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context_text}

Question:
{query}

Answer:
"""


def get_available_chat_model():
    models = client.models.list().data

    preferred_keywords = ["instruct", "chat"]

    for model in models:
        name = model.id.lower()
        if any(k in name for k in preferred_keywords):
            return model.id

    return models[0].id


def generate_answer(prompt):
    model_name = get_available_chat_model()
    # print(f"\n[Using model: {model_name}]")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content


def rag_query(query):
    contexts = retrieve_context(query)
    prompt = build_prompt(query, contexts)
    answer = generate_answer(prompt)
    return contexts, answer


if __name__ == "__main__":
    print("\n=== BASIC RAG CHATBOT ===")

    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        contexts, answer = rag_query(query)

        print("\nContext:")
        for c in contexts:
            print("-", c)

        print("\nAnswer:")
        print(answer)
