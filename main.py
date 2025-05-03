import streamlit as st
from langchain_openai import ChatOpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# para rodar local:
# load_dotenv()
# api_key = ""
# api_base = "https://openrouter.ai/api/v1"

api_key = st.secrets["OPENAI_API_KEY"]
api_base = st.secrets.get("OPENAI_API_BASE")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")

def chunks(df):
    docs = []
    for _, row in df.iterrows():
        doc = f"""
Product Name: {row['product_name']}
Product Group: {row.get('product_group', '')}
Category: {row.get('category_en', '')}
Price: R$ {row['price']}
Availability: {row['availability']}
Technical Details: {row.get('technical_details', '')}
"""
        docs.append(doc.strip())
    return docs

def faiss_index(docs):
    embeddings = embed_texts(docs)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, docs

def important_docs(user_question, index, docs, k=12):
    q_embedding = embed_texts([user_question])
    scores, indices = index.search(q_embedding, k)
    return [docs[i] for i in indices[0] if i < len(docs)]

def main():
    st.set_page_config(page_title="Document Q&A sobre catálogo")
    st.title("Pergunte sobre os produtos da Loro Pesca!")
    st.write(" Nós temos carretilhas, molinetes targus, varas trigger, iscas artificiais yara, linha monofilamento, alicates de contenção, boias torpedo e anzois marine sport.")

    if "history" not in st.session_state:
        st.session_state.history = []

    df = pd.read_csv('catalog.csv')
    docs = chunks(df)
    index, _, doc_chunks = faiss_index(docs)
    user_question = st.text_input("Digite sua pergunta sobre os produtos:")
    
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key = api_key,
        openai_api_base = api_base,
        model="deepseek/deepseek-chat-v3-0324:free"
    )

    if user_question:
        relevant_docs = important_docs(user_question, index, doc_chunks)
        context = "\n\n".join(relevant_docs)
        prompt = f"""
You are a product assistant. Use the product catalog below to answer the question:

CATALOG:
{context}

QUESTION: {user_question}
ANSWER:
"""
        response = llm.invoke(prompt)
        st.markdown("**Resposta:**")
        st.write(response.content)
        st.session_state.history.append((user_question, response.content))

        if st.session_state.history:
            st.markdown("### Conversa:")
            for i, (pergunta, resposta) in enumerate(st.session_state.history):
                st.markdown(f"""
                <div style="background-color:#DCF8C6; border-radius:10px; padding:10px; margin-bottom:10px; max-width:75%; margin-left:auto;">
                    <strong>Você:</strong><br>{pergunta}
                </div>
                <div style="background-color:#F1F0F0; border-radius:10px; padding:10px; margin-bottom:20px; max-width:75%;">
                    <strong>Loro Pesca:</strong><br>{resposta}
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
