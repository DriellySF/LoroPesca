# LORO desafio - linha de raciocínio

Primeira coisa que fiz foi analisar e tratar os dados. (Obserse no arquivo exploratory_analysis.ipynb.) Verifiquei se precisava tratar algum NaN, null e juntei os dois datasets em um só.

> You may choose any architecture: e.g., embeddings + retrieval, full-text search, or any other technique that fits. Use your own tech stack and tools.
> 

O que são cada uma dessas arquiteturas? O que considerar ao escolher cada uma?**
O que é streamlit? Quais conceitos teóricos preciso saber para entender a lógica de um chatbot?

A partir dessas perguntas ao ChatGPT, criei um mini roteiro de estudos. O primeiro vídeo que assisti foi

https://www.youtube.com/watch?v=tjeti5vXWOU,

Além do youtube, algumas referências que utilizei:

https://github.com/langchain-ai/langchain/discussions/27964

https://python.langchain.com/v0.1/docs/integrations/toolkits/csv/

Eis uma das primeiras versões do código:

```python
import streamlit as st
from langchain.agents import create_csv_agent
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.llms.openai import OpenAi
from langchain_community.llms.openai import OpenAI
from langchain_openai import ChatOpenAI


def main():

    st.set_page_config(page_title="ask your csv")
    st.header("Ask your csv")

    user_csv = st.file_uploader("Upload your csv file", type = "csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your csv")

        #llm = OpenAI(temperature=0)

        llm = ChatOpenAI(temperature=0, openai_api_key="", openai_api_base="https://openrouter.ai/api/v1", model="qwen/qwen3-0.6b-04-28:free")
        #0 é igual, 1 é criativo
        agent = create_csv_agent(llm, user_csv, verbose=True, allow_dangerous_code=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

        #verbose printa o processo de "pensamento do programa"

        if user_question is not None and user_question != "":
            st.write(f"Your Question was: {user_question}")

            response = agent.run(user_question)

            st.write(response)

if __name__ == "__main__":
    main()
```
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/image.png?raw=true">

Como teste, usei a primeira pergunta: “price of Carretilha Vizel Air 201”, um produto que não existe no dataset. O preço retornado é da CARRETILHA VIZEL AIR 713, indicando que o modelo dá match sem considerar a numeração final. O match deveria ser exato para funcionar corretamente.

Obviamente não funcionou. Aprendi depois o porquê. Pesquisando um pouco mais ( https://youtu.be/yfHHvmaMkcA?feature=shared e https://janeladodev.com.br/inteligencia-artificial/utilizando-sentence-transformers-para-geracao-de-embeddings/), entendi um pouquinho dos vetores. 

A primeira tentativa de embbeding foi similaridade dos cossenos do sklearn, pois já havia usado (embora como caixa preta) em projetos anteriores, mas com a adição do match exato, para considerar a numeração final de cada produto.

Testei diversos modelos gratuitos do openrouter. Com base da saída do verbose(que explica o “pensamento” do modelo), escolhi o "deepseek/deepseek-prover-v2:free"; a escolha do "all-MiniLM-L6-v2" para o sentence tranformer foi arbitrária (a mais comum nos tutoriais do youtube).

```python
import os
import streamlit as st
# from langchain.agents import create_csv_agent
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
# from langchain.llms.openai import OpenAi
# from langchain_community.llms.openai import OpenAI
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    return model.encode(texts, convert_to_tensor=True)

def find_strict_match(df, user_question, top_n=5):
    product_names = df["product_name"].astype(str).tolist()
    product_embeddings = embed_text(product_names)
    question_embedding = embed_text([user_question])

    similarities = cosine_similarity(question_embedding, product_embeddings)[0]
    df["similarity"] = similarities

    top_matches = df.sort_values("similarity", ascending=False).head(top_n)


    user_question_lower = user_question.lower()
    for _, row in top_matches.iterrows():
        name = row["product_name"].lower()
        if name in user_question_lower or all(word in user_question_lower for word in name.split()):
            return row
        product_numbers = ''.join(filter(str.isdigit, name))
        if product_numbers and product_numbers in user_question_lower:
            return row

    return None

def main():

    # load_dotenv()
    st.set_page_config(page_title="ask your csv")
    st.header("Ask your csv")

    user_csv = "ola"
    df = pd.read_csv('merged_catalog.csv')

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your csv")
    

        #llm = OpenAI(temperature=0)

        llm = ChatOpenAI(temperature=0, openai_api_key="", openai_api_base="https://openrouter.ai/api/v1", model="deepseek/deepseek-chat-v3-0324:free")
        #0 é igual, 1 é criativo
        agent = create_csv_agent(llm, 'merged_catalog.csv', verbose=True, allow_dangerous_code=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        #verbose printa o processo de "pensamento do programa"

        if user_question is not None and user_question != "":
            st.write(f"Your Question was: {user_question}")

        
#antes de mandar pro modelo, verifica se o produto existe
        if user_question:
            result = find_strict_match(df, user_question)

            if result is not None:
                st.write("Produto encontrado:")
                st.write(f"Similaridade: {result['similarity']:.2f}")
                response = agent.run(user_question)

                st.write(response)
            else:
                st.warning("Nenhum produto encontrado.")        



if __name__ == "__main__":
    main()
```

Agora a resposta estava correta, indicando que o produto não existia e retornando os existentes:

<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/image%201.png?raw=true">
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/image%202.png?raw=true">
Agora vinha a segunda pergunta:

<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/image%203.png?raw=true">

Percebi que ia precisar repensar a lógica do match exato e fazer a “tradução” dos nomes.

Pesquisei por mais referencias de libs e chats.

https://www.youtube.com/watch?v=hmqYxByTlRs

o principal vídeo foi esse, que me apresentou o faiss e a ideia dos chunks; por sugestão do gpt, usei a função IndexFlatL2

https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html

Depois disso, foi mais tentativa, erro, debugging e detalhes visuais. Assim, cheguei na versão que está postada no github.

Adorei esse case e gostei muito de aprender sobre os tipos de arquitetura. Também não conhecia o streamlit e achei bem prático.

Agradeço a oportunidade.
