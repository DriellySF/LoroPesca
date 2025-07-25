# Empresa de Pesca Q&A App
Este é um aplicativo de perguntas e respostas sobre o catálogo de produtos da **Empresa de Pesca**
---
(Case que fiz para uma entrevista)


## Tecnologias
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [OpenRouter](https://openrouter.ai/)
---
## Acesse o projeto clicando [aqui](https://mainpy-fcxwavudwmpvbpkhraxo6a.streamlit.app/)

## Como rodar o projeto localmente


### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/loro-pesca-qa.git
cd loro-pesca-qa
```

### 2. (Opcional) Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Crie o arquivo .env ou toml dp streamlit

No diretório raiz, crie a pasta `.streamlit` e dentro dela o arquivo `secrets.toml` ou .env com o seguinte conteúdo:

```toml
OPENAI_API_KEY = "chave-openrouter"
OPENAI_API_BASE = "https://openrouter.ai/api/v1"
```

### 6. Execute o app

```bash
streamlit run app.py
```

### 7. Acesse no navegador

```
http://localhost:8501
```

## Veja a seguir alguns exemplos de perguntas e respostas:

<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/img%20(5).png?raw=true" width="600" />
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/img%20(4).png?raw=trueg" width="600" />
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/img%20(3).png?raw=true" width="600" />
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/img%20(2).png?raw=true" width="600" />
<img src="https://github.com/DriellySF/LoroPesca/blob/main/img/img%20(1).png?raw=true" width="600" />

Para saber mais sobre o processo de criação e aprendizado, clique em [Linha de raciocínio](https://github.com/DriellySF/LoroPesca/blob/main/linha%20de%20raciocinnio.md)
