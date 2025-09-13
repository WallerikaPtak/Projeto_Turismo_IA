# Versão Final - main.py
# Este código usa a sintaxe moderna do LangChain (LCEL) para evitar erros de parsing.
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Carregar variáveis de ambiente
load_dotenv()

# --- 1. Configurações Iniciais ---
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant", # Usando o modelo estável
    api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

print(f"Conectando ao índice '{PINECONE_INDEX_NAME}' no Pinecone...")
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
print("Conexão bem-sucedida.")

# --- 2. Cadeias Especializadas (Runnables) ---

# Roteiro de Viagem
prompt_roteiro = PromptTemplate.from_template(
    """Você é um especialista em viagens. Crie um roteiro de viagem detalhado para a seguinte solicitação:
    {input}"""
)
chain_roteiro = prompt_roteiro | llm | StrOutputParser()

# Logística de Transporte
prompt_logistica = PromptTemplate.from_template(
    """Você é um especialista em logística de viagens. Responda à seguinte pergunta sobre transporte:
    {input}"""
)
chain_logistica = prompt_logistica | llm | StrOutputParser()

# Tradução
prompt_traducao = PromptTemplate.from_template(
    """Traduza ou forneça frases úteis para a seguinte solicitação de viagem:
    {input}"""
)
chain_traducao = prompt_traducao | llm | StrOutputParser()

# Informação Local (usando RAG)
chain_info_local = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# --- 3. Lógica do Roteador (estilo moderno) ---

router_template = """Dada uma pergunta, classifique-a em uma das seguintes categorias:
'roteiro-viagem', 'info-local', 'logistica-transporte', 'traducao-idiomas'.
Retorne apenas o nome da categoria.

Pergunta: {input}
Categoria:"""
prompt_router = PromptTemplate.from_template(router_template)

# Cadeia que apenas classifica a pergunta e retorna um texto simples
chain_classificadora = prompt_router | llm | StrOutputParser()


# --- 4. Execução e Lógica de Decisão ---

if __name__ == '__main__':
    print("🤖 Olá! Sou seu assistente de viagens. Como posso ajudar?")
    # ... (instruções) ...
    
    while True:
        user_input = input("\nSua pergunta: ")
        if user_input.lower() == 'sair':
            break
        
        print("Classificando sua pergunta...")
        # Primeiro, usamos a cadeia classificadora para obter o tópico
        topico = chain_classificadora.invoke({"input": user_input})
        print(f"Tópico identificado: {topico}")

        # Agora, tomamos a decisão em Python, o que é mais robusto
        if "roteiro-viagem" in topico:
            print("Gerando um roteiro...")
            response = chain_roteiro.invoke({"input": user_input})
            print("\nResposta:", response)
        elif "info-local" in topico:
            print("Buscando informação local...")
            # A cadeia de RAG espera a chave 'query'
            response = chain_info_local.invoke({"query": user_input})
            print("\nResposta:", response['result'])
        elif "logistica-transporte" in topico:
            print("Consultando logística...")
            response = chain_logistica.invoke({"input": user_input})
            print("\nResposta:", response)
        elif "traducao-idiomas" in topico:
            print("Traduzindo...")
            response = chain_traducao.invoke({"input": user_input})
            print("\nResposta:", response)
        else:
            # Rota padrão se a classificação não for clara
            print("Não tenho certeza da categoria, buscando informação local por padrão...")
            response = chain_info_local.invoke({"query": user_input})
            print("\nResposta:", response['result'])