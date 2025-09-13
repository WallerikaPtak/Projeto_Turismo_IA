import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def setup_pinecone_index():
    """
    Verifica se o índice existe no Pinecone. Se não, cria um novo
    com as especificações corretas (Serverless, 384 dimensões) e o popula.
    """
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    # Usando o nome do índice do seu arquivo .env
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise ValueError("⚠️ PINECONE_API_KEY ou PINECONE_INDEX_NAME não configurados no .env")

    print(f"Verificando o índice '{PINECONE_INDEX_NAME}' no Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Verifica se o índice já existe
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Índice '{PINECONE_INDEX_NAME}' não encontrado. Criando um novo...")

        # 1. Criar o índice com as especificações corretas
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # <-- A DIMENSÃO CORRETA PARA O NOSSO MODELO
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # Região da sua conta
            )
        )
        
        print("Aguardando o índice ficar pronto...")
        time.sleep(10) # Aguarda um pouco para o índice inicializar

        # 2. Carregar e processar os documentos
        print("Carregando documentos do arquivo 'data/info_turismo.txt'...")
        loader = TextLoader("data/info_turismo.txt", encoding="utf-8")
        documents = loader.load()

        print("Dividindo o texto em chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        print(f"Inicializando modelo de embeddings (384 dimensões)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 3. Adicionar os documentos ao novo índice
        print(f"Adicionando {len(docs)} documentos ao índice '{PINECONE_INDEX_NAME}'...")
        PineconeVectorStore.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)

        print("\n✅ Índice criado e populado com sucesso!")

    else:
        print(f"✅ Índice '{PINECONE_INDEX_NAME}' já existe e está pronto para ser usado.")


if __name__ == "__main__":
    setup_pinecone_index()