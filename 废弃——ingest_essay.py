# 加载论文pdf
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
from langchain.embeddings import OpenAIEmbeddings
import weaviate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseRetriever
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
essay_path = "/Users/tanhao/AI项目/LLM-projects/AIAgent_Projects/essay_read-Agent/essay.pdf"
WEAVIATE_URL = os.environ["WEAVIATE_URL2"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY2"]


def load_and_split_essay(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=200
    )
    essay_splited = text_spliter.split_documents(pages)
        
    # print(type(essay_splited))

    # for i, page in enumerate(essay_splited):
    #     print("!!!!!", i, page, "\n")
    return essay_splited

# 创建vectorstore
# 创建retriever chain
def ingest_essay(essay_path):
    essay_splited = load_and_split_essay(essay_path)
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
    )
    embedding = OpenAIEmbeddings(chunk_size=4000)
    vector_store = Weaviate.from_documents(essay_splited, embedding, client=weaviate_client, index_name="essay")
    # vector_store = Weaviate.from_texts(
    #     texts=essay_splited,
    #     embedding=embedding,
    #     client=weaviate_client,
    #     index_name="essay_langchain",
    # )
    retriever = vector_store.as_retriever(search_kwargs=dict(k=6))
    return retriever


if "__name__" == "__main__":
    ingest_essay()
