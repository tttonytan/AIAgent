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

weaviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)
splited_essay = load_and_split_essay(essay_path)
embedding = OpenAIEmbeddings(chunk_size=4000)

# 下面是使用一个weaviate 向量数据库对方法
# vector_store = Weaviate(
#     client=weaviate_client,
#     index_name="SPESC",  # 指定Weaviate中的索引名称，用于存储和检索向量数据。
#     text_key="text",
#     embedding=OpenAIEmbeddings(chunk_size=200),
#     by_text=False,
#     # attributes=["source", "title"]
# )

# 这是通过document和embedding来初始化一个向量数据库
vector_store = Weaviate.from_documents(splited_essay, embedding, client=weaviate_client, by_text=False, index_name="SPESC")
# print(type(weaviate_client))

