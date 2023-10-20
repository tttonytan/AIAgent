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
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

RESPONSE_TEMPLATE= """
You are an expert programmer and problem-solver, tasked with answering any question \
about given essay.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation.Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""
REPHRASE_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:
"""


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
def create_reriever_chain():
    # essay_splited = load_and_split_essay(essay_path)
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
    )
    embedding = OpenAIEmbeddings(chunk_size=4000)
    vector_store = Weaviate.from_documents(essay_splited, embedding, client=weaviate_client)
    # vector_store = Weaviate.from_texts(
    #     texts=essay_splited,
    #     embedding=embedding,
    #     client=weaviate_client,
    #     index_name="essay_langchain",
    # )
    retriever = vector_store.as_retriever(search_kwargs=dict(k=6))
    return retriever

# TODO
# 1. 还差把split过的doc 放进 vector store中

# 2. 没有加对应的prompt用这个RetrievalQAWithSourcesChain 链只做到了从原文档中找最接近的，然后返回原文档的内容，做不到根据这个文档回答问题
# 
# 3. 看来最好是自己搭建chain，如果是用官方给的chain的话，很多内容需要重写，比如RetrievalQAWithSourcesChain中的prompt template是不对的
def create_retieval_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k" ,temperature=0.1)
    retriever = create_vector_store_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, max_tokens_limit=8192, return_source_documents=True)
    result = chain({"question":"你是谁"})
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['source_documents'])}")

# def create_retieval_chain():


# def create_chain(
#     llm: BaseLanguageModel,
#     retriever: BaseRetriever,
#     user_history: bool
# ):



if __name__ == "__main__":
#    load_and_split_essay(essay_path)
    # main()






