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
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableMap, Runnable
from typing import Sequence, Optional, List, Dict
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from pydantic import BaseModel



from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

essay_path = "/Users/tanhao/AI项目/LLM-projects/AIAgent_Projects/essay_read-Agent/essay.pdf"
WEAVIATE_URL = os.environ["WEAVIATE_URL2"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY2"]

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

def get_retriever():
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
    )
    weaviate_vector_store = Weaviate(
        client=weaviate_client,
        index_name="essay",
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=8096),
        by_text=False
    )
    return weaviate_vector_store.as_retriever(search_kwargs=dict(k=3))

def create_retriever_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    user_history: bool
):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    if not user_history:
        initial_chain = itemgetter("question") | retriever
        return initial_chain
    else:
        condense_quesion_chain = (
            {"question": itemgetter("question")}
            |CONDENSE_QUESTION_PROMPT
            |llm
            |StrOutputParser()
        ).with_config(
            run_name="CondenseQuestion"
        )
        retriever_chain = condense_quesion_chain | retriever
        return retriever_chain

def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(llm: BaseLanguageModel, retriever: BaseRetriever, use_chat_history):
    retriever_chain = create_retriever_chain(llm, retriever, use_chat_history)
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")

        ]
    )
    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(run_name="GenerateResponse")
    return _context | response_synthesizer


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]]
    conversation_id: Optional[str]


def chat(request: ChatRequest): 
    question = request.message
    chat_history = request.history or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    # metadata = {
    #     "conversation_id": request.conversation_id
    # }

    llm = ChatOpenAI(
        model="gpt-4",
        streaming=False,
        temperature=0
    )

    retriever = get_retriever()
    answer_chain = create_chain(
        llm=llm,
        retriever=retriever,
        use_chat_history=bool(converted_chat_history)
    )
    answer = answer_chain.invoke({
            "question": question,
            "chat_history": converted_chat_history,
        })
    return answer


# TODO
# 1. 还差把split过的doc 放进 vector store中

# 2. 没有加对应的prompt用这个RetrievalQAWithSourcesChain 链只做到了从原文档中找最接近的，然后返回原文档的内容，做不到根据这个文档回答问题
# 
# 3. 看来最好是自己搭建chain，如果是用官方给的chain的话，很多内容需要重写，比如RetrievalQAWithSourcesChain中的prompt template是不对的
# def create_retieval_chain():
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k" ,temperature=0.1)
#     retriever = create_vector_store_retriever()
#     chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, max_tokens_limit=8192, return_source_documents=True)
#     result = chain({"question":"你是谁"})
#     print(f"Answer: {result['answer']}")
#     print(f"Sources: {len(result['source_documents'])}")

# def create_retieval_chain():


# def create_chain(
#     llm: BaseLanguageModel,
#     retriever: BaseRetriever,
#     user_history: bool
# ):



if __name__ == "__main__":
    user_input = ChatRequest(
        message="如何创建DSL",
        history=[],
        conversation_id="0"
    )
    a = chat(user_input)
    print(a)






