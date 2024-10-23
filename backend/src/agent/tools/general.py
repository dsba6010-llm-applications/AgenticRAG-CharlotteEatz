import requests
from langchain_core.tools import tool
import json
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
import os
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnableMap    
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from backend.src.agent.rag import load_faiss_store
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

token = os.environ.get("OPENAI_API_KEY")

@tool
def book_a_cab(userquery: str) -> str:
    """Fetches user query to book a cab or taxi to a restaurant and responds back with a confirmation
    
    Args:
        userquery (str): request to book a cab

    Returns:
        (str): Returns a string messgae confirming that the cab has been booked and will arrive as suugested
    """
    return "your taxi has been booked"


@tool
def book_a_table(userquery:str) -> str:
    """Fetches user query to book a table in a restaurant and responds back with a confirmation
    
    Args:
        userquery (str): full name of the user

    Returns:
        (str): Returns a strinf messgae confirming that the restaurant has been booked with date and time stamp
    """
    return "Restaurant table as been booked for requetsed party"
    

@tool
def answer_question(query: str):
    """Fetches and returns information using RAG for user query for general purposes and searches.
    
    Args:
        user_query (str): Query to search for answers

    Returns:
        str: response of user query fetched through RAG
    """
    token = os.environ.get("OPENAI_API_KEY") 
    llm = OpenAI(api_key=token)
    embedding_model = OpenAIEmbeddings()
    path = os.path.dirname(os.path.abspath(__file__)) + "/../"
    db = load_faiss_store(path+"faiss_store/", llm)
    query_embedding = embedding_model.embed_query(query)
    docs = db.max_marginal_relevance_search_with_score_by_vector(
        embedding=query_embedding, k=5, lambda_mult=0.1, fetch_k=30 
    )
    docs.sort(key=lambda x:x[1],reverse=True)
    context = "\n\n".join([doc[0].page_content for doc in docs])
    template = [
        ("system", 
        "Use the following context to answer the question at the end."
        " Process the context by removing any special characters that might be from a Markdown or other files."
        " If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "\nContext:\n"
        "{context}"
    ),
    ("user",
     "Question: {question}"
     "\nHelpful Answer:")
    ]
    rag_prompt_custom = ChatPromptTemplate.from_messages(template)
    chain = (
        rag_prompt_custom
        | llm
    )
    
    response = chain.invoke({
            "context":context,
            "question": query
        })
    
    return response.content
