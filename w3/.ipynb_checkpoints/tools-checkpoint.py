import numpy as np
import requests
import os
from dotenv import load_dotenv 
from langchain.tools import tool
from typing import Literal
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
#define tools

@tool
def quadratic_calculator(a:float, b:float, c:float) -> str:
    """
    Solves the quadratic equation
    """
    if a == 0: return "DNE";

    #test discriminant 
    discriminant = b**2-4*a*c
    
    if discriminant > 0:
        ans1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        ans2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        return f"({str(ans1)}, {str(ans2)})"
    elif discriminant == 0: 
        ans1 = -b/(2*a)
        return str(ans1)
    else:
        ans1a = -b/(2*a)
        ans1b = np.sqrt(np.abs(b**2 - 4*a*c))/(2*a)
        return f"({str(ans1a)}-{str(ans1b)}i, {str(ans1a)}+{str(ans1b)}i)"


@tool
def fib_list(n: int) -> list[int]:
    """
    Generate the first n fibonacci numbers
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fibs = [0, 1]
    while len(fibs) < n:
        next_fib = fibs[-1] + fibs[-2]
        fibs.append(next_fib)
    return fibs

CurrencyCode = Literal[
    "AUD", 
    "BGN", 
    "BRL", 
    "CAD", 
    "CHF",
    "CNY",
    "CZK",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "HRK",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "ISK",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PHP",
    "PLN",
    "RON",
    "RUB",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "USD",
    "ZAR"
]

@tool
def convert_currency(amount: float, fromCurrency: CurrencyCode, toCurrency: Literal["USD", "CAD", "EUR"]) -> float:
    """
    Convert amount from fromCurrency to either USD, EUR, or CAD.
    """
    base_url = "https://api.freecurrencyapi.com/v1/latest"
    params = {
        "apikey": "fca_live_ntSZWuM334ysr2yg3CZDYslXpjAL62xPNWXdUe53",
        "base_currency": fromCurrency,
        "currencies": toCurrency
    }

    response = requests.get(base_url, params=params).json()
    total = amount * response["data"][toCurrency]
    print(total)
    return total

@tool
def query_library(query: str) -> str:
    """
    Search the user's personal ebook library for information.
    Use this for questions about specific books, authors, history, or science 
    contained in the library.
    """

    embeddings = OpenAIEmbeddings()
    #intiialize LLM
    llm = ChatOpenAI(
        model="gpt-5-nano"
    )
    
    #create chroma
    vectorstore = Chroma(
        collection_name="ebook_texts", 
        embedding_function=embeddings,
        persist_directory="/outputs/chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 50
        }
    )
    
    system_prompt = """
        You are a strict RAG model. 
        Only answer using the provided context.
        Do not show literal \\n characters — output real line breaks.
        If the answer is not contained in the context, respond with:
        "I don't know — no supporting text found."
        """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict RAG model. 
        Only answer using the provided context.
        Do not show literal \\n characters — output real line breaks.
        If the answer is not contained in the context, respond with:
        "I don't know — no supporting text found."
        """),
        ("human", """
        Question:
        {query}
        Context:
        {context}
        """)
    ])
    
    def format_docs(docs):
        return "\n\n".join(
            [f"Source: {d.metadata}\nContent: {d.page_content}"
             for d in docs
            ])

    chain = (
        {
            "query": RunnablePassthrough(), 
            "context": retriever | RunnableLambda(format_docs)}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

@tool
def search_web(query: str) -> str:
    """
    Performs a web search on a certain topic. 
    """
    
    #intiialize LLM
    llm = ChatOpenAI(
        model="gpt-5-nano"
    )
    
    load_dotenv(dotenv_path="/app/.env")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        
    tavily_client = TavilyClient()
    
    search_result = tavily_client.search(query)
    
    prompt = ChatPromptTemplate.from_messages([ 
            ("system", """
            You are a helpful assistant that will help answer the user's search query with the given context. Please 
            try to answer as best as possible. Be concise unless the user requests you to be more verbose. Keep 
            your responses under 200 words.
            """), 
            ("user", """
                User Search Query:
                {query}
                ---
                Summary:
                {summary}
                """
            ) 
    ])
    #dont forget this step 
    
    chain = (
        {
            "query": RunnablePassthrough(),
            "summary": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    resp = chain.invoke({"query": query, "summary": search_result})
    return resp



