#----------------------------------------------------------Imports---------------------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from pydantic import Field
from langgraph.graph.message import add_messages
from langchain_community.utilities import GoogleSerperAPIWrapper 
import sqlite3 
from datetime import datetime
import pytz
from langgraph.checkpoint.sqlite import SqliteSaver 
from langchain_community.tools import WikipediaQueryRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from langsmith import traceable
from dotenv import load_dotenv 
from rag import search
import os

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_KEY")


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b", 
    task="text-generation",
    huggingfacehub_api_token=api_key

)

model = ChatHuggingFace(llm=llm)

#----------------------------------------------------------------Tools-------------------------------------------------------------------------------

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
)
youtube_tool = YouTubeSearchTool(num_results=3)

serper_search_wrapper = GoogleSerperAPIWrapper(k=5)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=3, 
    doc_content_chars_max=4000 
)
arxiv_search = ArxivQueryRun(api_wrapper=arxiv_wrapper)

@tool
@traceable(name='serper')
def serper_web_search(query: str) -> str:
    """
    Search the internet for real-time information, current events, and up-to-date facts
    using Google index. Use this tool for any question requiring external, current knowledge.
    """
    return serper_search_wrapper.run(query)

@tool
@traceable(name='wiki')
def wikipedia_search(query: str) -> str:
    """
    Search the Wikipedia knowledge base for factual, authoritative, and general information 
    about history, biographies, science, and definitions. Use this for non-current event facts.
    """
    return wikipedia_tool.run(query)


@tool
@traceable(name='calculator')
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
@traceable(name='get_current_date')
def get_current_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

@tool
@traceable(name='get_current_time')
def get_current_ist_time() -> str:
    """Return the current date and time in IST (India time) cleanly."""
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    return now_ist.strftime("%Y-%m-%d %H:%M:%S")

@tool
@traceable(name='stock_price')
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
@traceable(name='retrieve_docs')
def retrieve_docs(query: str):
    """Retrieve relevant documents from the FAISS vector store."""
    results = search(query)
    return "\n\n".join([r.page_content for r in results]) if results else "No documents found."



tools = [serper_web_search, youtube_tool, arxiv_search, wikipedia_search, get_stock_price, calculator, get_current_date,get_current_ist_time]
model_with_tools = model.bind_tools(tools=tools)

#------------------------------------------------------------------State----------------------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 
    
def chat_node(state:ChatState) -> ChatState:
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {'messages':response}

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False) 

checkpointer = SqliteSaver(conn=conn) 

tool_node = ToolNode(tools)

#-----------------------------------------------------------------Graph-----------------------------------------------------------------------------

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')  

chatbots = graph.compile(checkpointer=checkpointer) 

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)  