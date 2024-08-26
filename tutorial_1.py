from langchain_openai import AzureChatOpenAI    
import os
from dotenv import load_dotenv

load_dotenv()



llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv('API_VERSION'),
    temperature=0.0,
    max_tokens=900,
    timeout=None,
    api_key=os.getenv('API_KEY'),
    azure_endpoint=os.getenv('AZURE_ENDPOINT')
)


from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools.tavily_search import TavilySearchResults




class State(TypedDict):
    messages: Annotated[list, add_messages]
    

graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot",chatbot)


graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot", END)


graph = graph_builder.compile()



# this line of code just build a simple chat system using langgraphh
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": ("user", user_input)}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)