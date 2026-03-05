import asyncio
from langchain import messages
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langchain_community.embeddings import DashScopeEmbeddings
import uuid
from service import memory_manager
from service.memory_manager import get_memory_manager
from pydantic import BaseModel

llm=ChatTongyi(model="qwen-max")

class Memories(BaseModel):
    memories:list[str]=[]

def printt(obj):
    for _ in range(15):print("-",end="")
    print("")
    print(obj)
    for _ in range(15):print("-",end="")
    print("")

def get_full_message(state:MessagesState):
    full_message:list[str]=[]
    for message in state["messages"]:
        message_type=message.__class__.__name__
        full_message.append(f"""
        message_type:{message_type}\t
        message_content:{message.content}\t
        """)
    return full_message

async def save_new_memory(state:MessagesState):
    prompt:str=f"""
    你是一个"个人邮件智能体"中的聊天机器人,
    这个智能体具有解析分类邮件，自动起草回复，对话等功能。
    作为聊天机器人，请分析最近一次的HumanMessage，
    如果有任何信息可能对智能体有用，
    注意，不要错过任何可能有用的信息，宁多不少
    返回其列表作为记忆（确保内容一条条简明清晰）。
    没有记忆返回空memories列表即可。
    消息记录：{get_full_message(state)}
    """
    new_mems=await llm.with_structured_output(Memories).ainvoke(prompt)
    mem_manager=get_memory_manager()
    for mem in new_mems.memories:
        await mem_manager.store_memory("global_agent",mem)
    return {"messages":AIMessage(content=f"新保存的记忆:{new_mems.memories}")}


async def chat(state:MessagesState):
    #printt(f"Received Messages:{state["messages"]}")
    reply=await llm.ainvoke(get_full_message(state))
    
    # ... Analyze conversation and create a new memory
    
    return {"messages":[reply]}

store = InMemoryStore()
checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)

builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store,checkpointer=checkpointer)
config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "1"
    }
}

async def chatbot():
    while True:
        user_input=input("Send to AI:")
        input_state = {"messages": [HumanMessage(content=user_input)]}
        async for update in graph.astream(
            input_state,
            config,
            stream_mode="updates",
        ):
            ai_reply=update["chat"]["messages"][0].content
            print("AI Reply:\n"+ai_reply)


async def test():
    input_state = {"messages": [HumanMessage(
        content="我不喜欢吃西红柿")]}
    printt(await save_new_memory(input_state))

if __name__=="__main__":
    asyncio.run(test())