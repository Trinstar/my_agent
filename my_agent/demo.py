#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent, AgentState
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from my_agent.utils.models import select_model
from my_agent.utils.tools import web_search, sql_factory
from my_agent.utils.rag import RAGModule
from my_agent.utils.config import get_config


async def get_mcp_tools() -> list[BaseTool]:
    """从配置文件加载MCP工具"""
    config = get_config()
    mcp_servers = config.get_mcp_tools_config()
    mcp_client = MultiServerMCPClient(mcp_servers)
    mcp_tools = await mcp_client.get_tools()
    return mcp_tools


async def main():
    config = get_config()
    mcp_tools: list[BaseTool] = await get_mcp_tools()

    # 从配置文件获取数据库URI
    database_config = config.get_database_config()
    my_db = SQLDatabase.from_uri(database_config['uri'])
    execute_sql: callable = sql_factory(my_db)

    my_rag = RAGModule(rag_config=config.get_rag_config())
    my_rag.prepare_data()
    my_rag.load_index()

    @tool
    def rag_knowledge_search(query: str) -> str:
        """这是关于菜谱的知识库查询工具，如果用户提问涉及到食材以及如何做饭，做什么菜，使用这个工具查询"""
        docs = my_rag.index_similarity_search(query)
        combined_content = "\n".join([doc.page_content for doc in docs])
        return combined_content

    # tools list
    tools = [web_search, execute_sql, rag_knowledge_search] + mcp_tools

    # 系统提示词
    SYSTEM_PROMPT = """你是有用的助手，你需要遵循以下要求：
    1. 博客网站的网站是https://www.trinstar.cn，提问网站内容时直接静默使用web_search工具查询；
    2. 用户提问涉及到需要数据库查询，使用excute_sql工具；
    3. 不许编造内容，准确回答，无法把握的内容就诚实；
    4. 回答要简洁明了，避免冗长；
    5. 在回答问题时，根据用户的语言调整回答的语言，保证一致；
    6. 如果数据库查询结果错误，重新执行查询，直到结果正确为止；
    7. 禁止使用增加、删除、修改数据库的操作，只允许查询；
    8. 如果用户提问涉及到食材以及如何做饭，做什么菜，使用rag_knowledge_search工具查询；
    9. 使用rag_knowledge_search工具查询时，告诉用户使用的文档名字和id。
    """

    chat_model = select_model("deepseek-chat")
    blog_agent = create_agent(
        model=chat_model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=InMemorySaver(),
    )

    print("stream模型请选择1，invoke模式请选择2")
    chat_mode = int(input())
    print()

    if chat_mode == 2:
        while True:
            user_input = input("User: ")
            
            if user_input == "0":
                break

            messages = [
                HumanMessage(content=user_input)
            ]
            print()
            print("AI: ", end='', flush=True)
            res = await blog_agent.ainvoke(
                {"messages": messages},
                {"thread_id": "default_thread"},
            )
            for chunk in res["messages"]:
                chunk.pretty_print()
    elif chat_mode == 1:
        while True:
            user_input = input("User: ")

            if user_input == "0":
                break
            
            messages = [
                HumanMessage(content=user_input)
            ]

            print("AI: ", end='', flush=True)
            async for chunk, metadata in blog_agent.astream(
                {"messages": messages},
                {"thread_id": "default_thread"},
                stream_mode="messages"
            ):
                if chunk.content and metadata["langgraph_node"] != "tools":
                    print(chunk.content, end='', flush=True)
            print()

if __name__ == "__main__":
    asyncio.run(main())
