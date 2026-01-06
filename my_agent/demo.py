#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path
from rich import print as rprint
from dataclasses import dataclass

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent, AgentState
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import AsyncSqliteStore
from langchain.agents.middleware import FilesystemFileSearchMiddleware, ShellToolMiddleware, HostExecutionPolicy

from my_agent.utils.models import select_model
from my_agent.utils.tools import web_search, sql_factory, send_email
from my_agent.utils.rag import RAGModule
from my_agent.utils.config import get_config
from my_agent.utils.clean_txt import clean_text
from my_agent.utils.structured_output import WeatherOutput
from my_agent.utils.middleware import ReadStoreMiddleware, SaveStoreMiddleware


@dataclass
class Context:
    user_id: str


async def get_mcp_tools() -> list[BaseTool]:
    """从配置文件加载MCP工具"""
    config = get_config()
    mcp_servers = config.get_mcp_tools_config()
    mcp_tools = []
    for server_name, server_config in mcp_servers.items():
        single_mcp_server = {server_name: server_config}
        single_mcp_client = MultiServerMCPClient(single_mcp_server)
        try:
            single_mcp_tools = await single_mcp_client.get_tools()
            mcp_tools.extend(single_mcp_tools)
        except ExceptionGroup as eg:
            print(f"Failed to load MCP server '{server_name}':")
            for exc in eg.exceptions:
                print(f"  - {type(exc).__name__}: {exc}")
        except Exception as e:
            print(
                f"Failed to load MCP server '{server_name}': {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    return mcp_tools


async def main():
    config = get_config()
    # async load MCP tools
    mcp_tools = await get_mcp_tools()

    # sqlite_ckpt_saver (async)
    memory_saver_db_dir = Path("./data/Checkpoints")
    memory_saver_db_dir.mkdir(parents=True, exist_ok=True)
    memory_saver_db_path = memory_saver_db_dir / "sqlite_saver.db"

    # sqlite_store (async)
    store_db_dir = Path("./data/Stores")
    store_db_dir.mkdir(parents=True, exist_ok=True)
    store_db_path = store_db_dir / "sqlite_store.db"

    async with AsyncSqliteSaver.from_conn_string(str(memory_saver_db_path)) as sqlite_saver, \
            AsyncSqliteStore.from_conn_string(str(store_db_path)) as sqlite_store:

        # load database config
        database_config = config.get_database_config()
        my_db = SQLDatabase.from_uri(database_config['uri'])
        execute_sql: callable = sql_factory(my_db)

        # my_rag = RAGModule(rag_config=config.get_rag_config())
        # my_rag.prepare_data()
        # my_rag.load_index()
        #
        # @tool
        # def rag_knowledge_search(query: str) -> str:
        #     """这是关于菜谱的知识库查询工具，如果用户提问涉及到食材以及如何做饭，做什么菜，使用这个工具查询"""
        #     docs = my_rag.index_similarity_search(query)
        #     combined_content = "\n".join([doc.page_content for doc in docs])
        #     return combined_content

        # tools list
        tools = [web_search, execute_sql, send_email] + mcp_tools

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
        user_config = {"configurable": {"thread_id": "blog_agent_1"}, }
        user_context = Context(user_id="xlp")

        chat_model = select_model("deepseek-chat")
        blog_agent = create_agent(
            model=chat_model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            middleware=[
                ReadStoreMiddleware(),
                FilesystemFileSearchMiddleware(
                    root_path="/mnt/d/LangChain/my-agent",
                    use_ripgrep=True,
                    max_file_size_mb=5,
                ),
                SaveStoreMiddleware(),
            ],
            checkpointer=sqlite_saver,
            store=sqlite_store,
            context_schema=Context,
        )

        print(">stream模型请选择1，invoke模式请选择2")
        chat_mode = int(input())
        print()

        if chat_mode == 2:
            while True:
                user_input = input("User: ")

                if user_input == "0":
                    break

                user_input = clean_text(user_input)
                messages = [
                    HumanMessage(content=user_input)
                ]
                print()
                print("AI: ", end='', flush=True)
                res = await blog_agent.ainvoke(
                    {"messages": messages},
                    user_config,
                    context=user_context,
                )
                for chunk in res["messages"]:
                    chunk.pretty_print()

        elif chat_mode == 1:
            while True:
                user_input = input("User: ")
                user_input = clean_text(user_input)
                if user_input == "0":
                    break

                messages = [
                    HumanMessage(content=user_input)
                ]

                async for step in blog_agent.astream(
                    {"messages": messages},
                    user_config,
                    context=user_context,
                ):
                    for update in step.values():
                        for message in update.get("messages", []):
                            message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
