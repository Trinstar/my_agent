#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime
from pathlib import Path
import pytz
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
from my_agent.utils.tool_agents.flight_agent import flight_subagent
from my_agent.utils.tool_agents.hotel_agent import hotel_subagent
from my_agent.utils.tool_agents.car_agent import car_subagent
from langgraph.types import Command
from langgraph.runtime import get_runtime


class SubAgentContext:
    flight_agent : AgentState = None
    hotel_agent: AgentState = None
    car_agent: AgentState = None
    

@tool(parse_docstring=True)
def call_flight_subagent(query: str, resume: bool = False) -> str:
    """call the flight subagent to handle flight-related queries.

    Args:
        query (str): 用户的查询内容。
        resume (bool): 是否继续之前的对话上下文，默认为False。当需要权限确认时使用。然后可以再次调用该程序
    
    Returns:
        str: 子代理的响应结果。
    """
    runtime = get_runtime(SubAgentContext)
    flight_agent = runtime.context.flight_agent
    # flight_agent = flight_subagent(user_id)
    user_config = {"configurable": {"thread_id": "flight_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]

    if resume:
        result = flight_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        user_config,
    )
        return result['messages'][-1].content
    result = flight_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        return "此操作需要用户权限许可，请你询问用户的选择，'approve' or 'reject'. approve代表同意，reject代表拒绝。然后将用户的选择传递给该工具的resume参数再次调用该工具。"
    
    return result['messages'][-1].content


@tool(parse_docstring=True)
def call_hotel_subagent(query: str, resume: bool = False) -> str:
    """call the hotel subagent to handle hotel-related queries.

    Args:
        query (str): 用户的查询内容。
        resume (bool): 是否继续之前的对话上下文，默认为False。当需要权限确认时使用。然后可以再次调用该程序
    
    Returns:
        str: 子代理的响应结果。
    """
    runtime = get_runtime(SubAgentContext)
    hotel_agent = runtime.context.hotel_agent
    # flight_agent = flight_subagent(user_id)
    user_config = {"configurable": {"thread_id": "hotel_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]

    if resume:
        result = hotel_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        user_config,
    )
        return result['messages'][-1].content
    
    result = hotel_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        return "此操作需要用户权限许可，请你询问用户的选择，'approve' or 'reject'. approve代表同意，reject代表拒绝。然后将用户的选择传递给该工具的resume参数再次调用该工具。"
    
    return result['messages'][-1].content


@tool(parse_docstring=True)
def call_car_subagent(query: str, resume: bool = False) -> str:
    """call the car subagent to handle car-related queries.

    Args:
        query (str): 用户的查询内容。
        resume (bool): 是否继续之前的对话上下文，默认为False。当需要权限确认时使用。然后可以再次调用该程序
    
    Returns:
        str: 子代理的响应结果。
    """
    runtime = get_runtime(SubAgentContext)
    car_agent = runtime.context.car_agent
    user_config = {"configurable": {"thread_id": "car_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]


    if resume:        
        result = car_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        user_config,
    )
        return result['messages'][-1].content
    
    result = car_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        return "此操作需要用户权限许可，请你询问用户的选择，'approve' or 'reject'. approve代表同意，reject代表拒绝。然后将用户的选择传递给该工具的resume参数再次调用该工具。"

    return result['messages'][-1].content

async def main():
    config = get_config()

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

        tools = [call_flight_subagent, call_hotel_subagent, call_car_subagent]

        # 系统提示词
        time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")
        SYSTEM_PROMPT = f"""你是一位专业的旅游助理，负责协调航班、酒店和租车的查询与预订服务。

## 核心原则
1. **真实准确**：严禁编造信息，只基于工具返回的真实数据回答。不确定的内容必须诚实说明。
2. **语言一致**：根据用户使用的语言进行回复，保持对话语言的一致性。
3. **工具优先**：优先使用工具获取信息，避免凭空回答。

## 工作流程
### 任务处理顺序
- **顺序执行**：每次只处理一个主要需求（航班/酒店/租车），确保该需求完全完成后再处理下一个。
- **完成标志**：当用户明确确认预订成功、取消成功或修改成功后，才视为该需求完成。

### 子代理调用规则
**航班服务** - 使用 `call_flight_subagent`：
**酒店服务** - 使用 `call_hotel_subagent`：
**租车服务** - 使用 `call_car_subagent`：
 

### 权限确认处理
当子代理返回需要权限确认的提示时：
1. **识别提示**：子代理会返回包含 "approve or reject" 的提示信息。
2. **询问用户**：向用户清晰说明需要确认的操作，询问是否同意（approve）或拒绝（reject）。
3. **传递决策**：
   - 如果用户同意（说 "同意"/"好的"/"approve"等），调用对应子代理工具并设置 `resume=True`
   - 如果用户拒绝（说 "拒绝"/"不"/"reject"等），告知用户操作已取消，询问是否需要其他帮助

当前时间: {time}
        """

        user_config = {"configurable": {"thread_id": "xlp"}, }
        user_context = SubAgentContext()
        user_context.flight_agent = flight_subagent()
        user_context.hotel_agent = hotel_subagent()
        user_context.car_agent = car_subagent()

        chat_model = select_model("deepseek-chat")
        blog_agent = create_agent(
            model=chat_model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            checkpointer=sqlite_saver,
            context_schema=SubAgentContext
        )

        print(">stream模型请选择1，invoke模式请选择2")
        chat_mode = int(input())
        print()

        if chat_mode == 2:
            while True:
                user_input = input("\033[92mUser: \033[0m")

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
                user_input = input("\033[92mUser: \033[0m")
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
