#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime
from pathlib import Path
import pytz
from rich import print as rprint
from rich.prompt import Confirm
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
from langgraph.types import Command
from langgraph.runtime import get_runtime

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
from my_agent.utils.tool_agents.trip_agent import excursion_subagent



def handle_permission_confirmation(agent, user_config: dict, interrupt_list: list, agent_name: str):
    """处理权限确认的通用函数
    
    Args:
        agent: 子代理实例
        user_config: 用户配置
        interrupt_list: interrupt 列表
        agent_name: 代理名称（用于显示）
    
    Returns:
        执行结果消息
    """
    # 提取操作信息 - Interrupt 对象
    if interrupt_list:
        interrupt_obj = interrupt_list[0]  # Interrupt 对象
        action_requests = interrupt_obj.value.get('action_requests', [])
        
        if action_requests:
            action_info = action_requests[0]
            tool_name = action_info.get('name', '未知工具')
            tool_args = action_info.get('args', {})
            
            rprint(f"\n[yellow]⚠️  {agent_name}需要权限确认[/yellow]")
            rprint(f"[cyan]操作类型:[/cyan] {tool_name}")
            
            # 显示具体参数
            if tool_args:
                rprint("[cyan]操作详情:[/cyan]")
                for key, value in tool_args.items():
                    rprint(f"  • {key}: {value}")
        else:
            rprint(f"\n[yellow]⚠️  {agent_name}需要权限确认[/yellow]")
    else:
        rprint(f"\n[yellow]⚠️  {agent_name}需要权限确认[/yellow]")
    
    approval = Confirm.ask("\n是否同意执行此操作？", default=False)
    
    if approval:
        rprint("[green]✓ 已批准操作[/green]\n")
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            user_config,
        )
        # 找到最后一条 AIMessage，而不是 ToolMessage
        for msg in reversed(result['messages']):
            if hasattr(msg, 'type') and msg.type == 'ai':
                return msg.content
        # 如果没找到 AIMessage，返回最后一条消息
        return result['messages'][-1].content
    else:
        rprint("[red]✗ 已拒绝操作[/red]\n")
        agent.invoke(
            Command(resume={"decisions": [{"type": "reject"}]}),
            user_config,
        )
        return "用户在服务中取消了该操作。"


class SubAgentContext:
    flight_agent : AgentState = None
    hotel_agent: AgentState = None
    car_agent: AgentState = None
    excursion_agent: AgentState = None
    

@tool(parse_docstring=True)
def call_flight_subagent(query: str) -> str:
    """call the flight subagent to handle flight-related queries.

    Args:
        query (str): 用户的查询内容。
    
    Returns:
        str: 子代理的响应结果。权限确认已自动处理。
    """
    runtime = get_runtime(SubAgentContext)
    flight_agent = runtime.context.flight_agent
    # flight_agent = flight_subagent(user_id)
    user_config = {"configurable": {"thread_id": "flight_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]

    result = flight_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        interrupt_list = result.get('__interrupt__', [])
        return handle_permission_confirmation(
            agent=flight_agent,
            user_config=user_config,
            interrupt_list=interrupt_list,
            agent_name="航班子代理"
        )
    return result['messages'][-1].content


@tool(parse_docstring=True)
def call_hotel_subagent(query: str) -> str:
    """call the hotel subagent to handle hotel-related queries.

    Args:
        query (str): 用户的查询内容。
    
    Returns:
        str: 子代理的响应结果。权限确认已自动处理。
    """
    runtime = get_runtime(SubAgentContext)
    hotel_agent = runtime.context.hotel_agent
    user_config = {"configurable": {"thread_id": "hotel_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]
    
    result = hotel_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        interrupt_list = result.get('__interrupt__', [])
        return handle_permission_confirmation(
            agent=hotel_agent,
            user_config=user_config,
            interrupt_list=interrupt_list,
            agent_name="酒店子代理"
        )
    
    return result['messages'][-1].content


@tool(parse_docstring=True)
def call_car_subagent(query: str) -> str:
    """call the car subagent to handle car-related queries.

    Args:
        query (str): 用户的查询内容。
    
    Returns:
        str: 子代理的响应结果。权限确认已自动处理。
    """
    runtime = get_runtime(SubAgentContext)
    car_agent = runtime.context.car_agent
    user_config = {"configurable": {"thread_id": "car_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]

    result = car_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        interrupt_list = result.get('__interrupt__', [])
        return handle_permission_confirmation(
            agent=car_agent,
            user_config=user_config,
            interrupt_list=interrupt_list,
            agent_name="租车子代理"
        )

    return result['messages'][-1].content

def call_excursion_subagent(query: str) -> str:
    """call the excursion subagent to handle excursion-related queries.

    Args:
        query (str): 用户的查询内容。
    
    Returns:
        str: 子代理的响应结果。权限确认已自动处理。
    """
    runtime = get_runtime(SubAgentContext)
    excursion_agent = runtime.context.excursion_agent
    user_config = {"configurable": {"thread_id": "excursion_agent"}, }

    messages = [
        HumanMessage(content=query)
    ]

    result = excursion_agent.invoke(
        {"messages": messages},
        user_config,
    )

    if result.get('__interrupt__'):
        interrupt_list = result.get('__interrupt__', [])
        return handle_permission_confirmation(
            agent=excursion_agent,
            user_config=user_config,
            interrupt_list=interrupt_list,
            agent_name="旅行推荐子代理"
        )

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

        tools = [call_flight_subagent, call_hotel_subagent, call_car_subagent, call_excursion_subagent]

        # 系统提示词
        time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")
        SYSTEM_PROMPT = f"""你是一位专业的旅游助理，负责协调航班、酒店和租车的查询与预订服务。

## 核心原则
1. **真实准确**：严禁编造信息，只基于工具返回的真实数据回答。不确定的内容必须诚实说明。
2. **工具优先**：优先使用工具（子代理）获取信息，避免凭空回答，使用时候保证语言严谨，准确调用，用专业化术语不要口语化表达。
3. **对话为主**：和用户对话的时候，以对话形式与用户交流，确保理解用户需求。有礼貌且专业。

## 工作流程
### 任务处理顺序
- **顺序执行**：每次只处理一个主要需求（航班/酒店/租车），确保该需求完全完成后再处理下一个。
- **完成标志**：当用户明确确认预订成功、取消成功或修改成功后，才视为该需求完成。

### 子代理（工具）调用规则
**航班服务** - 使用 `call_flight_subagent`。
**酒店服务** - 使用 `call_hotel_subagent`。
**租车服务** - 使用 `call_car_subagent`。
**旅行推荐服务** - 使用 `call_excursion_subagent`。

### 权限确认
在调用子代理时，如果接收到“用户在服务中取消了该操作”的结果，说明是用户在过程中拒绝了操作不授予权限，不是其他的原因。

当前时间: {time}
        """

        user_config = {"configurable": {"thread_id": "xlp"}, }
        user_context = SubAgentContext()
        user_context.flight_agent = flight_subagent()
        user_context.hotel_agent = hotel_subagent()
        user_context.car_agent = car_subagent()
        user_context.excursion_agent = excursion_subagent()
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

                # stream mode = "updates"
                # async for step in blog_agent.astream(
                #     {"messages": messages},
                #     user_config,
                #     context=user_context,
                # ):
                #     for update in step.values():
                #         for message in update.get("messages", []):
                #             message.pretty_print()

                print("AI: ")
                async for data, metadata in blog_agent.astream(
                    {"messages": messages},
                    user_config,
                    context=user_context,
                    stream_mode="messages",
                ):
                    if isinstance(data, AIMessage) and 'tools' not in metadata.get('checkpoint_ns'): # [TODO] nb!
                        print(data.content, end="", flush=True)
                print()

if __name__ == "__main__":
    asyncio.run(main())
