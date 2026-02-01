#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, List, Dict
import pytz
from datetime import date, datetime

from sqlite3 import connect, Cursor
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from my_agent.utils.clean_txt import clean_text
from my_agent.utils.models import select_model

db = "./data/DB/travel.db"

@tool(parse_docstring=True)
def search_trip_recommendations(
        location: Optional[str] = None,
        name: Optional[str] = None,
        keywords: Optional[str] = None,
) -> List[dict]:
    """根据位置、名称和关键词搜索旅行推荐。

    Args:
        location (Optional[str]): 旅行推荐的位置。默认为None。
        name (Optional[str]): 旅行推荐的名称。默认为None。
        keywords (Optional[str]): 关联到旅行推荐的关键词。默认为None。

    Returns:
        list[dict]: 包含匹配搜索条件的旅行推荐字典列表。
    """
    conn = connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool(parse_docstring=True)
def search_excursions(
    user_id: str,
    location: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[dict]:
    """根据乘客ID和筛选条件搜索已预订的旅行项目。

    Args:
        user_id (str): 乘客/用户的ID。
        location (Optional[str]): 旅行地点。默认为None。
        status (Optional[str]): 订单状态（Booked/Cancelled/Completed）。默认为None。
        start_date (Optional[str]): 开始日期，格式：YYYY-MM-DD。默认为None。
        end_date (Optional[str]): 结束日期，格式：YYYY-MM-DD。默认为None。

    Returns:
        list[dict]: 包含该乘客已预订的旅行项目字典列表，按日期排序。
    """
    conn = connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM excursions WHERE passenger_id = ?"
    params = [user_id]

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if start_date:
        query += " AND excursion_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND excursion_date <= ?"
        params.append(end_date)
    
    # 按日期排序，最近的在前
    query += " ORDER BY excursion_date DESC"
    
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool(parse_docstring=True)
def book_excursion(
    passenger_id: str,
    recommendation_id: int,
    excursion_date: str,
    duration_hours: int = 4,
    price: float = 0.0
) -> str:
    """通过推荐ID预订一次旅行项目，在excursions表中创建订单。

    Args:
        passenger_id (str): 乘客/用户的ID。
        recommendation_id (int): 旅行推荐的ID。
        excursion_date (str): 旅行日期，格式：YYYY-MM-DD。
        duration_hours (int): 旅行时长（小时）。默认为4小时。
        price (float): 旅行价格。默认为0.0。

    Returns:
        str: 表明旅行项目是否成功预订的消息。

    """
    conn = connect(db)
    cursor = conn.cursor()

    # 先查询推荐信息
    cursor.execute(
        "SELECT name, location FROM trip_recommendations WHERE id = ?",
        (recommendation_id,)
    )
    recommendation = cursor.fetchone()
    
    if not recommendation:
        conn.close()
        return f"未找到ID为 {recommendation_id} 的旅行推荐。"
    
    excursion_name, location = recommendation
    
    # 在excursions表中创建订单
    cursor.execute(
        """INSERT INTO excursions 
        (passenger_id, excursion_name, location, excursion_date, duration_hours, price, status) 
        VALUES (?, ?, ?, ?, ?, ?, 'Booked')""",
        (passenger_id, excursion_name, location, excursion_date, duration_hours, price)
    )
    conn.commit()
    excursion_id = cursor.lastrowid
    
    conn.close()
    return f"旅行项目 '{excursion_name}' 已成功预订！订单ID: {excursion_id}，日期: {excursion_date}。"
    

@tool(parse_docstring=True)
def cancel_excursion(excursion_id: int) -> str:
    """根据订单ID取消旅行项目。

    Args:
        excursion_id (int): 要取消的旅行订单的ID。

    Returns:
        str: 表明旅行订单是否成功取消的消息。
    """
    conn = connect(db)
    cursor = conn.cursor()

    # 将status设置为Cancelled来表示取消订单
    cursor.execute(
        "UPDATE excursions SET status = 'Cancelled' WHERE excursion_id = ?",
        (excursion_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"旅行订单 {excursion_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到ID为 {excursion_id} 的旅行订单。"
    
@tool(parse_docstring=True)
def update_excursion(
    excursion_id: int,
    excursion_date: Optional[str] = None,
    duration_hours: Optional[int] = None,
    status: Optional[str] = None
) -> str:
    """根据订单ID更新旅行项目的信息。

    Args:
        excursion_id (int): 要更新的旅行订单的ID。
        excursion_date (Optional[str]): 新的旅行日期，格式：YYYY-MM-DD。
        duration_hours (Optional[int]): 新的旅行时长（小时）。
        status (Optional[str]): 新的订单状态（Booked/Cancelled/Completed）。

    Returns:
        str: 表明旅行订单是否成功更新的消息。

    """
    conn = connect(db)
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if excursion_date:
        updates.append("excursion_date = ?")
        params.append(excursion_date)
    if duration_hours:
        updates.append("duration_hours = ?")
        params.append(duration_hours)
    if status:
        updates.append("status = ?")
        params.append(status)
    
    if not updates:
        conn.close()
        return "没有提供要更新的字段。"
    
    params.append(excursion_id)
    query = f"UPDATE excursions SET {', '.join(updates)} WHERE excursion_id = ?"
    
    cursor.execute(query, params)
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"旅行订单 {excursion_id} 已成功更新。"
    else:
        conn.close()
        return f"未找到ID为 {excursion_id} 的旅行订单。"
    

def excursion_subagent() -> StateGraph:
    llm = select_model("qwen3-max")

    time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")

    SYSTEM_PROMPT = f"""

您是专门处理景点旅游预订的助理。
根据用户的偏好搜索可用景点旅游项目，并与客户确认预订详情。
不许编造歪曲，不确定的就直接诚实说明，尽量先用工具查询。
在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。
如果您需要更多信息或客户改变主意，请请求需要的内容。
\n当前时间: {time}.
\n\n如果用户需要帮助，并且您的工具都不适用，则直接回复。不要浪费用户的时间。不要编造无效的工具或功能。

"""  
    hotel_agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[search_trip_recommendations, update_excursion, cancel_excursion, book_excursion, search_excursions],
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "search_trip_recommendations": False,
                        "search_excursions": False,
                        "book_excursion": {"allowed_decisions": ["approve", "reject"]},
                        "update_excursion": {"allowed_decisions": ["approve", "reject"]},
                        "cancel_excursion": {"allowed_decisions": ["approve", "reject"]},
                    },
                    description_prefix="在执行关键操作前征求用户确认。",
            )],
            checkpointer=InMemorySaver(),
        )
    
    return hotel_agent