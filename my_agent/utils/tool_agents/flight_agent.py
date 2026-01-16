from pathlib import Path
from typing import Optional, List, Dict
import pytz
from datetime import date, datetime

from sqlite3 import connect, Cursor
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph.state import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from my_agent.utils.clean_txt import clean_text
from my_agent.utils.models import select_model


@tool(parse_docstring=True)
def fetch_user_flight_information(user_id: str) -> List[Dict]:
    """This tool fetches the flight information for a given user from the travel database.
    
    Args:
        user_id (str): The unique identifier of the passenger.
    
    Returns:
        A list of dictionaries containing details of each ticket, associated flight information, and seat assignments.
        or an error message if user_id is missing.
    
    """
    
    user_id = user_id
    if not user_id:
        return f"Error: Missing user_id."

    conn = connect("./data/DB/travel.db")
    cursor = conn.cursor()

    # SQL查询语句，连接多个表以获取所需信息
    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = []
    for row in rows:
        result_dict = {}
        for col_name, value in zip(column_names, row):
            # Convert datetime objects to strings
            if isinstance(value, (date, datetime)):
                result_dict[col_name] = value.isoformat()
            else:
                result_dict[col_name] = value
        results.append(result_dict)

    cursor.close()
    conn.close()

    return results

@tool(parse_docstring=True)
def search_flights(
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        start_time: Optional[date | datetime] = None,
        end_time: Optional[date | datetime] = None,
        limit: int = 20,
) -> List[Dict]:
    """Search for flights based on specified criteria.
    
    Searches the database for flights matching the given parameters such as 
    departure airport, arrival airport, and departure time range.
    
    Args:
        departure_airport (Optional[str]): The departure airport code.
        arrival_airport (Optional[str]): The arrival airport code.
        start_time (Optional[date | datetime]): Start of the departure time range.
        end_time (Optional[date | datetime]): End of the departure time range.
        limit (int): Maximum number of results to return. Defaults to 20.
    
    Returns:
        List[Dict]: A list of dictionaries containing flight information matching the criteria.
    """
    conn = connect("./data/DB/travel.db")
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)

    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = []
    for row in rows:
        result_dict = {}
        for col_name, value in zip(column_names, row):
            # 将 datetime 对象转换为字符串
            if isinstance(value, (date, datetime)):
                result_dict[col_name] = value.isoformat()
            else:
                result_dict[col_name] = value
        results.append(result_dict)

    cursor.close()
    conn.close()

    return results

@tool(parse_docstring=True)
def update_ticket_to_new_flight(
        ticket_no: str, new_flight_id: int, user_id: str
) -> str:
    """Update a user's ticket to a new valid flight.
    
    This tool validates the new flight, checks ownership, and updates the ticket
    to the new flight if all conditions are met (e.g., flight exists, departure
    is at least 3 hours away, user owns the ticket).

    Args:
        ticket_no (str): The ticket number to update.
        new_flight_id (int): The ID of the new flight.
        user_id (str): The ID of the user requesting the update.
    
    Returns:
        str: A message indicating the result of the operation.
    """
    passenger_id = user_id
    if not passenger_id:
        return "未配置乘客 ID。"

    conn = connect("./data/DB/travel.db")
    cursor = conn.cursor()

    # 查询新航班的信息
    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "提供的新的航班 ID 无效。"
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))

    # 设置时区并计算当前时间和新航班起飞时间之间的差值
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"不允许重新安排到距离当前时间少于 3 小时的航班。所选航班时间为 {departure_time}。"

    # 确认原机票的存在性
    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "未找到给定机票号码的现有机票。"

    # 确认已登录用户确实拥有此机票
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"当前登录的乘客 ID 为 {passenger_id}，不是机票 {ticket_no} 的拥有者。"

    # 更新机票对应的航班ID
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "机票已成功更新为新的航班。"


@tool
def cancel_ticket(ticket_no: str, user_id: str) -> str:
    """Cancel a user's ticket and remove it from the database.

    This tool validates the ticket ownership and removes the ticket along with
    all associated records from the database.

    Args:
        ticket_no (str): The ticket number to cancel.
        user_id (str): The ID of the user requesting the cancellation.

    Returns:
        str: A message indicating the result of the operation.
    """
    passenger_id = user_id
    if not passenger_id:
        return "未配置乘客 ID。"

    conn = connect("./data/DB/travel.db")
    cursor = conn.cursor()

    # 查询给定机票号是否存在
    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "未找到给定机票号码的现有机票。"

    # 确认已登录用户确实拥有此机票
    cursor.execute(
        "SELECT ticket_no FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"当前登录的乘客 ID 为 {passenger_id}，不是机票 {ticket_no} 的拥有者。"

    # 删除机票对应的所有记录（按照外键依赖顺序）
    # 1. 先删除登机牌记录
    cursor.execute("DELETE FROM boarding_passes WHERE ticket_no = ?", (ticket_no,))
    # 2. 删除机票-航班关联记录
    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    # 3. 最后删除机票主记录
    cursor.execute("DELETE FROM tickets WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "机票已成功取消。"

def flight_subagent() -> StateGraph:
    llm = select_model("qwen3-max")


    time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")
    SYSTEM_PROMPT = f"""
您是专门处理航班查询，改签和预定的助理。
只能通过查询结果告知，不能编造信息
当用户需要帮助更新他们的预订时，主助理会将工作委托给您。
请与客户确认更新后的航班详情，并告知他们任何额外费用。
在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。
如果您需要更多信息或客户改变主意，直接返回。
请记住，在相关工具成功使用后，预订才算完成。
\n当前时间: {time}.
如果用户需要帮助，并且您的工具都不适用，则不要浪费用户的时间。不要编造无效的工具或功能。',
"""

    flight_agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket],
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "fetch_user_flight_information": False,
                        "search_flights": False,
                        "update_ticket_to_new_flight": {"allowed_decisions": ["approve", "reject"]},
                        "cancel_ticket": {"allowed_decisions": ["approve", "reject"]},
                    },
                    description_prefix="在执行关键操作前征求用户确认。",
            )],
            checkpointer=InMemorySaver(),
        )
    
    return flight_agent