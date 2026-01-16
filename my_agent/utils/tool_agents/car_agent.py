from pathlib import Path
from typing import Optional, List, Dict, Union
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

db = "./data/DB/travel.db"

@tool(parse_docstring=True)
def search_available_cars(
        location: Optional[str] = None,
        car_type: Optional[str] = None,
        name: Optional[str] = None
) -> list[dict]:
    """该工具用来检索可用的租车信息。返回包括该地区的租车列表，包含租车公司的名称、车型、价格等信息。

    Based on the provided criteria, search for available rental cars.

    Args:
      location (Optional[str]): Location of the car. Defaults to None.
      car_type (Optional[str]): Type of the car (e.g., sedan, SUV, business car). Defaults to None.
      name (Optional[str]): Name of the rental company. Defaults to None.
    
    Returns:
      list[dict]: A list of dictionaries containing information about available rental cars.
      
    """
    conn = connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM available_cars WHERE available > 0"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if car_type:
        query += " AND car_type LIKE ?"
        params.append(f"%{car_type}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool(parse_docstring=True)
def search_rental_orders(
        user_id: Optional[str] = None,
        order_id: Optional[int] = None,
        status: Optional[str] = None
) -> list[dict]:
    """Base on the provided criteria, search for rental car orders.

    Based on the provided criteria, search for rental car orders.
    
    Args:
      user_id (Optional[str]): User ID. Defaults to None.
      order_id (Optional[int]): Order ID. Defaults to None.
      status (Optional[str]): Order status (active, cancelled, completed). Defaults to None.

    Returns:
      list[dict]: A list of dictionaries containing rental car order information.

    """
    conn = connect(db)
    cursor = conn.cursor()
    
    query = """
    SELECT o.*, c.name as car_name, c.location, c.car_type, c.price_per_day 
    FROM car_rental_orders o 
    JOIN available_cars c ON o.car_id = c.id 
    WHERE 1=1
    """
    params = []

    if user_id:
        query += " AND o.user_id = ?"
        params.append(user_id)
    if order_id:
        query += " AND o.id = ?"
        params.append(order_id)
    if status:
        query += " AND o.status = ?"
        params.append(status)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool(parse_docstring=True)
def create_rental_order(
        car_id: int,
        user_id: str,
        start_date: Union[datetime, date, str],
        end_date: Union[datetime, date, str]
) -> str:
    """Create a new car rental order.

    Create a new car rental order.

    Args:
      car_id (int): ID of the car to be rented.
      user_id (str): User ID.
      start_date (Union[datetime, date, str]): Rental start date.
      end_date (Union[datetime, date, str]): Rental end date.

    Returns:
      str: Message indicating whether the order was successfully created.

    """
    conn = connect(db)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT available, price_per_day, rented_quantity, total_quantity FROM available_cars WHERE id = ?", (car_id,))
        car_info = cursor.fetchone()
        
        if not car_info:
            conn.close()
            return f"未找到ID为 {car_id} 的车辆。"
        
        available, price_per_day, rented_quantity, total_quantity = car_info
        
        if not available or rented_quantity >= total_quantity:
            conn.close()
            return f"车辆 {car_id} 当前不可用。"
        
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, (datetime, date)):
            end_date = end_date.strftime('%Y-%m-%d')
        
        from datetime import datetime as dt
        days = (dt.strptime(end_date, '%Y-%m-%d') - dt.strptime(start_date, '%Y-%m-%d')).days + 1
        total_price = days * price_per_day
        
        cursor.execute(
            "INSERT INTO car_rental_orders (car_id, user_id, start_date, end_date, total_price, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (car_id, user_id, start_date, end_date, total_price)
        )
        
        order_id = cursor.lastrowid
        
        new_rented_quantity = rented_quantity + 1
        new_available = total_quantity - new_rented_quantity
        
        cursor.execute(
            "UPDATE available_cars SET rented_quantity = ?, available = ? WHERE id = ?",
            (new_rented_quantity, new_available, car_id)
        )
        
        conn.commit()
        conn.close()
        
        return f"订单 {order_id} 创建成功！租赁车辆ID: {car_id}，总价: {total_price}元。"
    
    except Exception as e:
        conn.close()
        return f"创建订单失败: {str(e)}"

@tool(parse_docstring=True)
def update_rental_order(
        order_id: int,
        start_date: Optional[Union[datetime, date, str]] = None,
        end_date: Optional[Union[datetime, date, str]] = None,
) -> str:
    """Update the rental dates of a car rental order.

    Update the rental start and/or end dates of an existing car rental order.
    
    Args:
        order_id (int): ID of the order to be updated.
        start_date (Optional[Union[datetime, date, str]]): New start date. Defaults to None.
        end_date (Optional[Union[datetime, date, str]]): New end date. Defaults to None.

    Returns:
        str: Message indicating whether the order was successfully updated.
    """
    conn = connect(db)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT car_id, start_date, end_date FROM car_rental_orders WHERE id = ?", (order_id,))
        order_info = cursor.fetchone()
        
        if not order_info:
            conn.close()
            return f"未找到ID为 {order_id} 的订单。"
        
        car_id, old_start_date, old_end_date = order_info

        cursor.execute("SELECT price_per_day FROM available_cars WHERE id = ?", (car_id,))
        price_info = cursor.fetchone()
        
        if not price_info:
            conn.close()
            return f"未找到相关车辆信息。"
        
        price_per_day = price_info[0]
        
        new_start_date = start_date if start_date else old_start_date
        new_end_date = end_date if end_date else old_end_date
        

        if isinstance(new_start_date, (datetime, date)):
            new_start_date = new_start_date.strftime('%Y-%m-%d')
        if isinstance(new_end_date, (datetime, date)):
            new_end_date = new_end_date.strftime('%Y-%m-%d')
        
        # 重新计算总价
        from datetime import datetime as dt
        days = (dt.strptime(new_end_date, '%Y-%m-%d') - dt.strptime(new_start_date, '%Y-%m-%d')).days + 1
        total_price = days * price_per_day
        
        cursor.execute(
            "UPDATE car_rental_orders SET start_date = ?, end_date = ?, total_price = ? WHERE id = ?",
            (new_start_date, new_end_date, total_price, order_id)
        )
        
        conn.commit()
        conn.close()
        
        return f"订单 {order_id} 更新成功！新的租赁日期: {new_start_date} 至 {new_end_date}，总价: {total_price}元。"
    
    except Exception as e:
        conn.close()
        return f"更新订单失败: {str(e)}"


@tool(parse_docstring=True)
def cancel_rental_order(order_id: int) -> str:
    """Cancel a car rental order.

    Cancel a car rental order.

    Args:
        order_id (int): ID of the order to be cancelled.

    Returns:
        str: Message indicating whether the order was successfully cancelled.
    
    """
    conn = connect(db)
    cursor = conn.cursor()

    try:
        # 获取订单信息
        cursor.execute("SELECT car_id, status FROM car_rental_orders WHERE id = ?", (order_id,))
        order_info = cursor.fetchone()
        
        if not order_info:
            conn.close()
            return f"未找到ID为 {order_id} 的订单。"
        
        car_id, status = order_info
        
        if status == 'cancelled':
            conn.close()
            return f"订单 {order_id} 已经被取消。"
        
        # 更新订单状态为已取消
        cursor.execute("UPDATE car_rental_orders SET status = 'cancelled' WHERE id = ?", (order_id,))
        
        # 更新车辆可用数量
        cursor.execute("SELECT rented_quantity, total_quantity FROM available_cars WHERE id = ?", (car_id,))
        car_info = cursor.fetchone()
        
        if car_info:
            rented_quantity, total_quantity = car_info
            new_rented_quantity = max(0, rented_quantity - 1)
            new_available = total_quantity - new_rented_quantity
            
            cursor.execute(
                "UPDATE available_cars SET rented_quantity = ?, available = ? WHERE id = ?",
                (new_rented_quantity, new_available, car_id)
            )
        
        conn.commit()
        conn.close()
        
        return f"订单 {order_id} 已成功取消。"
    
    except Exception as e:
        conn.close()
        return f"取消订单失败: {str(e)}"
    

def car_subagent() -> StateGraph:
    llm = select_model("qwen3-max")

    time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")

    SYSTEM_PROMPT = f"""

您是专门处理租车预订的助理。当用户需要帮助预订租车时，会将工作委托给您。
不许编造内容，准确回答，无法把握的内容就诚实。
根据用户的偏好搜索可用租车，并与客户确认预订详情。
在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。
如果您需要更多信息或客户改变主意，请请求需要的内容。
现在的时间是 {time}。
"""  
    car_agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[search_available_cars, search_rental_orders, update_rental_order, cancel_rental_order, create_rental_order],
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "search_available_cars": False,
                        "search_rental_orders": False,
                        "update_rental_order": {"allowed_decisions": ["approve", "reject"]},
                        "cancel_rental_order": {"allowed_decisions": ["approve", "reject"]},
                        "create_rental_order": {"allowed_decisions": ["approve", "reject"]},
                    },
                    description_prefix="在执行关键操作前征求用户确认。",
            )],
            checkpointer=InMemorySaver(),
        )
    
    return car_agent