from typing import Optional, Union

from sqlite3 import connect
from datetime import date, datetime
from langchain.tools import tool
from langgraph.graph.state import StateGraph
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import pytz
from my_agent.utils.models import select_model

db = "./data/DB/travel.db"

@tool(parse_docstring=True)
def search_available_hotels(
        location: Optional[str] = None,
        name: Optional[str] = None,
        room_type: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None
) -> list[dict]:
    """Search for available hotels to get recommendations for booking for the user.

    Search for available hotels based on location, name, room type, rating and price to get hotel recommendations.

    Args:
        location (Optional[str]): The location/city of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        room_type (Optional[str]): The type of room (Standard/Deluxe/Suite). Defaults to None.
        min_rating (Optional[float]): Minimum hotel rating. Defaults to None.
        max_price (Optional[float]): Maximum price per night. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries containing available hotel information for booking recommendations.
    """

    conn = connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM hotels WHERE available_rooms > 0"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if room_type:
        query += " AND room_type = ?"
        params.append(room_type)
    if min_rating:
        query += " AND rating >= ?"
        params.append(min_rating)
    if max_price:
        query += " AND price_per_night <= ?"
        params.append(max_price)

    query += " ORDER BY rating DESC, price_per_night ASC"

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

@tool(parse_docstring=True)
def search_user_hotels(
        location: Optional[str] = None,
        name: Optional[str] = None,
        passenger_id: Optional[str] = None
) -> list[dict]:
    """Search for existing hotel bookings by location, name and passenger_id.

    Search for existing hotel bookings by location, name and passenger_id.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        passenger_id (Optional[str]): The passenger ID. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries containing hotel booking information that matches the search criteria.
    """

    conn = connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM hotel_bookings WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND hotel_name LIKE ?"
        params.append(f"%{name}%")
    if passenger_id:
        query += " AND passenger_id = ?"
        params.append(passenger_id)

    # print('查询酒店的SQL：' + query, '参数: ', params)
    cursor.execute(query, params)
    results = cursor.fetchall()
    # print('查询酒店的结果: ', results)
    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool(parse_docstring=True)
def book_hotel(
        booking_id: Optional[int] = None,
        hotel_id: Optional[int] = None,
        passenger_id: Optional[str] = None,
        check_in_date: Optional[Union[datetime, date, str]] = None,
        check_out_date: Optional[Union[datetime, date, str]] = None,
        rooms_needed: Optional[int] = 1
) -> str:
    """
    Confirm an existing hotel booking or create a new hotel booking.
    
    If booking_id is provided, confirms the existing booking.
    If hotel_id is provided, creates a new booking based on the hotel information from hotels table.

    Args:
        booking_id (Optional[int]): The booking ID to confirm an existing booking. Defaults to None.
        hotel_id (Optional[int]): The hotel ID from hotels table for new booking. Required when creating new booking.
        passenger_id (Optional[str]): The passenger ID for new booking. Required when creating new booking.
        check_in_date (Optional[Union[datetime, date, str]]): Check-in date for new booking. Required when creating new booking.
        check_out_date (Optional[Union[datetime, date, str]]): Check-out date for new booking. Required when creating new booking.
        rooms_needed (Optional[int]): Number of rooms needed. Defaults to 1.

    Returns:
        str: A message indicating whether the hotel booking was successfully confirmed or created.
    """
    conn = connect(db)
    cursor = conn.cursor()

    # 方式1: 如果提供了booking_id，确认现有预订
    if booking_id is not None:
        cursor.execute("UPDATE hotel_bookings SET status = 'Confirmed' WHERE booking_id = ?", (booking_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"酒店预订 {booking_id} 已成功确认。"
        else:
            conn.close()
            return f"未找到预订ID为 {booking_id} 的酒店预订。"
    
    # 方式2: 如果提供了hotel_id，基于酒店信息创建新预订
    elif hotel_id is not None:
        # 验证必需参数
        if not all([passenger_id, check_in_date, check_out_date]):
            conn.close()
            return "创建新预订失败：缺少必需参数（passenger_id, check_in_date, check_out_date）。"
        
        # 验证房间数量
        if rooms_needed is None or rooms_needed < 1:
            rooms_needed = 1
        
        # 验证日期
        try:
            check_in = str(check_in_date)
            check_out = str(check_out_date)
            if check_in >= check_out:
                conn.close()
                return "创建新预订失败：退房日期必须晚于入住日期。"
        except Exception as e:
            conn.close()
            return f"创建新预订失败：日期格式错误 - {str(e)}。"
        
        # 从hotels表获取酒店信息
        cursor.execute(
            "SELECT id, name, location, room_type, price_per_night, available_rooms, rating FROM hotels WHERE id = ?",
            (hotel_id,)
        )
        hotel_info = cursor.fetchone()
        
        if not hotel_info:
            conn.close()
            return f"创建新预订失败：未找到ID为 {hotel_id} 的酒店。"
        
        hotel_id_db, hotel_name, location, room_type, price_per_night, available_rooms, rating = hotel_info
        
        # 检查是否有足够的可用房间
        if available_rooms < rooms_needed:
            conn.close()
            return f"创建新预订失败：酒店 {hotel_name} 只剩 {available_rooms} 间 {room_type} 房，无法满足 {rooms_needed} 间的需求。"
        
        # 创建新预订
        cursor.execute(
            """INSERT INTO hotel_bookings 
            (passenger_id, hotel_name, location, check_in_date, check_out_date, room_type, price_per_night, status) 
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Confirmed')""",
            (passenger_id, hotel_name, location, check_in, check_out, room_type, price_per_night)
        )
        new_booking_id = cursor.lastrowid
        
        # 减少可用房间数
        cursor.execute(
            "UPDATE hotels SET available_rooms = available_rooms - ? WHERE id = ?",
            (rooms_needed, hotel_id)
        )
        
        conn.commit()
        conn.close()
        
        # 计算总价
        from datetime import datetime as dt
        try:
            days = (dt.fromisoformat(check_out) - dt.fromisoformat(check_in)).days
        except:
            days = 1
        total_price = price_per_night * days * rooms_needed
        
        return (f"✅ 预订成功！\n"
                f"预订ID: {new_booking_id}\n"
                f"酒店: {hotel_name} (评分: {rating}⭐)\n"
                f"位置: {location}\n"
                f"房型: {room_type}\n"
                f"入住日期: {check_in}\n"
                f"退房日期: {check_out}\n"
                f"房间数量: {rooms_needed}\n"
                f"价格: ¥{price_per_night}/晚 × {days}晚 × {rooms_needed}间 = ¥{total_price:.2f}")
    
    else:
        conn.close()
        return "创建新预订失败：必须提供 booking_id（确认预订）或 hotel_id（创建新预订）。"


@tool
def update_hotel(
        booking_id: int,
        check_in_date: Optional[Union[datetime, date]] = None,
        check_out_date: Optional[Union[datetime, date]] = None,
) -> str:
    """update hotel booking information such as check-in and check-out dates.
    
    Args:
        booking_id (int): The booking ID of the hotel booking to be updated.
        check_in_date (Optional[Union[datetime, date]]): The new check-in date for the hotel. Defaults to None.
        check_out_date (Optional[Union[datetime, date]]): The new check-out date for the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel booking was successfully updated.
    """
    conn = connect(db)
    cursor = conn.cursor()

    if check_in_date:
        cursor.execute(
            "UPDATE hotel_bookings SET check_in_date = ? WHERE booking_id = ?", (check_in_date, booking_id)
        )
    if check_out_date:
        cursor.execute(
            "UPDATE hotel_bookings SET check_out_date = ? WHERE booking_id = ?", (check_out_date, booking_id)
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店预订 {booking_id} 已成功更新。"
    else:
        conn.close()
        return f"未找到预订ID为 {booking_id} 的酒店预订。"


@tool
def cancel_hotel(booking_id: int) -> str:
    """
    Cancel a hotel booking by its booking ID.

    Args:
        booking_id (int): The booking ID of the hotel booking to be canceled.

    Returns:
        str: A message indicating whether the hotel booking was successfully canceled.
    """
    conn = connect(db)
    cursor = conn.cursor()

    # 将status字段设置为'Cancelled'来表示取消预订
    cursor.execute("UPDATE hotel_bookings SET status = 'Cancelled' WHERE booking_id = ?", (booking_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"酒店预订 {booking_id} 已成功取消。"
    else:
        conn.close()
        return f"未找到预订ID为 {booking_id} 的酒店预订。"
    

def hotel_subagent() -> StateGraph:
    llm = select_model("qwen3-max")

    time = datetime.now().astimezone(pytz.timezone("Etc/GMT-3")).strftime("%Y-%m-%d %H:%M:%S %Z")

    SYSTEM_PROMPT = f"""

您是专门处理酒店预订的助理。
根据用户的偏好搜索可用酒店，并与客户确认预订详情。
不许编造歪曲，不确定的就直接诚实说明，尽量先用工具查询。
在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。
如果您需要更多信息或客户改变主意，请请求需要的内容。
\n当前时间: {time}.
\n\n如果用户需要帮助，并且您的工具都不适用，则直接回复。不要浪费用户的时间。不要编造无效的工具或功能。

"""  
    hotel_agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[search_available_hotels, search_user_hotels, update_hotel, cancel_hotel, book_hotel],
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "search_available_hotels": False,
                        "search_user_hotels": False,
                        "book_hotel": {"allowed_decisions": ["approve", "reject"]},
                        "update_hotel": {"allowed_decisions": ["approve", "reject"]},
                        "cancel_hotel": {"allowed_decisions": ["approve", "reject"]},
                    },
                    description_prefix="在执行关键操作前征求用户确认。",
            )],
            checkpointer=InMemorySaver(),
        )
    
    return hotel_agent