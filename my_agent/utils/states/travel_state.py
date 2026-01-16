from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

def update_dialog_stack(left: list[str], right: str) -> list[str]:
    if not right:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class TravelState(TypedDict):
    """定义一个TravelAgent的状态结构体，用来存储对话状态
    
    Attributes:
        messages: Annotated[list[AnyMessage], add_messages]
            存储对话消息的列表，使用add_messages中间件进行管理。
        user_id: str
            用户的唯一标识符。
        dialog_state: Annotated[
            list[  # 其元素严格限定为上述五个字符串值之一。这种做法确保了对话状态管理逻辑的一致性和正确性，避免了意外的状态值导致的潜在问题。
                Literal[
                    "assistant",
                    "update_flight",
                    "book_car_rental",
                    "book_hotel",
                    "book_excursion",
                ]
            ],
            update_dialog_stack,
        ]
            对话状态栈，使用update_dialog_stack函数进行管理。
    
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: list[dict]
    dialog_state: Annotated[
        list[  # 其元素严格限定为上述五个字符串值之一。这种做法确保了对话状态管理逻辑的一致性和正确性，避免了意外的状态值导致的潜在问题。
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
    

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, SystemMessage

    state: TravelState = {
        "messages": [],
        "user_id": "user_123",
        "dialog_state": ["assistant"],
    }

    state["messages"].append(HumanMessage(content="Hello!"))
    state["dialog_state"] = update_dialog_stack(state["dialog_state"], "book_hotel")
    print(state)