from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import ToolMessage
from langgraph.runtime import Runtime
from typing import Any
import uuid

class SaveStoreMiddleware(AgentMiddleware):
    """用于从存储中读取信息的中间件示例"""

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        user_id = runtime.context.user_id
        name_space = (user_id, "memories")
        store = runtime.store

        msg = state["messages"][-1].content
        await store.aput(name_space, str(uuid.uuid4()), {"content": msg})

        return None
