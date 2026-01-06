from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime
from typing import Any

class ReadStoreMiddleware(AgentMiddleware):
    """用于从存储中读取信息的中间件示例"""

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        user_id = runtime.context.user_id
        name_space = (user_id, "memories")
        store = runtime.store

        memories = await store.asearch(name_space)
        
        if memories:
            memory_text = "\n".join([f"- {mem.value.get('content', '')}" for mem in memories])
            return {"messages": [SystemMessage(content=f"用户历史记忆:\n{memory_text}")]}

        return None
