import asyncio

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent, AgentState
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from my_agent.utils.models import select_model
from my_agent.utils.tools import web_search, sql_factory
from my_agent.utils.rag import RAGModule
from my_agent.utils.config import get_config


# 获取配置
config = get_config()


async def get_mcp_tools() -> list[BaseTool]:
    """从配置文件加载MCP工具"""
    mcp_servers = config.get_mcp_tools_config()
    mcp_client = MultiServerMCPClient(mcp_servers)
    mcp_tools = await mcp_client.get_tools()
    return mcp_tools


def load_mcp_tools() -> list[BaseTool]:
    return asyncio.run(get_mcp_tools())


mcp_tools: list[BaseTool] = load_mcp_tools()

database_config = config.get_database_config()
my_db = SQLDatabase.from_uri(database_config['uri'])
SCHEMA = my_db.get_table_info()
execute_sql: callable = sql_factory(my_db)

tools = [web_search, execute_sql] + mcp_tools

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

chat_model = select_model("deepseek-chat")
blog_agent = create_agent(
    model=chat_model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
)
