import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tools import check_duplicate, recommend_category

load_dotenv()

st.set_page_config(page_title="物料主数据智能申请助手", page_icon="📦", layout="wide")

# ========== 初始化（只执行一次） ==========
@st.cache_resource
def init_resources():
    client = OpenAI(
        api_key=os.getenv("TENCENT_API_KEY"),
        base_url=os.getenv("TENCENT_BASE_URL")
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("TENCENT_API_KEY"),
        base_url=os.getenv("TENCENT_BASE_URL"),
        model="kinfra-text-embedding-4b",
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"}
    )
    vectorstore = Chroma(
        persist_directory="./data/chroma_db",
        embedding_function=embeddings
    )
    return client, vectorstore

client, vectorstore = init_resources()

# ========== 工具定义 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": "输入物料描述，查询历史物料中是否有相似物料，防止一物多码",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "物料描述"}
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_category",
            "description": "根据物料描述推荐物料分类",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "物料描述"}
                },
                "required": ["description"]
            }
        }
    }
]

# ========== RAG 检索 ==========
def search_standard(query, k=2):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

# ========== Agent 核心逻辑 ==========
def run_agent(user_query):
    context = search_standard(user_query)

    messages = [
        {
            "role": "system",
            "content": f"""你是物料主数据助手。请严格根据以下公司标准回答，禁止编造任何属性或数值：

{context}

工作流程：
1. 先查重（check_duplicate）
2. 再推荐分类（recommend_category）
3. 根据分类和公司标准，生成申请单草稿
4. 最后用JSON输出申请单
5.  请仔细阅读历史对话。如果用户已经提供了某些信息，绝不重复询问。只询问缺失的必填项。"""
        },
        {"role": "user", "content": user_query}
    ]
        
    # 2. 把之前的聊天记录加进来（保留最近 6 轮对话，防止 token 超限）
    history = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
    
    for msg in history:
        # 过滤掉系统提示和可能存在的复杂对象，只保留纯文本对话
        if msg["role"] in ["user", "assistant"] and isinstance(msg["content"], str):
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    # 3. 把最新的一句加进去
    messages.append({"role": "user", "content": user_query})

    steps = []  # 记录过程，用于前端展示
    max_rounds = 5

    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model="hy4-preview",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            steps.append({"type": "thinking", "content": reasoning})

        if not msg.tool_calls:
            steps.append({"type": "final", "content": msg.content})
            return steps

        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "check_duplicate":
                result = check_duplicate(args["description"])
            elif func_name == "recommend_category":
                result = recommend_category(args["description"])
            else:
                result = "未知工具"

            steps.append({
                "type": "tool",
                "name": func_name,
                "args": args,
                "result": result
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

    steps.append({"type": "final", "content": "达到最大循环次数，Agent 停止。"})
    return steps

# ========== 界面 ==========
st.title("📦 物料主数据智能申请助手")
st.caption("基于 RAG + Agent 的物料主数据智能申请 MVP | 主数据治理 · 数据标准 · 智能查重")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
user_input = st.chat_input("请输入物料申请需求，例如：我要申请单晶硅片 210mm")

if user_input:
    # 展示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 运行 Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在检索标准、调用工具、生成申请单..."):
            steps = run_agent(user_input)

        # 展示思考过程和工具调用
        for step in steps:
            if step["type"] == "thinking":
                with st.expander("🧠 模型思考过程", expanded=False):
                    st.markdown(step["content"])
            elif step["type"] == "tool":
                with st.expander(f"🔧 调用工具：{step['name']}", expanded=False):
                    st.markdown(f"**参数**：`{step['args']}`")
                    st.markdown(f"**返回**：`{step['result']}`")
            elif step["type"] == "final":
                st.markdown(step["content"])
                st.session_state.messages.append({"role": "assistant", "content": step["content"]})

# 侧边栏
with st.sidebar:
    st.header("关于本助手")
    st.markdown("""
    **功能**：
    - 智能查重，防止一物多码
    - 基于企业标准推荐分类
    - 严格引用知识库，禁止编造
    - 自动生成结构化申请单草稿

    **技术栈**：
    - RAG：ChromaDB + 腾讯云 Embedding
    - Agent：Hy4-preview + Function Calling
    - 前端：Streamlit
    """)
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
