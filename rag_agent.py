import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tools import check_duplicate, recommend_category, recommend_attributes

load_dotenv()

client = OpenAI(
    api_key=os.getenv("TENCENT_API_KEY"),
    base_url=os.getenv("TENCENT_BASE_URL")
)

# ========== 1. 加载向量库 ==========
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

# ========== 2. RAG 检索函数 ==========
def search_standard(query, k=2):
    """从知识库检索相关标准"""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

# ========== 3. 工具定义（同 agent_test.py） ==========
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

# ========== 4. 用户需求 ==========
user_query = "我要申请单晶硅片 210mm"

# ========== 5. 先检索知识库 ==========
context = search_standard(user_query)
print("=" * 50)
print("【RAG 检索到的企业标准】")
print("=" * 50)
print(context[:500] + "...\n")

# ========== 6. 把标准注入 System Prompt，进入 Agent 循环 ==========
messages = [
    {
        "role": "system",
        "content": f"""你是物料主数据助手。请严格根据以下公司标准回答，禁止编造任何属性或数值：

{context}

工作流程：
1. 先查重（check_duplicate）
2. 再推荐分类（recommend_category）
3. 根据分类和公司标准，生成申请单草稿
4. 最后用JSON输出申请单"""
    },
    {"role": "user", "content": user_query}
]

max_rounds = 5
for round_num in range(max_rounds):
    print(f"\n{'='*50}")
    print(f"第 {round_num + 1} 轮")
    print(f"{'='*50}")

    response = client.chat.completions.create(
        model="hy4-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    # 打印思考过程
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        print("\n【模型思考过程】")
        print(reasoning[:300] + "...")

    # 没有工具调用，输出最终结果
    if not msg.tool_calls:
        print("\n【最终申请单草稿】")
        print(msg.content)
        break

    # 有工具调用
    print("\n【模型决定调用以下工具】")
    for tool_call in msg.tool_calls:
        print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")

    messages.append(msg)

    print("\n【工具执行结果】")
    for tool_call in msg.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if func_name == "check_duplicate":
            result = check_duplicate(args["description"])
        elif func_name == "recommend_category":
            result = recommend_category(args["description"])
        else:
            result = "未知工具"

        print(f"  {func_name} 返回: {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
else:
    print("\n达到最大循环次数，Agent 停止。")

