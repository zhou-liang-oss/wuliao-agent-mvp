import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import check_duplicate, recommend_category, recommend_attributes

load_dotenv()

client = OpenAI(
    api_key=os.getenv("TENCENT_API_KEY"),
    base_url=os.getenv("TENCENT_BASE_URL")
)

# 1. 定义工具说明书（告诉模型有哪些工具可用）
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": "输入物料描述，查询历史物料中是否有相似物料，防止一物多码",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "物料描述，如 单晶硅片 210mm"}
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
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_attributes",
            "description": "根据物料分类返回必填属性列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "物料分类"}
                },
                "required": ["category"]
            }
        }
    }
]

# 2. 用户需求
messages = [
    {"role": "system", "content": "你是物料主数据助手。用户提出物料申请时，你需要先查重，再推荐分类，再推荐属性。最后用JSON输出申请单草稿。"},
    {"role": "user", "content": "我要申请单晶硅片 210mm"}
]

# 核心：循环，直到模型不再调用工具
max_rounds = 5  # 防止死循环
for round_num in range(max_rounds):
    print(f"\n=== 第 {round_num + 1} 轮 ===")
    
    response = client.chat.completions.create(
        model="hy4-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # ---- 3.1 打印模型的思考过程（如果有）----
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        print("\n【模型思考过程】")
        print(reasoning)
        
    # 如果模型没有调用工具，说明它准备好最终输出了
    if not msg.tool_calls:
        print("\n=== 最终申请单草稿 ===")
        print(msg.content)
        break
    
    # 有工具调用：把模型的请求加入历史
    messages.append(msg)
    
    # 逐个执行工具
    for tool_call in msg.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"调用工具：{func_name}，参数：{args}")
        
        if func_name == "check_duplicate":
            result = check_duplicate(args["description"])
        elif func_name == "recommend_category":
            result = recommend_category(args["description"])
        elif func_name == "recommend_attributes":
            result = recommend_attributes(args["category"])
        else:
            result = "未知工具"
        
        print(f"工具返回：{result}")
        
        # 把工具结果返回给模型
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
else:
    print("\n达到最大循环次数，停止。")