import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("TENCENT_API_KEY"),
    base_url=os.getenv("TENCENT_BASE_URL")
)

response = client.chat.completions.create(
    model="hy4-preview", 

    messages=[
        {"role": "system", 
	"content": "你是一个物料主数据助手。请以纯JSON格式输出，包含字段：category（分类）、attributes（属性列表）、confidence（置信度0-1）。只输出JSON，不要markdown代码块，不要任何解释文字。"
	},
        {"role": "user", 
	"content": "单晶硅片 210mm 应该归到哪个物料分类?需要哪些属性？"
	}
    ]
)

import json, re
content = response.choices[0].message.content
match = re.search(r'\{.*\}', content, re.DOTALL)
if match:
    data = json.loads(match.group())
    print(data)
else:
    print("未找到JSON:", content)
print(response.choices[0].message.content)
