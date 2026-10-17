import json

def load_materials():
    with open("data/materials.json", "r", encoding="utf-8") as f:
        return json.load(f)

def check_duplicate(description: str):
    """输入物料描述，返回相似历史物料列表"""
    materials = load_materials()
    results = []
    for m in materials:
        if any(word in m["name"] for word in description.split()):
            results.append({"code": m["code"], "name": m["name"], "category": m["category"]})
    return results

def recommend_category(description: str):
    """根据描述推荐分类"""
    if "硅片" in description or "硅" in description:
        return "硅片"
    if "玻璃" in description:
        return "玻璃"
    if "胶膜" in description or "EVA" in description:
        return "封装材料"
    return "待确认"

def recommend_attributes(category: str):
    """根据分类返回必填属性"""
    mapping = {
        "硅片": ["尺寸", "纯度", "电阻率"],
        "晶体": ["直径", "类型", "纯度"],
        "玻璃": ["厚度", "透光率"],
        "封装材料": ["厚度", "类型"]
    }
    return mapping.get(category, [])
