# %%
import requests
import json
import faiss
import numpy as np

from flask import Flask, Response, request, jsonify
import os
from analyzer import analyze_video
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Missing video file"}), 400

    video = request.files["video"]
    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    try:
        result = analyze_video(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)

    app.run(debug=True)
# %%
# === 配置项 ===
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "EntropyYue/chatglm3"  # Ollama 中的 LLM 名称
EMBED_MODEL = "dentonzst/text2vec-base-chinese"  # Ollama 中的嵌入模型名称

INDEX_PATH = "faiss_index/faiss_index.index"
METADATA_PATH = "faiss_index/metadata.json"


# %%
# === 加载 metadata.json（用于显示文本） ===
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata_json = json.load(f)

# %%
# 提取文本内容
texts = metadata_json.get("texts", [])
# （可选）提取每段文本的来源信息，后续可在生成回答中展示引用
sources = metadata_json.get("metadata", [])


# %%
# === 加载 FAISS 索引 ===
index = faiss.read_index(INDEX_PATH)

# %%
# === 使用 Ollama 的嵌入模型获取 query 向量 ===
def embed_with_ollama(text):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

# %%

# === 检索相关文本块 ===
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embed_with_ollama(query)
    D, I = index.search(np.array([query_embedding], dtype='float32'), top_k)
    return [texts[i] for i in I[0]]


# === 构造 Prompt 并调用 Ollama 的 Qwen 模型 ===
def generate_answer(query):
    chunks = retrieve_relevant_chunks(query)
    context = "\n".join(chunks)
    prompt = f"请基于以下内容回答问题：\n{context}\n\n问题：{query}\n回答："

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


# %%
def generate(text):

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": text,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


TEMPLATE = """
请根据以下学生的课堂行为数据、情绪识别结果和考试成绩，撰写一份课堂综合表现分析报告，包含以下内容：

1. 【专注度分析】：
根据课堂中“阅读”（占比{reading:.0%}）、“书写”（占比{writing:.0%}）、“使用手机”（占比{phone:.0%}）、“低头”（占比{bowing:.0%}）、“伏案”（占比{leaning:.0%}）等行为，分析该生的注意力集中程度。

2. 【互动表现】：
课堂中举手发言行为占比为{hand:.0%}，体现了该生在课堂上参与互动的积极性和表达意愿。

3. 【情绪状态】：
课堂中以“开心”（{happy:.0%}）、“中性”（{neutral:.0%}）为主，同时伴有一定比例的“悲伤”（{sad:.0%}）和“愤怒”（{angry:.0%}）情绪，说明学生情绪整体较为{emotion_tone}，但仍存在{emotion_issue}。

4. 【学习成果】：
从近期成绩来看，学生在语文（{score[语文]}）、数学（{score[数学]}）、英语（{score[英语]}）、物理（{score[物理]}）、化学（{score[化学]}）、生物（{score[生物]}）等学科中表现{performance}，其中{advantage}是其优势学科，{weakness}则有待加强。

5. 【综合建议】：
该生课堂参与度{participation}，情绪状态{emotion_summary}，学习成绩{score_summary}。建议进一步{suggest_behavior}，同时保持在{highlight_subject}方面的良好表现，以促进全面发展。
"""



def render_report(data):
    action = data["action_ratios"]
    emotion = data["emotion_ratios"]
    score = data["score"]

    # 情绪状态分析逻辑
    emotion_positive = emotion.get("happy", 0) + emotion.get("neutral", 0)
    emotion_negative = emotion.get("sad", 0) + emotion.get("angry", 0)
    emotion_tone = "稳定" if emotion_positive > 0.6 else "波动"
    emotion_issue = (
        "一定的负面情绪，如悲伤或愤怒" if emotion_negative > 0.2 else "较少的负面情绪"
    )

    # 优势与弱势学科识别
    sorted_scores = sorted(score.items(), key=lambda x: x[1], reverse=True)
    advantage = f"{sorted_scores[0][0]}（{sorted_scores[0][1]}）"
    weakness = f"{sorted_scores[-1][0]}（{sorted_scores[-1][1]}）"

    # 平均分与综合学习表现
    avg_score = sum(score.values()) / len(score)
    if avg_score >= 85:
        performance = "优异"
        score_summary = "整体优秀"
    elif avg_score >= 75:
        performance = "良好"
        score_summary = "稳中有进"
    else:
        performance = "一般"
        score_summary = "存在较大提升空间"

    # 参与度和建议
    participation = "较高" if action.get("hand-raising", 0) >= 0.2 else "一般"
    emotion_summary = "积极向上" if emotion.get("happy", 0) > 0.2 else "略显低落"
    distract_behaviors = ["using phone", "bowing the head"]
    if any(action.get(b, 0) > 0.05 for b in distract_behaviors):
        suggest_behavior = "加强专注力训练"
    else:
        suggest_behavior = "保持良好的学习习惯"
    highlight_subject = sorted_scores[0][0]

    return TEMPLATE.format(
        reading=action.get("reading", 0),
        writing=action.get("writing", 0),
        phone=action.get("using phone", 0),
        bowing=action.get("bowing the head", 0),
        leaning=action.get("leaning over the table", 0),
        hand=action.get("hand-raising", 0),
        happy=emotion.get("happy", 0),
        neutral=emotion.get("neutral", 0),
        sad=emotion.get("sad", 0),
        angry=emotion.get("angry", 0),
        emotion_tone=emotion_tone,
        emotion_issue=emotion_issue,
        score=score,
        performance=performance,
        advantage=advantage,
        weakness=weakness,
        participation=participation,
        emotion_summary=emotion_summary,
        score_summary=score_summary,
        suggest_behavior=suggest_behavior,
        highlight_subject=highlight_subject
    )



# %%
graph_prompt = '''下面是一些资料参考：

学生在课堂中展现出“认真听讲”行为，体现了其在“信息整合能力”方面的优势，
这一能力属于“学习方法”评价维度，支撑其核心素养“学会学习”的发展。

学生在课堂中展现出“专注完成作业”行为，体现了其在“自主探究能力”方面的优势，
这一能力属于“学习能力”评价维度，支撑其核心素养“学会学习”的发展。

学生在课堂中展现出“使用电子设备学习”行为，体现了其在“技术应用能力”方面的优势，
这一能力属于“创新精神”评价维度，支撑其核心素养“科学精神”的发展。

学生在课堂中展现出“与同学交流学习”行为，体现了其在“跨文化沟通能力”方面的优势，
这一能力属于“文化理解”评价维度，支撑其核心素养“人文底蕴”的发展。

学生在课堂中展现出“组织班级活动”行为，体现了其在“项目执行能力”方面的优势，
这一能力属于“社会实践”评价维度，支撑其核心素养“实践创新”的发展。

学生在课堂中展现出“安慰受挫同学”行为，体现了其在“情绪调节能力”方面的优势，
这一能力属于“身心健康”评价维度，支撑其核心素养“健康生活”的发展。

学生在课堂中展现出“课后向老师请教”行为，体现了其在“元认知能力”方面的优势，
这一能力属于“学习方法”评价维度，支撑其核心素养“学会学习”的发展。

学生在课堂中展现出“坚持记录情绪日记”行为，体现了其在“情绪调节能力”方面的优势，
这一能力属于“身心健康”评价维度，支撑其核心素养“健康生活”的发展。

学生在课堂中展现出“定期检查学习目标”行为，体现了其在“元认知能力”方面的优势，
这一能力属于“学习方法”评价维度，支撑其核心素养“学会学习”的发展。

学生在课堂中展现出“主动承担班务工作”行为，体现了其在“公民意识”方面的优势，
这一能力属于“社会责任感”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“主动查阅资料”行为，体现了其在“信息整合能力”方面的优势，
这一能力属于“学习方法”评价维度，支撑其核心素养“学会学习”的发展。

学生在课堂中展现出“提出创新想法”行为，体现了其在“创造性解决问题”方面的优势，
这一能力属于“创新精神”评价维度，支撑其核心素养“科学精神”的发展。

学生在课堂中展现出“小组中担任记录员”行为，体现了其在“项目执行能力”方面的优势，
这一能力属于“社会实践”评价维度，支撑其核心素养“实践创新”的发展。

学生在课堂中展现出“参与社区服务”行为，体现了其在“公民意识”方面的优势，
这一能力属于“社会责任感”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“表达不同意见”行为，体现了其在“批判性思维”方面的优势，
这一能力属于“思想品德”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“主动进行体育锻炼”行为，体现了其在“健康管理能力”方面的优势，
这一能力属于“身心健康”评价维度，支撑其核心素养“健康生活”的发展。

学生在课堂中展现出“遵守交通规则”行为，体现了其在“风险规避能力”方面的优势，
这一能力属于“社会责任感”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“通过绘画表达情感”行为，体现了其在“艺术鉴赏能力”方面的优势，
这一能力属于“艺术素养”评价维度，支撑其核心素养“人文底蕴”的发展。

学生在课堂中展现出“跨班级进行学术交流”行为，体现了其在“跨文化沟通能力”方面的优势，
这一能力属于“文化理解”评价维度，支撑其核心素养“人文底蕴”的发展。

学生在课堂中展现出“协助家庭完成家务”行为，体现了其在“家庭责任意识”方面的优势，
这一能力属于“思想品德”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“提出班级改进建议”行为，体现了其在“伦理决策能力”方面的优势，
这一能力属于“思想品德”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“参与科学实验”行为，体现了其在“实证分析能力”方面的优势，
这一能力属于“学业水平”评价维度，支撑其核心素养“科学精神”的发展。

学生在课堂中展现出“使用工具制作模型”行为，体现了其在“动手操作能力”方面的优势，
这一能力属于“劳动技能”评价维度，支撑其核心素养“实践创新”的发展。

学生在课堂中展现出“记录身体健康数据”行为，体现了其在“健康管理能力”方面的优势，
这一能力属于“身心健康”评价维度，支撑其核心素养“健康生活”的发展。

学生在课堂中展现出“规避网络诈骗”行为，体现了其在“风险规避能力”方面的优势，
这一能力属于“社会责任感”评价维度，支撑其核心素养“责任担当”的发展。

学生在课堂中展现出“参与线上研讨会”行为，体现了其在“信息整合能力”方面的优势，
这一能力属于“学习方法”评价维度，支撑其核心素养“学会学习”的发展。'''


@app.route('/report', methods=['POST'])
def report():
    try:
        data = request.get_json()
        report_text = render_report(data)
        result = generate(report_text + graph_prompt)
        return Response(
            json.dumps({"report": result}, ensure_ascii=False),
            content_type="application/json"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

