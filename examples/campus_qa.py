# campus_qa.py — 校园问答示例

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
import parlant.sdk as p


# 通过环境变量配置知识库 API 地址，例如：
#   export CAMPUS_KB_API=https://kb.example.com/search
CAMPUS_KB_API = os.getenv("CAMPUS_KB_API", "http://localhost:8080/search")


@p.tool
async def search_campus_kb(context: p.ToolContext, query: str, top_k: int = 5) -> p.ToolResult:
    """
    调用外部知识库 API（入参为字符串），执行语义 & 全文检索，返回候选片段。

    兼容返回结构（示例为数组）：
    [
      {
        "id": "...",
        "q": "chunk 原文或问答段",
        "a": "可选答案/标题",
        "sourceName": "QA.csv",
        "score": [ {"type": "embedding", "value": 0.57, "index": 5}, ... ],
        ...
      }, ...
    ]
    也兼容 {"results": [...] } 形式。
    """
    # 优先按“入参是字符串”调用；服务端接收 JSON 字符串。
    # 若你的服务需要 {"query": "..."}，可调整为 json={"query": query}。
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(CAMPUS_KB_API, json=query)
            resp.raise_for_status()
            data: Any = resp.json()
    except Exception as exc:
        return p.ToolResult(
            data=[],
            diagnostics={
                "error": str(exc),
                "api": CAMPUS_KB_API,
                "payload": query,
            },
        )

    # 兼容多种响应包装
    if isinstance(data, dict) and "results" in data:
        raw_items = data.get("results", [])
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    # 规范化：以 q 或 text 为主要内容，保留来源与分数信息
    normalized: List[Dict[str, Any]] = []
    for r in raw_items[:top_k]:
        text: Optional[str] = r.get("q") or r.get("text") or r.get("content")
        if not text:
            continue
        normalized.append(
            {
                "id": r.get("id"),
                "text": text,
                "source": r.get("sourceName") or r.get("source") or r.get("datasetId"),
                "score": r.get("score"),  # 可能为分数组合，原样透传
                "chunkIndex": r.get("chunkIndex"),
            }
        )

    return p.ToolResult(data=normalized)


async def add_domain_terms(agent: p.Agent) -> None:
    await agent.create_term(name="校园问答", description="围绕校园信息咨询的领域知识与服务。")
    await agent.create_term(name="教务处", description="负责教学与课程管理的部门。")
    await agent.create_term(name="图书馆", description="提供文献资源与学习空间的公共服务机构。")
    # 新增术语
    await agent.create_term(name="集大", description="集美大学的简称。")
    await agent.create_term(
        name="评教",
        description="期末评价教师课堂教学质量的环节，通常发生在学生查看期末考试成绩之前。",
    )
    await agent.create_term(
        name="i集大",
        description=(
            "集美大学一站式信息化服务平台，整合教务、学工、科研、财务等业务系统，"
            "提供统一入口、统一认证与统一消息的办事服务。"
        ),
    )


async def main() -> None:
    async with p.Server() as server:
        agent = await server.create_agent(
            name="集大爱问",
            description=(
                "\"集大爱问\"是集大专属的智慧校园智能体平台，深度融合学校办学特色与教育教学场景，"
                "通过整合校内信息资源、数据资产及核心应用系统，构建具备校园认知能力的智能服务体系，"
                "为师生提供涵盖教学科研、管理服务、校园生活等领域的个性化智能支持，提升师生工作学习效率，"
                "优化校园智慧服务体验。"
            ),
        )

        # 初始问候与常用 Canned Responses（用于不支持/拒绝类场景）
        await agent.create_canned_response("你好，我是集大爱问，请问你想了解什么？")
        await agent.create_canned_response(
            "抱歉，我目前不支持该请求。我是“集大爱问”，主要提供与集美大学相关的校园问答与服务指引。"
            "你可以描述想了解的校规、校历、图书馆、办事流程、联系方式等问题。"
        )
        await agent.create_canned_response(
            "涉及账号激活、密码重置、权限开通、绑定手机/邮箱等系统级操作，请前往 i集大 办理；"
            "如需指引，我可以提供入口与流程说明。"
        )
        await agent.create_canned_response(
            "查询个人成绩、课表、个人档案等需要登录认证，请在 i集大 办理。本助手不处理个人隐私或需要认证的数据。"
        )

        # 领域术语/词汇
        await add_domain_terms(agent)

        # 核心指南：答案基于知识库检索结果进行组织与引用
        await agent.create_guideline(
            condition=(
                "用户提出与校园相关的问题（如院系、图书馆、办事流程、校历、联系方式等）"
            ),
            action=(
                "先调用 search_campus_kb(query) 获取候选片段；基于最相关的片段进行回答，"
                "必要时引用来源（如来源链接或文档路径）；若检索为空或无法确认，明确告知未找到并建议进一步查询。"
            ),
            tools=[search_campus_kb],
        )

        # 指南 1：任何时候被问到功能，统一回答为“校园问答功能，满足校园信息咨询”
        await agent.create_guideline(
            condition="用户询问你的功能/能做什么/提供哪些服务",
            action=(
                "说明“集大爱问”的功能：面向校园问答，整合校内信息资源，提供教学科研、管理服务、校园生活等智能支持；"
                "核心能力为基于知识库的语义与全文检索并解答，满足校园信息咨询与服务指引。"
            ),
        )

        # 指南 2：被问名字或开发者，统一回答为“集美大学自研的模型和智能体”
        await agent.create_guideline(
            condition="用户问你的名字/你是谁/由谁开发",
            action="回答：我叫集大爱问，是集美大学自研的模型和智能体。",
        )

        # 离题/不支持时优先使用 Canned Response
        await agent.create_guideline(
            condition="用户问题与校园无关或超出本助手职责",
            action=(
                "优先选择最匹配的 Canned Response 进行礼貌拒绝，并说明“集大爱问”的服务范围；"
                "必要时提供官方渠道链接或部门联系方式。"
            ),
        )

        # A. 直接操作请求（代查/代改/代办）→ 固定回复：不代办，指引用 i集大
        await agent.create_guideline(
            condition=(
                "用户要求我直接代为操作或访问需认证的个人数据（如代查成绩、代查课表、修改/重置密码、"
                "开通权限、账号激活、绑定手机/邮箱等）"
            ),
            action=(
                "不执行任何直接操作，也不收集或处理凭据；优先使用对应的 Canned Response 进行礼貌说明，"
                "并明确需前往 i集大 办理。如需指引，可提供入口与流程说明链接。"
            ),
        )

        # B. “如何操作/在哪里查”类咨询 → 允许用 KB 检索步骤后给出指南
        await agent.create_guideline(
            condition=(
                "用户咨询如何在 i集大/教务系统中查询成绩/课表，或如何进行账号激活/密码重置/权限开通等流程"
            ),
            action=(
                "先调用 search_campus_kb(query) 检索官方步骤/说明；基于最相关片段给出操作路径、入口与注意事项；"
                "提醒需在官方系统完成并保护隐私信息。"
            ),
            tools=[search_campus_kb],
        )

        # Journeys
        regulations = await agent.create_journey(
            title="规范性文件查询",
            description="针对校规校纪、管理办法等规范性文件进行检索与引用回答。",
            conditions=["用户咨询校规/校纪/管理办法/政策条款"],
        )
        r0 = await regulations.initial_state.transition_to(tool_state=search_campus_kb)
        await r0.target.transition_to(
            chat_state=(
                "基于检索到的条款进行回答，引用关键片段并附上来源。如存在歧义，指导用户以官方发布为准。"
            ),
            condition="检索命中且片段相关",
        )
        await r0.target.transition_to(
            chat_state=(
                "未检索到明确条款时，说明未找到对应信息，并给出官方查询路径（如校内门户/教务处）。"
            ),
            condition="未命中或相关度不足",
        )

        faq = await agent.create_journey(
            title="校园常见问题",
            description="校历、图书馆位置、期末考试时间、办事流程等 FAQ。",
            conditions=["用户咨询校历/图书馆位置/考试安排/常见校园事务"],
        )
        f0 = await faq.initial_state.transition_to(tool_state=search_campus_kb)
        await f0.target.transition_to(
            chat_state=(
                "给出简明直接的答案，并在需要时附加来源链接/路径；如信息随时间变化，提醒以最新公告为准。"
            ),
            condition="检索命中且可直接回答",
        )
        await f0.target.transition_to(
            chat_state=(
                "无法确定答案时，提供相关部门联系方式或官方查询方式（如教务处、图书馆官网）。"
            ),
            condition="未命中或信息不确定",
        )

        smalltalk_homework = await agent.create_journey(
            title="闲聊与学术作业",
            description="与用户进行礼貌性闲聊；对作业/学术题目，提供思路与资源而非代做。",
            conditions=["用户闲聊/寒暄/请求完成作业/学术问题求解"],
        )
        s0 = await smalltalk_homework.initial_state.transition_to(
            chat_state="对闲聊进行简短礼貌回应，但强调我主要提供校园信息咨询。",
            condition="用户进行闲聊/寒暄",
        )
        await smalltalk_homework.initial_state.transition_to(
            chat_state=(
                "拒绝代做作业或直接给出答案；可以提供解题思路、相关课程资源或官方参考链接，"
                "并鼓励与老师/助教沟通，维护学术诚信。"
            ),
            condition="用户请求代做作业或学术问题求解",
        )

        # 当意图不明确时用于三者之间的消歧
        unclear = await agent.create_observation("用户表达不明确，需要判断是在问校规、FAQ 还是闲聊/作业")
        await unclear.disambiguate([regulations, faq, smalltalk_homework])


if __name__ == "__main__":
    asyncio.run(main())
