# healthcare.py

import parlant.sdk as p
import asyncio
import json
import os
import aiohttp
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Annotated
from parlant.core.tools import ToolParameterOptions
from parlant.core.loggers import Logger, LogLevel, StdoutLogger
from parlant.core.contextual_correlator import ContextualCorrelator

# load env
from dotenv import load_dotenv
load_dotenv()

logger = None


# 通用 API 调用配置加载和工具生成
def load_tools_config(config_path: str = "tools_config.json") -> Dict[str, Any]:
    """加载工具配置文件"""
    try:
        logger.info(f"加载工具配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载 {len(config)} 个工具配置")
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 不存在")
        return []
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return []


def replace_placeholders(template: Any, params: Dict[str, Any]) -> Any:
    """递归替换模板中的占位符
    
    支持：
    - 字符串中的 {param_name} 占位符
    - 嵌套的字典和列表结构
    - 保持原始数据类型
    """
    if isinstance(template, str):
        # 替换字符串中的所有占位符
        result = template
        for param_name, param_value in params.items():
            placeholder = f"{{{param_name}}}"
            if placeholder in result:
                # 如果整个字符串就是一个占位符，直接返回原始值（保持类型）
                if result == placeholder:
                    return param_value
                # 否则转换为字符串进行替换
                result = result.replace(placeholder, str(param_value))
        return result
    elif isinstance(template, dict):
        # 递归处理字典
        return {key: replace_placeholders(value, params) for key, value in template.items()}
    elif isinstance(template, list):
        # 递归处理列表
        return [replace_placeholders(item, params) for item in template]
    else:
        # 其他类型直接返回
        return template


async def call_api(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """通用 API 调用函数，支持灵活的配置格式"""
    
    # 过滤掉 None 值
    params = {k: v for k, v in params.items() if v is not None}
    
    # 支持两种配置格式：api 和 endpoint
    if "endpoint" in config:
        # 新格式：更灵活的配置
        endpoint_config = config["endpoint"]
        
        # 替换 URL 中的占位符
        url = replace_placeholders(endpoint_config["url"], params)
        method = endpoint_config.get("method", "GET").upper()
        
        # 替换 headers 中的占位符
        headers = replace_placeholders(endpoint_config.get("headers", {}), params)
        
        # 替换 body 中的占位符（对于非 GET 请求）
        if method != "GET" and "body" in endpoint_config:
            body = replace_placeholders(endpoint_config["body"], params)
        else:
            body = None
        
        # GET 请求使用未在 URL/headers/body 中使用的参数作为 query 参数
        if method == "GET":
            # 收集已使用的参数
            used_params = set()
            for param_name in params:
                if f"{{{param_name}}}" in endpoint_config.get("url", ""):
                    used_params.add(param_name)
                # 检查 headers 中是否使用
                headers_str = str(endpoint_config.get("headers", {}))
                if f"{{{param_name}}}" in headers_str:
                    used_params.add(param_name)
            # 剩余参数作为 query 参数
            query_params = {k: v for k, v in params.items() if k not in used_params}
        else:
            query_params = {}
    
    # 记录API调用信息
    logger.info(f"🚀 API调用: {method} {url}")
    if headers:
        logger.debug(f"📋 请求头: {headers}")
    if query_params:
        logger.debug(f"❓ Query参数: {query_params}")
    if body:
        logger.debug(f"📦 请求体: {json.dumps(body, ensure_ascii=False, indent=2)}")
    
    # 发送请求
    async with aiohttp.ClientSession() as session:
        try:
            if method == "GET":
                async with session.get(url, params=query_params, headers=headers) as response:
                    logger.info(f"✅ 响应状态: {response.status}")
                    result = await response.json()
                    logger.debug(f"📨 响应数据: {result}")
                    return result
            else:
                async with session.request(method, url, json=body, params=query_params, headers=headers) as response:
                    logger.info(f"✅ 响应状态: {response.status}")
                    result = await response.json()
                    logger.debug(f"📨 响应数据: {result}")
                    return result
        except Exception as e:
            logger.error(f"❌ API调用失败: {str(e)}")
            raise


def create_dynamic_tool(tool_config: Dict[str, Any]):
    """根据配置动态创建工具函数，使用 Annotated 传递参数描述"""
    from inspect import Parameter, Signature
    
    tool_name = tool_config["name"]
    description = tool_config["description"]
    parameters = tool_config.get("parameters", {})
    properties = parameters.get("properties", {})
    required_params = parameters.get("required", [])
    
    # 类型映射
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }
    
    # 构建函数签名
    sig_params = [Parameter('context', Parameter.POSITIONAL_OR_KEYWORD, annotation=p.ToolContext)]
    call_params = []
    
    # 处理所有参数（先必需参数，后可选参数）
    for param_name, param_config in properties.items():
        param_type = type_mapping.get(param_config.get("type", "string"), str)
        is_required = param_name in required_params
        default_value = param_config.get("default") if not is_required else None

        # 从配置中提取参数描述
        param_description = param_config.get("description", f"Parameter {param_name}")
        param_examples = param_config.get("examples", [])
        
        # 只传递必要的描述信息，避免冗余
        # 框架会自动将 ToolParameterOptions 中的信息复制到 ToolParameterDescriptor
        annotated_type = Annotated[param_type, ToolParameterOptions(
            description=param_description,
            examples=param_examples
        )]
        
        # 创建参数，根据是否必需设置默认值
        if is_required:
            sig_params.append(Parameter(param_name, Parameter.POSITIONAL_OR_KEYWORD, annotation=annotated_type))
        else:
            sig_params.append(Parameter(param_name, Parameter.POSITIONAL_OR_KEYWORD, annotation=annotated_type, default=default_value))
        
        call_params.append(param_name)
    
    # 创建函数签名
    signature = Signature(sig_params, return_annotation=p.ToolResult)
    
    # 定义函数体
    async def dynamic_tool_func(*args, **kwargs):
        try:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 直接使用参数作为 API 请求参数
            params = {}
            for param_name in call_params:
                if param_name in bound_args.arguments:
                    params[param_name] = bound_args.arguments[param_name]
            
            logger.info(f"调用动态工具: {tool_name}")
            logger.debug(f"工具参数: {params}")
            
            result = await call_api(tool_config, params)
            
            logger.info(f"工具 {tool_name} 执行成功")
            return p.ToolResult(data=result)
            
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {str(e)}")
            return p.ToolResult(data={"error": str(e)})
    
    # 设置元数据
    dynamic_tool_func.__name__ = tool_name
    dynamic_tool_func.__doc__ = description
    dynamic_tool_func.__signature__ = signature

    return p.tool(dynamic_tool_func)


# 全局变量声明
tools_list = []
dynamic_tools = {}


async def initialize_tools() -> None:
    """初始化动态工具"""
    global tools_list, dynamic_tools
    
    logger.info("开始初始化动态工具...")
    
    # 加载配置
    tools_list = load_tools_config()
    
    # 创建动态工具
    logger.info("开始创建动态工具...")
    for tool_config in tools_list:
        try:
            tool_func = create_dynamic_tool(tool_config)
            dynamic_tools[tool_config["name"]] = tool_func
            logger.debug(f"成功创建工具: {tool_config['name']}")
        except Exception as e:
            logger.error(f"创建工具 {tool_config.get('name', 'unknown')} 失败: {str(e)}")
    
    logger.info(f"动态工具创建完成，共创建 {len(dynamic_tools)} 个工具")

async def setup_agent_guidelines(agent) -> None:
    """为智能体设置工具指导原则"""
    logger.info("开始为智能体添加工具指导原则...")
    
    for tool_name, tool_func in dynamic_tools.items():
        # 根据工具配置生成条件和动作描述
        tool_config = next(
            (config for config in tools_list 
             if config["name"] == tool_name), 
            None
        )
        
        if tool_config:
            action = f"Use the {tool_name} tool: {tool_config['description']}"
            
            try:
                await agent.create_guideline(
                    # todo 根据工具类型生成更精准的条件描述
                    condition=tool_config['description'],
                    action=action,
                    tools=[tool_func],
                )
                logger.debug(f"成功为工具 {tool_name} 添加指导原则")
            except Exception as e:
                logger.error(f"为工具 {tool_name} 添加指导原则失败: {str(e)}")
    
    logger.info("工具指导原则添加完成")

async def main() -> None:
    async with p.Server(
        nlp_service=p.NLPServices.openrouter,
        log_level=LogLevel.DEBUG
    ) as server:
        # 获取Parlant SDK的日志器
        global logger
        logger = server._container[p.Logger]
        
        logger.info("启动 Parlant 服务器...")
        
        # 初始化工具
        await initialize_tools()
        
        # 创建智能体
        logger.info("创建智能体...")
        agent = await server.create_agent(
            name="Greeting Agent",
            description="支持基本的问候语，并根据用户的问题调用外部工具，使用与输入相同的语言回复",
        )
        logger.info("智能体创建成功")
        
        # 设置工具指导原则
        await setup_agent_guidelines(agent)
        


if __name__ == "__main__":
    asyncio.run(main())
