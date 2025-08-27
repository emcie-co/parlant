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

# 认证类型枚举
class AuthType(Enum):
    QUERY_PARAM = "query_param"
    HEADER = "header"

# 统一的认证处理器
class AuthHandler:
    @staticmethod
    def apply_auth(auth_type: str, headers: Dict[str, str], params: Dict[str, Any], auth_config: Dict[str, Any]) -> None:
        """根据认证类型应用认证配置"""
        key = auth_config.get("key")
        value = auth_config.get("value")
        
        if not (key and value):
            return
            
        try:
            auth_enum = AuthType(auth_type)
            if auth_enum == AuthType.QUERY_PARAM:
                params[key] = value
            elif auth_enum == AuthType.HEADER:
                headers[key] = value
        except ValueError:
            logger.warning(f"未知的认证类型: {auth_type}")

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


def validate_and_format_param(param_value: Any, param_schema: Dict[str, Any]) -> Any:
    """根据 JSON Schema 验证和格式化参数
    
    支持的类型：
    - 基础类型: string, number, integer, boolean, null
    - 复合类型: array, object
    - 组合类型: oneOf, anyOf, allOf
    """
    param_type = param_schema.get("type")
    
    # 处理字符串形式的数组和对象（来自框架的类型转换）
    if isinstance(param_value, str):
        if param_type == "array":
            try:
                # 尝试解析为 JSON 数组
                import json
                parsed_value = json.loads(param_value)
                if isinstance(parsed_value, list):
                    param_value = parsed_value
                else:
                    logger.warning(f"字符串 {param_value} 解析后不是数组")
                    return param_value
            except json.JSONDecodeError:
                logger.warning(f"无法将字符串 {param_value} 解析为 JSON 数组")
                return param_value
        elif param_type == "object":
            try:
                # 尝试解析为 JSON 对象
                import json
                parsed_value = json.loads(param_value)
                if isinstance(parsed_value, dict):
                    param_value = parsed_value
                else:
                    logger.warning(f"字符串 {param_value} 解析后不是对象")
                    return param_value
            except json.JSONDecodeError:
                logger.warning(f"无法将字符串 {param_value} 解析为 JSON 对象")
                return param_value
    
    # 处理 oneOf/anyOf/allOf 组合类型
    if "oneOf" in param_schema:
        # oneOf: 必须匹配其中一个schema
        for schema in param_schema["oneOf"]:
            try:
                return validate_and_format_param(param_value, schema)
            except:
                continue
        # 如果都不匹配，返回原值
        logger.warning(f"参数值 {param_value} 不匹配任何 oneOf schema")
        return param_value
    
    if "anyOf" in param_schema:
        # anyOf: 至少匹配一个schema
        for schema in param_schema["anyOf"]:
            try:
                return validate_and_format_param(param_value, schema)
            except:
                continue
        logger.warning(f"参数值 {param_value} 不匹配任何 anyOf schema")
        return param_value
    
    # 处理基础类型
    if param_type == "array":
        if not isinstance(param_value, list):
            logger.warning(f"期望数组类型，但获得 {type(param_value)}")
            return param_value
        
        # 处理数组项
        items_schema = param_schema.get("items", {})
        if items_schema:
            # 如果定义了items schema，验证每个元素
            formatted_array = []
            for item in param_value:
                formatted_item = validate_and_format_param(item, items_schema)
                formatted_array.append(formatted_item)
            return formatted_array
        return param_value
    
    elif param_type == "object":
        if not isinstance(param_value, dict):
            logger.warning(f"期望对象类型，但获得 {type(param_value)}")
            return param_value
        
        # 处理对象属性
        properties = param_schema.get("properties", {})
        if properties:
            formatted_obj = {}
            for key, value in param_value.items():
                if key in properties:
                    formatted_obj[key] = validate_and_format_param(value, properties[key])
                else:
                    # 保留未定义的属性
                    formatted_obj[key] = value
            return formatted_obj
        return param_value
    
    elif param_type == "string":
        # 检查枚举值
        if "enum" in param_schema and param_value not in param_schema["enum"]:
            logger.warning(f"参数值 {param_value} 不在枚举值 {param_schema['enum']} 中")
        return str(param_value) if param_value is not None else param_value
    
    elif param_type == "number":
        try:
            return float(param_value)
        except (ValueError, TypeError):
            logger.warning(f"无法将 {param_value} 转换为数字")
            return param_value
    
    elif param_type == "integer":
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.warning(f"无法将 {param_value} 转换为整数")
            return param_value
    
    elif param_type == "boolean":
        if isinstance(param_value, bool):
            return param_value
        return str(param_value).lower() in ("true", "1", "yes")
    
    elif param_type == "null":
        return None
    
    # 如果没有指定类型或类型未知，返回原值
    return param_value


async def call_api(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """通用 API 调用函数，支持完整的 JSON Schema 参数解析"""
    api_config = config["api"]
    
    url = api_config["url"]
    method = api_config.get("method", "GET").upper()
    headers = {}
    
    # 获取参数schema定义
    param_schemas = config.get("parameters", {}).get("properties", {})
    
    # 验证和格式化参数
    formatted_params = {}
    for param_name, param_value in params.items():
        if param_value is None:
            continue
        
        if param_name in param_schemas:
            # 根据schema验证和格式化参数
            formatted_value = validate_and_format_param(param_value, param_schemas[param_name])
            formatted_params[param_name] = formatted_value
        else:
            # 没有schema定义的参数，保持原样
            formatted_params[param_name] = param_value
    
    request_params = formatted_params
    
    # 处理 URL 中的参数替换（例如 {base} -> USD）
    url_params = set()
    for param_name, param_value in request_params.items():
        placeholder = f"{{{param_name}}}"
        if placeholder in url:
            url = url.replace(placeholder, str(param_value))
            url_params.add(param_name)
    
    # 从请求参数中移除已用于URL替换的参数
    request_params = {k: v for k, v in request_params.items() if k not in url_params}
    
    # 记录API调用开始
    logger.info(f"开始API调用: {method} {url}")
    logger.debug(f"原始参数: {params}")
    logger.debug(f"格式化后的参数: {request_params}")
    
    # 详细记录复杂参数类型
    for param_name, param_value in request_params.items():
        if isinstance(param_value, list):
            logger.debug(f"数组参数 '{param_name}': 包含 {len(param_value)} 个元素")
            if param_value and param_name in param_schemas:
                # 显示数组中不同类型的元素
                types_in_array = set(type(item).__name__ for item in param_value)
                logger.debug(f"  - 元素类型: {', '.join(types_in_array)}")
        elif isinstance(param_value, dict):
            logger.debug(f"对象参数 '{param_name}': 包含 {len(param_value)} 个属性")
    
    # 处理认证
    auth_configs = api_config.get("auth", [])
    if isinstance(auth_configs, dict):
        # 向后兼容：如果是单个认证配置，转换为数组
        auth_configs = [auth_configs]
    
    for auth_config in auth_configs:
        auth_type = auth_config.get("type")
        if auth_type:
            AuthHandler.apply_auth(auth_type, headers, request_params, auth_config)
    
    # 记录认证信息（不包含敏感数据）
    if headers:
        logger.debug(f"请求头: {dict(headers)}")
    
    # 发送请求
    async with aiohttp.ClientSession() as session:
        try:
            if method == "GET":
                async with session.get(url, params=request_params, headers=headers) as response:
                    logger.info(f"API响应状态: {response.status}")
                    result = await response.json()
                    logger.debug(f"API响应数据: {result}")
                    return result
            else:
                async with session.request(method, url, json=request_params, headers=headers) as response:
                    logger.info(f"API响应状态: {response.status}")
                    result = await response.json()
                    logger.debug(f"API响应数据: {result}")
                    return result
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            raise


def create_dynamic_tool(tool_config: Dict[str, Any]):
    """根据配置动态创建工具函数，使用 Annotated 传递参数描述"""
    from inspect import Parameter, Signature
    
    tool_name = tool_config["name"]
    description = tool_config["description"]
    parameters = tool_config.get("parameters", {})
    properties = parameters.get("properties", {})
    required_params = parameters.get("required", [])
    
    # 类型映射 - 使用框架支持的基础类型
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": str,  # 使用 str 类型，在运行时解析为 list
        "object": str   # 使用 str 类型，在运行时解析为 dict
    }
    
    # 构建函数签名
    sig_params = [Parameter('context', Parameter.POSITIONAL_OR_KEYWORD, annotation=p.ToolContext)]
    call_params = []
    
    # 分离必需参数和可选参数
    required_param_configs = []
    optional_param_configs = []
    
    for param_name, param_config in properties.items():
        is_required = param_name in required_params
        if is_required:
            required_param_configs.append((param_name, param_config))
        else:
            optional_param_configs.append((param_name, param_config))
    
    # 先处理必需参数，再处理可选参数
    for param_name, param_config in required_param_configs + optional_param_configs:
        # 获取参数类型
        if "oneOf" in param_config or "anyOf" in param_config or "allOf" in param_config:
            # 对于组合类型，使用 str 类型，在运行时解析
            param_type = str
        else:
            param_type = type_mapping.get(param_config.get("type", "string"), str)
        
        is_required = param_name in required_params
        default_value = param_config.get("default") if not is_required else None

        # 从配置中提取参数描述
        param_description = param_config.get("description", f"Parameter {param_name}")
        param_examples = param_config.get("examples", [])
        
        # 增强描述信息，包含类型信息
        if "oneOf" in param_config:
            types = [s.get("type", "any") for s in param_config["oneOf"]]
            param_description += f" (JSON格式，可以是: {', '.join(types)})"
        elif param_config.get("type") == "array" and "items" in param_config:
            items_schema = param_config["items"]
            if "oneOf" in items_schema:
                types = [s.get("type", "any") for s in items_schema["oneOf"]]
                param_description += f" (JSON数组格式，元素可以是: {', '.join(types)})"
            else:
                param_description += " (JSON数组格式)"
        elif param_config.get("type") == "object":
            param_description += " (JSON对象格式)"
        
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
