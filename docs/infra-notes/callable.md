`Callable` 是Python类型提示（Type Hints）中的一个类型，用于表示**可调用对象**。

## Callable 的基本语法

```python
from typing import Callable

# 基本语法：Callable[[参数类型], 返回值类型]
Callable[[int, str], bool]  # 接受 int 和 str 参数，返回 bool
```

## 常见用法

### 1. **函数类型注解**
```python
from typing import Callable

# 定义一个函数类型
Handler = Callable[[str], None]

def process_message(message: str) -> None:
    print(f"处理消息: {message}")

# 使用函数类型
def register_handler(handler: Handler) -> None:
    handler("测试消息")

register_handler(process_message)
```

### 2. **回调函数**
```python
from typing import Callable

def execute_with_callback(
    data: list[int], 
    callback: Callable[[int], str]
) -> list[str]:
    return [callback(item) for item in data]

# 使用
result = execute_with_callback(
    [1, 2, 3], 
    lambda x: f"数字: {x}"
)
# 结果: ["数字: 1", "数字: 2", "数字: 3"]
```

### 3. **方法装饰器**
```python
from typing import Callable, Any

def log_function(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_function
def add(a: int, b: int) -> int:
    return a + b
```

### 4. **类方法类型**
```python
from typing import Callable

class EventHandler:
    def __init__(self):
        self.handlers: list[Callable[[str], None]] = []
    
    def add_handler(self, handler: Callable[[str], None]) -> None:
        self.handlers.append(handler)
    
    def trigger_event(self, event: str) -> None:
        for handler in self.handlers:
            handler(event)
```

### 5. **泛型 Callable**
```python
from typing import Callable, TypeVar, Generic

T = TypeVar('T')
R = TypeVar('R')

class FunctionWrapper(Generic[T, R]):
    def __init__(self, func: Callable[[T], R]):
        self.func = func
    
    def execute(self, data: T) -> R:
        return self.func(data)

# 使用
wrapper = FunctionWrapper(lambda x: x * 2)
result = wrapper.execute(5)  # 10
```

### 6. **异步函数类型**
```python
from typing import Callable, Awaitable

# 异步函数类型
AsyncHandler = Callable[[str], Awaitable[None]]

async def async_process(message: str) -> None:
    await asyncio.sleep(1)
    print(f"异步处理: {message}")

async def run_async_handler(handler: AsyncHandler) -> None:
    await handler("异步消息")
```

### 7. **可选参数和关键字参数**
```python
from typing import Callable, Optional

# 带可选参数的函数
OptionalHandler = Callable[[str, Optional[int]], None]

def handler_with_optional(message: str, count: Optional[int] = None) -> None:
    print(f"消息: {message}, 计数: {count}")

# 带关键字参数的函数
KwargsHandler = Callable[..., None]  # 接受任意参数

def flexible_handler(*args, **kwargs) -> None:
    print(f"参数: {args}, 关键字: {kwargs}")
```

## 在 Parlant 项目中的应用

在你的代码中，`Callable` 通常用于：

1. **配置函数类型**：定义配置容器的函数类型
2. **回调函数**：定义事件处理器的类型
3. **工厂函数**：定义创建对象的函数类型
4. **中间件函数**：定义处理请求的函数类型

`Callable` 让代码更加类型安全，IDE 可以提供更好的代码补全和错误检查！