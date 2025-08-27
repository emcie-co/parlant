# API Hardening

Parlant provides a robust authorization and rate limiting system to protect your API from unauthorized access and abuse. This guide explains how to implement custom authorization policies and rate limiters to secure your production deployment.

## Overview

The API hardening system consists of two main components:

1. **Authorization Policies** - Control who can access what resources and perform which actions
2. **Rate Limiters** - Prevent abuse by limiting the frequency of requests

Both components work together to provide comprehensive API protection, with support for different limits based on access tokens or user tiers.

## Authorization Policies

### Understanding the AuthorizationPolicy Abstract Class

All authorization policies inherit from the `AuthorizationPolicy` abstract base class, which defines three key methods:

```python
class AuthorizationPolicy:
    @abstractmethod
    async def check_permission(
        self,
        request: fastapi.Request,
        permission: AuthorizationPermission
    ) -> bool:
        """Check if the request has permission to perform the action"""
        ...

    @abstractmethod
    async def check_rate_limit(
        self,
        request: fastapi.Request,
        permission: AuthorizationPermission
    ) -> bool:
        """Check if the request is within rate limits"""
        ...

    async def authorize(
        self,
        request: fastapi.Request,
        permission: AuthorizationPermission
    ) -> None:
        """Combined authorization check (permission + rate limit)"""
        # This method usually isn't overriden, as its default implementation
        # calls the two abstract methods in sequence and raises an authorization
        # error if anything is denied.
        ...
```

### Authorization Permissions

Parlant defines a comprehensive set of permissions as an enum covering all API operations:

- Agent operations (create, read, update, delete)
- Customer management
- Session handling
- And many more...

### Built-in Authorization Policies

#### DevelopmentAuthorizationPolicy
Allows all actions - suitable for development environments only:

```python
class DevelopmentAuthorizationPolicy(AuthorizationPolicy):
    async def check_permission(
        self,
        request: fastapi.Request,
        permission: AuthorizationPermission
    ) -> bool:
        return True

    async def check_rate_limit(
        self,
        request: fastapi.Request,
        permission: AuthorizationPermission
    ) -> bool:
        return True
```

#### ProductionAuthorizationPolicy
Implements stricter controls for production use with configurable rules.

## Implementing Custom Authorization Policies

When you implement your own authorization policy in real-world deployments, you typically want to extend the existing production policy rather than building from scratch. The recommended approach is to subclass `ProductionAuthorizationPolicy` and customize it for your specific needs.

Here's a reference implementation that demonstrates how to create a custom policy with JWT authentication:

```python
import parlant.sdk as p

import jwt
from fastapi import HTTPException
from limits import RateLimitItemPerMinute, RateLimitItemPerHour
from limits.storage import RedisStorage
from limits.strategies import SlidingWindowCounterRateLimiter

class CustomAuthorizationPolicy(p.ProductionAuthorizationPolicy):
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        super().__init__()
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def _extract_token(self, request: fastapi.Request) -> dict | None:
        """Extract and validate JWT token from request"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.JWTError:
            # Raise 403 for invalid tokens, None for missing tokens is OK
            raise HTTPException(
                status_code=403,
                detail="Invalid access token"
            )

    async def check_permission(
        self,
        request: fastapi.Request,
        operation: p.Operation
    ) -> bool:
        """Enhanced permission checking with M2M token support"""
        token_payload = await self._extract_token(request)

        # If we have a valid M2M (machine-to-machine) token, allow additional operations
        if token_payload and token_payload.get("type") == "m2m":
            m2m_operations = {
                # Allow M2M tokens to perform administrative operations
                p.Operation.CREATE_AGENT,
                p.Operation.READ_AGENT,
                p.Operation.UPDATE_AGENT,
                p.Operation.DELETE_AGENT,
                p.Operation.CREATE_CUSTOMER,
                p.Operation.READ_CUSTOMER,
                p.Operation.UPDATE_CUSTOMER,
                p.Operation.DELETE_CUSTOMER,
                p.Operation.CREATE_CUSTOMER_SESSION,
                p.Operation.LIST_SESSIONS,
                p.Operation.UPDATE_SESSION,
                p.Operation.DELETE_SESSION,
                # Add other operations your M2M integration needs
            }

            if operation in m2m_operations:
                return True

        # For all other cases, delegate to the parent ProductionAuthorizationPolicy
        return await super().check_permission(request, operation)
```

## Rate Limiting Customization Options

The `ProductionAuthorizationPolicy` provides several ways to customize rate limiting behavior:

### 1. Override the Default Rate Limiter (Recommended)

The most common approach is to override `self.default_limiter` with your own `BasicRateLimiter` configuration. **Note that BasicRateLimiter limits apply per IP address** - so when you configure `RateLimitItemPerMinute(100)`, it means 100 requests per minute per IP address.

```python
from limits import RateLimitItemPerMinute, RateLimitItemPerHour
from limits.storage import RedisStorage
from limits.strategies import SlidingWindowCounterRateLimiter

# Example with Redis storage and custom limits
class CustomAuthorizationPolicy(p.ProductionAuthorizationPolicy):
    def __init__(self, ...):
        super().__init__()

        # ...

        self.default_limiter = p.BasicRateLimiter(
            rate_limit_item_per_operation={
                # Use the default rate limit for most operations
                **self.default_limiter.rate_limit_item_per_operation,
                # Override specific operations with custom limits
                p.Operation.READ_SESSION: RateLimitItemPerMinute(200),
                p.Operation.LIST_EVENTS: RateLimitItemPerMinute(1000),
            },
            # Use a custom storage backend (e.g., Redis)
            storage=RedisStorage("redis://localhost:6379"),
            # Use a custom window strategy
            limiter_type=SlidingWindowCounterRateLimiter,
        )
```

The `BasicRateLimiter` uses the `limits` library and supports:
- **Rate limit items**: `RateLimitItemPerMinute(n)`, `RateLimitItemPerSecond(n)`, `RateLimitItemPerHour(n)`
- **Storage options**: `RedisStorage()`, `MemoryStorage()`, and others from the limits library
- **Limiter strategies**: `MovingWindowRateLimiter`, `FixedWindowRateLimiter`, `SlidingWindowCounterRateLimiter`

For complete control, you can implement your own `RateLimiter` from scratch by subclassing the abstract `RateLimiter` class and assigning it to `self.default_limiter`.

### 2. Custom Limiter Functions for Specific Operations

Use `self.specific_limiters` to provide custom rate limiting functions for particular operations. These are functions that take a request and operation and return a boolean indicating whether the rate is within the limit.

```python
class CustomAuthorizationPolicy(p.ProductionAuthorizationPolicy):
    def __init__(self, ...):
        super().__init__()

        # ...

        self.specific_limiters[p.Operation.DELETE_AGENT] = self._custom_delete_limiter

    async def _custom_delete_limiter(
        self,
        request: fastapi.Request,
        operation: p.Operation
    ) -> bool:
        # Implement your custom logic here
        ...
```

If you need complete control over both permission checking and rate limiting, you can also subclass the abstract `AuthorizationPolicy` directly and implement all methods from scratch. This gives you full flexibility but requires more implementation work. The approach shown above is recommended for most use cases as it builds on the robust foundation of `ProductionAuthorizationPolicy`.

## Integrating Your Custom Authorization Policy

### Using configure_container

Integrate your custom authorization policy and rate limiter with your Parlant agent:

```python
async def configure_container(
    container: p.Container
) -> p.Container:
    container[p.AuthorizationPolicy] = CustomAuthorizationPolicy(
        secret_key="your-jwt-secret-key",
        algorithm="HS256",
    )

    return container
```
```python
async def main():
    # Create Parlant server with custom authorization
    async with p.Server(
        configure_container=configure_container,
    ) as server:
        # Your agent logic here
        await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
```
