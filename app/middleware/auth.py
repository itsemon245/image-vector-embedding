from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from secrets import compare_digest
import os

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"}
            )
            
        # Validate Bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format. Use 'Bearer TOKEN'"}
            )
            
        token = parts[1]
        app_key = os.getenv("APP_KEY")
        if not compare_digest(token, app_key):  # Constant-time comparison
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"}
            )
            
        return await call_next(request) 