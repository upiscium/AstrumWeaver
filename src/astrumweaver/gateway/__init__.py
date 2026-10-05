"""Client-facing serving gateway adapters."""

from .api import ChatGatewayService, create_chat_router, load_chat_catalog
from .chat import (
    CHAT_CATALOG_SCHEMA,
    CHAT_JOB_SCHEMA,
    CHAT_OPERATION_SCHEMA,
    LLAMA_CPP_CHAT_ADAPTER,
    ChatGatewayError,
    ChatGatewayProfile,
    ChatProfileCatalog,
    CompiledChatRequest,
    compile_chat_request,
    normalize_chat_completion,
)

__all__ = [
    "ChatGatewayService",
    "create_chat_router",
    "load_chat_catalog",
    "CHAT_CATALOG_SCHEMA",
    "CHAT_JOB_SCHEMA",
    "CHAT_OPERATION_SCHEMA",
    "LLAMA_CPP_CHAT_ADAPTER",
    "ChatGatewayError",
    "ChatGatewayProfile",
    "ChatProfileCatalog",
    "CompiledChatRequest",
    "compile_chat_request",
    "normalize_chat_completion",
]
