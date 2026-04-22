import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

MIN_KEY_LENGTH = 16

if not ADMIN_API_KEY:
    logging.warning(
        "ADMIN_API_KEY not set - API endpoints are UNSECURED! "
        "Set ADMIN_API_KEY environment variable for production use."
    )
else:
    if len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
        logging.warning(
            "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
            MIN_KEY_LENGTH,
        )
    logging.info("API key authentication enabled")

POSTGRES_HOST = "1"
POSTGRES_PORT = "2"
POSTGRES_DB = "3"
POSTGRES_USER = "4"
POSTGRES_PASSWORD = "5"
POSTGRES_COLLECTION_NAME = "6"


OPENAI_API_KEY ="7"
OPENAI_BASE_URL = "8"

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
            "embedding_model_dims": 1024
        },
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "qwen3-max", "openai_base_url": OPENAI_BASE_URL}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY , "model": "text-embedding-v3", "openai_base_url": OPENAI_BASE_URL}}
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

app = FastAPI(
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "When the ADMIN_API_KEY environment variable is set, all endpoints require "
        "the `X-API-Key` header for authentication."
    ),
    version="1.0.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate the API key when ADMIN_API_KEY is configured. No-op otherwise."""
    if ADMIN_API_KEY:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not secrets.compare_digest(api_key, ADMIN_API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return api_key


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = Field(None, description="Whether to extract facts from messages. Defaults to True.")
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core').")
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")


class MemoryUpdate(BaseModel):
    text: str = Field(..., description="New content to update the memory with.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")


ENTITY_FILTER_KEYS = ("user_id", "agent_id", "run_id")


def _is_target_scope_memory(
    memory: Dict[str, Any],
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
) -> bool:
    if user_id is not None and memory.get("user_id") != user_id:
        return False

    if agent_id is not None and memory.get("agent_id") != agent_id:
        return False

    if run_id is not None:
        return memory.get("run_id") == run_id

    if memory.get("run_id"):
        return False

    if agent_id is not None:
        return True

    return not memory.get("agent_id")


def _filter_target_scope_memories(
    response: Any,
    user_id: Optional[str],
    agent_id: Optional[str],
    run_id: Optional[str],
) -> Any:
    if isinstance(response, dict) and isinstance(response.get("results"), list):
        filtered_results = [
            memory
            for memory in response["results"]
            if isinstance(memory, dict) and _is_target_scope_memory(memory, user_id, agent_id, run_id)
        ]

        return {**response, "results": filtered_results}

    if not isinstance(response, list):
        return response

    filtered_results = [
        memory
        for memory in response
        if isinstance(memory, dict) and _is_target_scope_memory(memory, user_id, agent_id, run_id)
    ]

    return filtered_results


def _normalize_target_identifier(
    value: Optional[str],
    name: str,
    *,
    allow_blank: bool = False,
) -> Optional[str]:
    if value is None:
        return None

    normalized_value = value.strip()
    if normalized_value:
        return normalized_value

    if allow_blank:
        return None

    raise HTTPException(status_code=400, detail=f"{name} cannot be blank.")


def _split_entity_filters(filters: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    entity_filters = {key: value for key, value in filters.items() if key in ENTITY_FILTER_KEYS}
    metadata_filters = {key: value for key, value in filters.items() if key not in ENTITY_FILTER_KEYS}
    return entity_filters, metadata_filters


def _is_get_all_compatibility_error(error: Exception) -> bool:
    message = str(error)
    return (
        "unexpected keyword argument 'filters'" in message
        or "At least one of 'user_id', 'agent_id', or 'run_id' must be provided" in message
    )


def _get_all_memories_legacy(filters: Dict[str, Any]) -> Any:
    entity_filters, metadata_filters = _split_entity_filters(filters)
    params = dict(entity_filters)
    if metadata_filters:
        params["filters"] = metadata_filters

    return MEMORY_INSTANCE.get_all(**params)


def _get_all_memories(filters: Dict[str, Any]) -> Any:
    try:
        return MEMORY_INSTANCE.get_all(filters=filters)
    except Exception as error:
        if not _is_get_all_compatibility_error(error):
            raise

    return _get_all_memories_legacy(filters)


def _build_memory_filters(
    *,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    memory_filters = dict(filters) if filters else {}
    for key, value in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items():
        if value is not None:
            memory_filters[key] = value
    return memory_filters


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        return _get_all_memories(filters=_build_memory_filters(user_id=user_id, run_id=run_id, agent_id=agent_id))
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/target", summary="Get target-scoped memories")
def get_target_memories(
    user_id: Optional[str] = Query(None, description="User ID whose target-scoped memories should be retrieved."),
    agent_id: Optional[str] = Query(
        None, description="Agent ID. When provided, only this agent-level scope is returned."
    ),
    run_id: Optional[str] = Query(None, description="Run ID. When provided, only this run-level scope is returned."),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve only the requested target scope."""
    normalized_user_id = _normalize_target_identifier(user_id, "user_id")
    normalized_agent_id = _normalize_target_identifier(agent_id, "agent_id", allow_blank=True)
    normalized_run_id = _normalize_target_identifier(run_id, "run_id")

    if not normalized_user_id and not normalized_run_id:
        raise HTTPException(status_code=400, detail="user_id or run_id is required.")

    filters = {}
    if normalized_user_id:
        filters["user_id"] = normalized_user_id
    if normalized_agent_id:
        filters["agent_id"] = normalized_agent_id
    if normalized_run_id:
        filters["run_id"] = normalized_run_id

    try:
        response = _get_all_memories(filters=filters)
        return _filter_target_scope_memories(response, normalized_user_id, normalized_agent_id, normalized_run_id)
    except Exception as e:
        logging.exception("Error in get_target_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Search for memories based on a query."""
    try:
        params = {
            k: v
            for k, v in search_req.model_dump().items()
            if v is not None and k not in {"query", "user_id", "run_id", "agent_id", "filters"}
        }
        params["filters"] = _build_memory_filters(
            user_id=search_req.user_id,
            run_id=search_req.run_id,
            agent_id=search_req.agent_id,
            filters=search_req.filters,
        )
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: MemoryUpdate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory with new content.

    Args:
        memory_id (str): ID of the memory to update
        updated_memory (MemoryUpdate): New content and optional metadata to update the memory with

    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory.text, metadata=updated_memory.metadata)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory(_api_key: Optional[str] = Depends(verify_api_key)):
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
