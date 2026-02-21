import json
import os
import tempfile
from typing import Dict, TextIO

from simplebayes.constants import CATEGORY_PATTERN
from simplebayes.errors import (
    InvalidModelStateError,
    PersistencePathError,
    UnsupportedModelVersionError,
)

PERSISTED_MODEL_VERSION = 1
DEFAULT_MODEL_FILE_PATH = "/tmp/simplebayes-model.json"


def dump_model_state(stream: TextIO, model_state: Dict) -> None:
    if stream is None:
        raise InvalidModelStateError("destination stream is required")

    json.dump(model_state, stream)


def load_model_state(stream: TextIO) -> Dict:
    if stream is None:
        raise InvalidModelStateError("source stream is required")

    try:
        state = json.load(stream)
    except json.JSONDecodeError as exc:
        raise InvalidModelStateError("unable to decode persisted model") from exc

    if not isinstance(state, dict):
        raise InvalidModelStateError("persisted model root must be an object")

    return state


def resolve_model_path(path: str = "") -> str:
    resolved_path = path.strip() if path else DEFAULT_MODEL_FILE_PATH
    if not os.path.isabs(resolved_path):
        raise PersistencePathError("model file path must be absolute")
    return resolved_path


def save_model_state_to_file(path: str, model_state: Dict) -> None:
    resolved_path = resolve_model_path(path)
    model_directory = os.path.dirname(resolved_path)
    os.makedirs(model_directory, exist_ok=True)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=model_directory,
            prefix=".simplebayes-",
            suffix=".tmp",
        ) as temp_file:
            temp_path = temp_file.name
            dump_model_state(temp_file, model_state)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, resolved_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def load_model_state_from_file(path: str) -> Dict:
    resolved_path = resolve_model_path(path)
    with open(resolved_path, "r", encoding="utf-8") as source_file:
        return load_model_state(source_file)


def validate_model_state(state: Dict) -> None:
    version = state.get("version")
    if version != PERSISTED_MODEL_VERSION:
        raise UnsupportedModelVersionError(f"unsupported model version: {version}")

    categories = state.get("categories")
    if not isinstance(categories, dict):
        raise InvalidModelStateError("persisted categories must be an object")

    for category_name, category_state in categories.items():
        if (
            not isinstance(category_name, str)
            or not category_name
            or not CATEGORY_PATTERN.match(category_name)
        ):
            raise InvalidModelStateError("invalid category name in persisted model")

        if not isinstance(category_state, dict):
            raise InvalidModelStateError("invalid category payload in persisted model")

        tally = category_state.get("tally")
        tokens = category_state.get("tokens")

        if not isinstance(tally, int) or tally < 0:
            raise InvalidModelStateError("invalid category tally in persisted model")

        if not isinstance(tokens, dict):
            raise InvalidModelStateError("invalid token map in persisted model")

        token_sum = 0
        for token, count in tokens.items():
            if not isinstance(token, str) or not token:
                raise InvalidModelStateError("invalid token name in persisted model")
            if not isinstance(count, int) or count <= 0:
                raise InvalidModelStateError("invalid token count in persisted model")
            token_sum += count

        if token_sum != tally:
            raise InvalidModelStateError("token tally mismatch in persisted model")
