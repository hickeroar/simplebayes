import io
import json
import os
import tempfile

import pytest

from simplebayes import SimpleBayes
from simplebayes.errors import (
    InvalidCategoryError,
    InvalidModelStateError,
    PersistencePathError,
    UnsupportedModelVersionError,
)
from simplebayes.persistence import (
    PERSISTED_MODEL_VERSION,
    dump_model_state,
    load_model_state_from_file,
    load_model_state,
    resolve_model_path,
    save_model_state_to_file,
    validate_model_state,
)


def test_save_and_load_round_trip_stream():
    classifier = SimpleBayes()
    classifier.train("spam", "buy now limited offer")
    classifier.train("ham", "team schedule meeting")

    destination = io.StringIO()
    classifier.save(destination)

    destination.seek(0)
    loaded = SimpleBayes()
    loaded.load(destination)

    result = loaded.classify_result("limited offer")
    assert result.category == "spam"
    assert result.score > 0


def test_save_and_load_round_trip_file():
    classifier = SimpleBayes()
    classifier.train("alpha", "one two three")

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "model.json")
        classifier.save_to_file(path)

        loaded = SimpleBayes()
        loaded.load_from_file(path)
        assert loaded.tally("alpha") == 3


def test_resolve_model_path_requires_absolute():
    with pytest.raises(PersistencePathError):
        resolve_model_path("relative/path.json")


def test_load_model_state_invalid_json():
    with pytest.raises(InvalidModelStateError):
        load_model_state(io.StringIO("{not json"))


def test_load_model_state_none_and_non_object():
    with pytest.raises(InvalidModelStateError):
        load_model_state(None)  # type: ignore[arg-type]

    with pytest.raises(InvalidModelStateError):
        load_model_state(io.StringIO("[]"))


def test_dump_model_state_requires_stream():
    with pytest.raises(InvalidModelStateError):
        dump_model_state(None, {})  # type: ignore[arg-type]


def test_validate_model_state_errors():
    with pytest.raises(UnsupportedModelVersionError):
        validate_model_state({"version": 999, "categories": {}})

    with pytest.raises(InvalidModelStateError):
        validate_model_state({"version": PERSISTED_MODEL_VERSION, "categories": []})

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": {"tally": 1, "tokens": {"": 1}}},
            },
        )

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": {"tally": 2, "tokens": {"token": 1}}},
            },
        )

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": []},
            },
        )

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": {"tally": -1, "tokens": {"token": 1}}},
            },
        )

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": {"tally": 1, "tokens": []}},
            },
        )

    with pytest.raises(InvalidModelStateError):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"alpha": {"tally": 1, "tokens": {"token": 0}}},
            },
        )


def test_load_rejects_invalid_payload():
    classifier = SimpleBayes()
    state = {
        "version": PERSISTED_MODEL_VERSION,
        "categories": {"bad category": {"tally": 1, "tokens": {"x": 1}}},
    }
    payload = io.StringIO(json.dumps(state))
    with pytest.raises(InvalidModelStateError, match="invalid category name"):
        classifier.load(payload)


def test_category_validation_consistent_between_runtime_and_persistence():
    for category in ["alpha-1", "A_B", "x" * 64]:
        assert SimpleBayes.normalize_category(category) == category
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {category: {"tally": 1, "tokens": {"token": 1}}},
            },
        )

    with pytest.raises(InvalidCategoryError):
        SimpleBayes.normalize_category("bad category")
    with pytest.raises(InvalidModelStateError, match="invalid category name"):
        validate_model_state(
            {
                "version": PERSISTED_MODEL_VERSION,
                "categories": {"bad category": {"tally": 1, "tokens": {"token": 1}}},
            },
        )


def test_save_model_state_cleanup_on_replace_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.json")
        state = {"version": PERSISTED_MODEL_VERSION, "categories": {}}

        def _raise_replace(_src, _dst):
            raise RuntimeError("replace failed")

        monkeypatch.setattr("simplebayes.persistence.os.replace", _raise_replace)

        with pytest.raises(RuntimeError):
            save_model_state_to_file(model_path, state)


def test_load_model_state_from_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model_state_from_file("/tmp/simplebayes-missing-model.json")
