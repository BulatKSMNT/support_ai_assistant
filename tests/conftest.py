import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TEST_KEY = "test-service-key-with-at-least-32-characters"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        api_key=TEST_KEY,
        data_dir=tmp_path,
        search_mode="lexical",
        generation_mode="extractive",
        upstream_attempts=2,
        upstream_timeout=2,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as connection:
        connection.headers["Authorization"] = "Bearer " + TEST_KEY
        yield connection


@pytest.fixture
def document():
    return {
        "source_key": "delivery-policy",
        "title": "Порядок доставки",
        "text": "Для доставки менеджер уточняет город назначения и способ получения. "
        "Срок поставки подтверждается после проверки наличия и комплектации.",
    }
