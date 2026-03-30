"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app import create_app
from environment import OpenEnv, TASKS


@pytest.fixture
def sample_env() -> OpenEnv:
    return OpenEnv()


@pytest.fixture
def test_client() -> TestClient:
    with TestClient(create_app(OpenEnv())) as client:
        yield client


@pytest.fixture
def mock_tasks() -> dict:
    return TASKS
