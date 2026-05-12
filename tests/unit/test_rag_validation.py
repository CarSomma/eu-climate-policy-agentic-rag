"""Unit tests for RAG agent validation and error handling."""

import pytest
from openai import OpenAI
from pydantic import ValidationError

from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent


def test_rag_config_model_validates_supported_models() -> None:
    """RagConfigModel should use Literal type for model validation."""
    from eu_climate_policy_rag.core.models import RagConfigModel

    # Valid models should work
    valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-5.4-mini"]
    for model in valid_models:
        config = RagConfigModel(model=model)
        assert config.model == model

    # Invalid models should raise ValidationError
    invalid_models = ["gpt-5", "claude-3", "invalid-xyz"]
    for model in invalid_models:
        with pytest.raises(ValidationError, match="Input should be"):
            RagConfigModel(model=model)


def test_agent_uses_pydantic_validation_for_model() -> None:
    """Agent should validate model through RagConfigModel, not manual check."""
    # Invalid model should raise Pydantic ValidationError
    with pytest.raises(ValidationError, match="Input should be"):
        ClimatePolicyAgent(
            documents=[],
            openai_client=None,
            model="invalid-model",
        )


def test_agent_raises_on_invalid_model() -> None:
    """Agent should fail fast with clear error for invalid model names."""
    invalid_models = ["gpt-5", "claude-3", "llama-2", "nonexistent-model"]

    for model in invalid_models:
        with pytest.raises(ValidationError):
            ClimatePolicyAgent(
                documents=[],
                openai_client=OpenAI(api_key="test"),
                model=model,
            )


def test_agent_accepts_valid_models() -> None:
    """Agent should accept known valid OpenAI model names."""
    valid_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.4-mini",
        "gpt-4-turbo",
    ]

    for model in valid_models:
        # Should not raise
        agent = ClimatePolicyAgent(
            documents=[],
            openai_client=None,  # Don't make actual API calls
            model=model,
        )
        assert agent.config.model == model


def test_cli_validates_model_before_execution(tmp_path) -> None:
    """CLI should validate model parameter before starting agent."""
    from typer.testing import CliRunner
    from eu_climate_policy_rag.qa.rag import app

    runner = CliRunner()

    # Create minimal test data with all required fields
    test_data = tmp_path / "test.json"
    test_data.write_text(
        '[{"source": "test", "topic": "climate_law", "article": "document", '
        '"text": "content", "file_path": "test.md", "content_hash": "abc"}]'
    )

    result = runner.invoke(
        app,
        ["What is the target?", "--data", str(test_data), "--model", "invalid-model-xyz"],
    )

    # Should fail with clear error
    assert result.exit_code != 0
    # Check exception message (Pydantic ValidationError)
    assert result.exception is not None
    assert "validation error" in str(result.exception).lower() or "input should be" in str(result.exception).lower()
