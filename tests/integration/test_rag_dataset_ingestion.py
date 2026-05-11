import inspect

from eu_climate_policy_rag.collection.cleaning import rag_dataset_ingestion
from eu_climate_policy_rag.collection.cleaning.rag_dataset_ingestion import run_cli


def test_run_cli_calls_cleaning_agent(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    output_path = tmp_path / "out.json"

    class Agent:
        instances = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.run_calls = []
            Agent.instances.append(self)

        def run(self, max_turns: int) -> str:
            self.run_calls.append(max_turns)
            return "Cleaned 0 docs."

    monkeypatch.setattr(rag_dataset_ingestion, "CleaningCurationAgent", Agent)

    run_cli(
        input_directory=input_dir,
        output_path=output_path,
        model="gpt-4.1-mini",
        max_turns=5,
    )

    assert len(Agent.instances) == 1
    assert Agent.instances[0].kwargs["model"] == "gpt-4.1-mini"
    assert Agent.instances[0].run_calls == [5]


def test_run_cli_signature_accepts_model() -> None:
    signature = inspect.signature(run_cli)

    assert "model" in signature.parameters
    assert signature.parameters["model"].default == "gpt-5.4-mini"
