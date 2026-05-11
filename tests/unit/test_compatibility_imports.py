def test_public_root_imports_still_work() -> None:
    from eu_climate_policy_rag import (
        ClimatePolicyAgent,
        CleaningToolbox,
        DocumentFetchAgent,
        DocumentQualityCheck,
    )

    assert ClimatePolicyAgent.__name__ == "ClimatePolicyAgent"
    assert CleaningToolbox.__name__ == "CleaningToolbox"
    assert DocumentFetchAgent.__name__ == "DocumentFetchAgent"
    assert DocumentQualityCheck.__name__ == "DocumentQualityCheck"


def test_supported_subpackage_imports_work() -> None:
    from eu_climate_policy_rag.collection.cleaning.cleaning_toolbox import (
        CleaningToolbox,
    )
    from eu_climate_policy_rag.collection.fetching.fetch_agent import DocumentFetchAgent
    from eu_climate_policy_rag.qa.rag import ClimatePolicyAgent

    assert CleaningToolbox.__module__ == (
        "eu_climate_policy_rag.collection.cleaning.cleaning_toolbox"
    )
    assert (
        DocumentFetchAgent.__module__
        == "eu_climate_policy_rag.collection.fetching.fetch_agent"
    )
    assert ClimatePolicyAgent.__module__ == "eu_climate_policy_rag.qa.rag"
