"""Core support code used by collection and RAG modules."""

from eu_climate_policy_rag.core.agent import AbstractAgent
from eu_climate_policy_rag.core.agent_loop import OpenAIResponsesToolLoop
from eu_climate_policy_rag.core.logging_utils import ColoredLogger
from eu_climate_policy_rag.core.models import (
    CleanedDocumentRecordModel,
    IngestionConfigModel,
    IngestionResultModel,
    LinkModel,
    PageSnapshotModel,
    PipelineConfigModel,
    PipelineResultModel,
    PreselectionResultModel,
)

__all__ = [
    "AbstractAgent",
    "CleanedDocumentRecordModel",
    "ColoredLogger",
    "IngestionConfigModel",
    "IngestionResultModel",
    "LinkModel",
    "OpenAIResponsesToolLoop",
    "PageSnapshotModel",
    "PipelineConfigModel",
    "PipelineResultModel",
    "PreselectionResultModel",
]
