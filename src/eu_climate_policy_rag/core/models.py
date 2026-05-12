"""Validated data models crossing module boundaries."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from eu_climate_policy_rag.core.types import PipelineFetchResult


class ProjectModel(BaseModel):
    """Base model configuration shared by project models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LinkModel(ProjectModel):
    """Validated browser or scraper link."""

    text: str = Field(min_length=1)
    href: str = Field(min_length=1)
    index: int | None = None

    @field_validator("text", "href")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Trim required link fields and reject blank values."""

        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class ButtonModel(ProjectModel):
    """Validated visible button metadata from Playwright snapshots."""

    text: str = Field(min_length=1)
    index: int | None = None

    @field_validator("text")
    @classmethod
    def strip_button_text(cls, value: str) -> str:
        """Trim button text and reject blank values."""

        value = value.strip()
        if not value:
            raise ValueError("button text cannot be blank")
        return value


class PageSnapshotModel(ProjectModel):
    """Validated snapshot returned by the browser inspection tool."""

    title: str
    url: str
    links: list[LinkModel]
    buttons: list[ButtonModel]
    download_buttons: list[ButtonModel]
    has_downloadable_documents: bool
    has_download_buttons: bool
    document_links: list[LinkModel]
    note: str | None = None


class PreselectionResultModel(ProjectModel):
    """Decision made before persisting fetched Markdown."""

    accepted: bool
    reason: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)


class PipelineResultModel(ProjectModel):
    """Summary returned by the document fetch pipeline."""

    discovered_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    fetched_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    results: list[PipelineFetchResult]


class IngestionResultModel(ProjectModel):
    """Summary returned after cleaning fetched Markdown."""

    input_count: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    output_path: str
    skipped: list[dict[str, str]]


class CleanedDocumentRecordModel(ProjectModel):
    """One cleaned document record written to the RAG JSON dataset."""

    source: str = Field(min_length=1)
    title: str | None = None
    topic: str = Field(min_length=1)
    article: str = Field(default="document", min_length=1)
    file_path: str
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)


class PipelineConfigModel(ProjectModel):
    """Runtime configuration for the document fetch pipeline."""

    source_url: str | HttpUrl
    limit: int | None = Field(default=None, ge=1)
    model: str = Field(default="gpt-5.4-mini", min_length=1)
    max_turns: int = Field(default=12, ge=1)
    output_directory: Path = Path("climate_policy_docs")
    fetch_all: bool = False


class IngestionConfigModel(ProjectModel):
    """Runtime configuration for cleaning-agent ingestion."""

    input_directory: Path = Path("climate_policy_docs")
    output_path: Path = Path("data/eu_climate_policy.json")
    model: str = Field(default="gpt-5.4-mini", min_length=1)
    max_turns: int = Field(default=50, ge=1)


class RagConfigModel(ProjectModel):
    """Runtime configuration for the RAG assistant."""

    data_path: Path = Path("data/eu_climate_policy.json")
    model: Literal[
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.4-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
    ] = "gpt-5.4-mini"
    num_results: int = Field(default=5, ge=1)
    instructions: str = Field(default="", min_length=0)
    max_chars_per_doc: int = Field(default=2000, ge=100)


class RagAnswerModel(ProjectModel):
    """Structured response returned by the RAG assistant."""

    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    sources: list[str]


class SearchDocumentsInputModel(ProjectModel):
    """Validated arguments for the local document search tool."""

    query: str = Field(
        min_length=1,
        description="A concise search query, e.g. '2030 emissions reduction target'.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        """Trim search text and reject blank queries."""

        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class SearchDocumentsResultModel(ProjectModel):
    """Structured output produced by the local document search tool."""

    query: str = Field(min_length=1)
    context: str = Field(min_length=1)
    sources: list[str]
    documents: list[dict[str, object]]


class EmptyToolInputModel(ProjectModel):
    """Validated empty argument object for tools without inputs."""

    pass


class GetPageSnapshotInputModel(ProjectModel):
    """Arguments for the page snapshot fetch tool."""

    url: str = Field(min_length=1)


class ClickAndCaptureInputModel(ProjectModel):
    """Arguments for fetching or clicking a document link."""

    url: str = Field(min_length=1)
    link_href: str = Field(min_length=1)


class ClickDownloadButtonInputModel(ProjectModel):
    """Arguments for clicking a visible download button."""

    url: str = Field(min_length=1)
    button_text: str = Field(min_length=1)


class ConvertToMarkdownInputModel(ProjectModel):
    """Arguments for converting cached content to Markdown."""

    content_id: str = Field(min_length=1)


class SaveContentToFileInputModel(ProjectModel):
    """Arguments for saving cached Markdown to disk."""

    markdown_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    directory: str = Field(default="climate_policy_docs", min_length=1)


class DocumentPathInputModel(ProjectModel):
    """Arguments for tools that operate on one document path."""

    path: str = Field(min_length=1)


class SkipDocumentInputModel(ProjectModel):
    """Arguments for recording a skipped document."""

    path: str = Field(min_length=1)
    reason: str = Field(min_length=1)
