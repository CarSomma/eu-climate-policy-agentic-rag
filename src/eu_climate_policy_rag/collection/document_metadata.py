"""Document metadata enrichment for links collected from EU policy pages."""

import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlparse

from eu_climate_policy_rag.collection.url_utils import UrlNormalizer
from eu_climate_policy_rag.core.models import DocumentMetadataModel, LinkModel
from eu_climate_policy_rag.core.types import DocumentMetadata, Link


class MetadataEnricher:
    """Convert raw page links into normalized document metadata."""

    def __init__(self, url_normalizer: UrlNormalizer | None = None) -> None:
        self.url_normalizer = url_normalizer or UrlNormalizer()

    def enrich(
        self,
        sections: Mapping[str, Sequence[Link]],
    ) -> dict[str, list[DocumentMetadata]]:
        """Enrich all links in each discovered documentation section."""

        return {
            section: [self.enrich_link(link) for link in links]
            for section, links in sections.items()
        }

    def enrich_link(self, link: Link) -> DocumentMetadata:
        """Normalize one link and infer basic metadata for filtering and search."""

        validated_link = LinkModel.model_validate(link)
        title = validated_link.text
        url = self.url_normalizer.normalize(validated_link.href)
        metadata = DocumentMetadataModel(
            title=title,
            url=url,
            type=self._detect_type(title),
            year=self._extract_year(title, url),
            identifier=self._extract_identifier(url),
            source=self._extract_source(url),
            format=self._detect_format(url),
            topic=self._detect_topic(title),
        )
        return metadata.model_dump()

    @staticmethod
    def _detect_type(title: str) -> str:
        title_lower = title.lower()
        type_keywords = (
            ("staff working document", "staff_document"),
            ("press release", "press_release"),
            ("factsheet", "factsheet"),
            ("questions", "qa"),
            ("q&a", "qa"),
            ("regulation", "regulation"),
            ("proposal", "proposal"),
            ("communication", "communication"),
        )
        for keyword, document_type in type_keywords:
            if keyword in title_lower:
                return document_type
        return "other"

    @staticmethod
    def _extract_year(title: str, url: str) -> int | None:
        match = re.search(r"\b(20\d{2})\b", f"{title} {url}")
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_identifier(url: str) -> str | None:
        for marker in ("CELEX:", "COM:"):
            if marker in url:
                return url.split(marker, maxsplit=1)[-1]

        match = re.search(r"(SWD_\d+_\d+)", url)
        return match.group(1) if match else None

    @staticmethod
    def _extract_source(url: str) -> str:
        domain = urlparse(url).netloc
        if "eur-lex" in domain:
            return "eur-lex"
        if "climate.ec.europa.eu" in domain:
            return "climate-portal"
        if "ec.europa.eu" in domain or "commission.europa.eu" in domain:
            return "commission"
        return "other"

    @staticmethod
    def _detect_format(url: str) -> str:
        url_lower = url.lower()
        if ".pdf" in url_lower:
            return "pdf"
        if "presscorner" in url_lower:
            return "press_page"
        return "html"

    @staticmethod
    def _detect_topic(title: str) -> str:
        title_lower = title.lower()
        if "climate law" in title_lower:
            return "climate_law"
        if "2040" in title_lower:
            return "climate_target_2040"
        if "industrial deal" in title_lower:
            return "clean_industrial_deal"
        return "general"
