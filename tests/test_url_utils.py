from eu_climate_policy_rag.collection.url_utils import (
    UrlNormalizer,
    filename_from_url,
    is_document_url,
    is_download_button,
)


class UrlUtilsTests:
    def test_url_normalizer_preserves_existing_html_eur_lex_urls(self) -> None:
        url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32021R1119"

        assert UrlNormalizer().normalize(url) == url

    def test_is_document_url_detects_eu_commission_download_pattern(self) -> None:
        url = "https://commission.europa.eu/document/download/abc-123?filename=climate-law.pdf"

        assert is_document_url(url)
        assert filename_from_url(url) == "climate-law.pdf"

    def test_is_download_button_is_multilingual(self) -> None:
        assert is_download_button("Download PDF")
        assert is_download_button("Télécharger")
