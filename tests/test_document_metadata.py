from eu_climate_policy_rag.collection.document_metadata import MetadataEnricher


class MetadataEnricherTests:
    def test_enrich_normalizes_eur_lex_html_links(self) -> None:
        sections = {
            "Documentation": [
                {
                    "text": "Proposal for a regulation on the European Climate Law 2040",
                    "href": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52025PC0524",
                }
            ]
        }

        enriched = MetadataEnricher().enrich(sections)
        document = enriched["Documentation"][0]

        assert "/TXT/HTML/" in document["url"]
        assert document["type"] == "regulation"
        assert document["year"] == 2040
        assert document["identifier"] == "52025PC0524"
        assert document["source"] == "eur-lex"
        assert document["topic"] == "climate_law"

    def test_enrich_detects_commission_press_page(self) -> None:
        link = {
            "text": "Press release on the 2040 EU climate target",
            "href": "https://ec.europa.eu/commission/presscorner/detail/en/ip_25_2040",
        }

        document = MetadataEnricher().enrich_link(link)

        assert document["type"] == "press_release"
        assert document["format"] == "press_page"
        assert document["source"] == "commission"
        assert document["topic"] == "climate_target_2040"
