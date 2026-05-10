import unittest

from eu_climate_policy_rag.collection.document_metadata import MetadataEnricher


class MetadataEnricherTests(unittest.TestCase):
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

        self.assertIn("/TXT/HTML/", document["url"])
        self.assertEqual(document["type"], "regulation")
        self.assertEqual(document["year"], 2040)
        self.assertEqual(document["identifier"], "52025PC0524")
        self.assertEqual(document["source"], "eur-lex")
        self.assertEqual(document["topic"], "climate_law")

    def test_enrich_detects_commission_press_page(self) -> None:
        link = {
            "text": "Press release on the 2040 EU climate target",
            "href": "https://ec.europa.eu/commission/presscorner/detail/en/ip_25_2040",
        }

        document = MetadataEnricher().enrich_link(link)

        self.assertEqual(document["type"], "press_release")
        self.assertEqual(document["format"], "press_page")
        self.assertEqual(document["source"], "commission")
        self.assertEqual(document["topic"], "climate_target_2040")


if __name__ == "__main__":
    unittest.main()
