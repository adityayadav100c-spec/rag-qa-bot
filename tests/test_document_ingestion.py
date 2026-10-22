import unittest

try:
    from rag import load_documents
except ModuleNotFoundError as error:
    if error.name not in {
        "langchain_chroma",
        "langchain_core",
        "langchain_openai",
        "langchain_text_splitters",
        "pypdf",
    }:
        raise
    load_documents = None


@unittest.skipUnless(load_documents is not None, "Install rag-qa-bot requirements.")
class DocumentIngestionTests(unittest.TestCase):
    def test_loads_multiple_text_sources_with_metadata(self):
        documents = load_documents()
        sources = {document.metadata["source"] for document in documents}
        self.assertIn("knowledge_base.md", sources)
        self.assertIn("product_guide.txt", sources)
        self.assertTrue(
            all(
                document.page_content.strip()
                and document.metadata.get("source")
                and document.metadata.get("section")
                for document in documents
            )
        )

    def test_markdown_sections_keep_handbook_headings(self):
        documents = load_documents()
        sections = {
            document.metadata["section"]
            for document in documents
            if document.metadata["source"] == "knowledge_base.md"
        }
        self.assertIn("Product: Northstar Travel Mug", sections)
        self.assertIn("Returns and refunds", sections)
        self.assertGreaterEqual(len(sections), 10)


if __name__ == "__main__":
    unittest.main()
