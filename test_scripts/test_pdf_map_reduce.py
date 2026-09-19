"""Offline checks for the large-PDF map-reduce helpers."""

import unittest
from unittest.mock import patch

from src import agent


class PdfMapReduceTests(unittest.TestCase):
    def test_section_chunks_split_oversized_sections(self):
        text = "Abstract\nShort overview.\n\n1 Introduction\n" + ("evidence " * 200)
        with patch.object(agent, "logger"):
            chunks = agent._build_pdf_chunks(text, target_tokens=30, overlap_tokens=3)

        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0][0], "Abstract")
        self.assertTrue(any(label.startswith("Introduction (part") for label, _ in chunks))
        self.assertTrue(all(agent._count_pdf_tokens(content) <= 35 for _, content in chunks))

    def test_large_pdf_is_recursively_reduced_before_final_brief(self):
        text = "\n\n".join(
            f"{name}\n" + ("result method limitation " * 70)
            for name in ("Abstract", "Introduction", "Methods", "Results", "Discussion")
        )
        calls: list[str] = []

        def fake_invoke(_model, *, instruction, text, **_kwargs):
            calls.append(instruction)
            return "brief " * 50

        with patch.dict("os.environ", {
            "PDF_MAP_CHUNK_TOKENS": "80",
            "PDF_MAP_CHUNK_OVERLAP_TOKENS": "5",
            "PDF_REDUCE_INPUT_TOKENS": "800",
            "PDF_SUMMARY_MAX_OUTPUT_TOKENS": "200",
        }, clear=False), patch.object(agent, "logger"), patch.object(
            agent, "_model", return_value=object()
        ), patch.object(agent, "_invoke_pdf_summary", side_effect=fake_invoke):
            brief = agent._summarize_large_pdf(text)

        self.assertTrue(brief.strip())
        self.assertGreater(len(calls), 6)  # Multiple map calls plus at least one reduce.
        self.assertIn("Synthesize", calls[-1])


if __name__ == "__main__":
    unittest.main()
