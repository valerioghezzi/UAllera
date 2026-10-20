"""Forensic CLI output is local, atomic and explicitly separate from HTML."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit_cli
from test_reliability_pipeline import delivery


class ForensicCliTests(unittest.TestCase):
    def test_forensic_export_and_cache_option_produce_reviewable_local_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.zip';source.write_bytes(delivery())
            html=root/'report.html';forensic=root/'forensic.json'
            with patch('sys.argv',['audit_cli.py',str(source),'-o',str(html),'--forensic',str(forensic),'--no-cache']),contextlib.redirect_stdout(io.StringIO()):
                audit_cli.main()
            self.assertTrue(html.is_file());result=json.loads(forensic.read_text())
            self.assertEqual(result['schema_version'],'1.0');self.assertEqual(len(result['students']),2)
            self.assertEqual(result['cache_stats']['hits'],0)
            self.assertNotIn('columns',json.dumps(result))
            self.assertFalse(list(root.glob('.rapporto-*')))

    def test_forensic_path_must_not_overwrite_source_or_html(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.zip';original=delivery();source.write_bytes(original)
            for target in [source,root/'report.html']:
                with patch('sys.argv',['audit_cli.py',str(source),'-o',str(root/'report.html'),'--forensic',str(target)]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):audit_cli.main()
            self.assertEqual(source.read_bytes(),original)

if __name__=='__main__':unittest.main()
