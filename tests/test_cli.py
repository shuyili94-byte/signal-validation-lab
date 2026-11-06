"""End-to-end commands, output integrity and protocol provenance."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.test-work'


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(exist_ok=True)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'signal_validation', *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8')

    def test_demo_runs_the_research_pipeline_and_labels_all_results_synthetic(self):
        # Catches a demo that bypasses WFA or presents fabricated market evidence.
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            out = Path(folder) / 'run'
            proc = self.run_cli('demo', '--output', out, '--bootstrap-samples', '20')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['data_kind'], 'synthetic')
            self.assertEqual(result['mode'], 'demo')
            self.assertGreater(result['walk_forward']['aggregate']['fold_count'], 0)
            self.assertTrue((out / 'report.md').is_file())
            self.assertTrue((out / 'protocol.json').is_file())
            self.assertTrue((out / 'synthetic.csv').is_file())

    def test_prior_report_cannot_be_overwritten(self):
        # Catches accidental destruction of an earlier experiment.
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            out = Path(folder)
            (out / 'result.json').write_text('keep me', encoding='utf-8')
            proc = self.run_cli('demo', '--output', out, '--bootstrap-samples', '20')
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual((out / 'result.json').read_text(encoding='utf-8'), 'keep me')

    def test_edited_protocol_is_rejected_before_a_validation_run(self):
        # Catches silently changed candidate sets being passed off as a frozen protocol.
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            root = Path(folder)
            made = self.run_cli('demo', '--output', root / 'demo', '--bootstrap-samples', '20')
            self.assertEqual(made.returncode, 0, made.stderr)
            protocol_path = root / 'demo' / 'protocol.json'
            protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
            protocol['gap_days'] = 0
            protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
            proc = self.run_cli('walk-forward', '--input', root / 'demo' / 'synthetic.csv',
                                '--protocol', protocol_path, '--output', root / 'changed')
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse((root / 'changed' / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
