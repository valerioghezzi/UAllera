"""Behavioral tests for offline comparison reporting, without model inference."""
import copy
import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jamovi_audit.validation_cases import VALIDATION_CASES


def report_module():
    assert importlib.util.find_spec('tools.render_model_comparison') is not None, 'Report renderer is missing'
    return importlib.import_module('tools.render_model_comparison')


def case(index=0, dataset='baseline'):
    return dict(VALIDATION_CASES[index], dataset=dataset)


def row(planned, observed='coerente', status='complete'):
    return dict(planned, status=status, observed=observed, inference_attempted=True,
                findings=[{'assessment': observed, 'reason': 'Motivo del modello'}] if status == 'complete' else [],
                seconds=1.5, native_reasoning_observed=True)


def protocol(reasoning=128):
    return dict(context_tokens=4096, output_tokens=1024, reasoning_budget=reasoning,
                temperature=0, seed=0, per_part_seconds=180, per_request_seconds=300)


def write_group(path, plan, rows, reasoning=128, name='test'):
    path.mkdir(parents=True, exist_ok=True)
    scope='all' if any(c['dataset']=='baseline' for c in plan) else 'fresh'
    defaults=[dict(c,dataset='baseline') for c in VALIDATION_CASES] if scope=='all' else []
    defaults+=fresh_plan()
    present={c['id'] for c in plan}
    plan=list(plan)+[c for c in defaults if c['id'] not in present]
    fresh=[c for c in plan if c['dataset']=='fresh']
    frozen=json.dumps(fresh).encode()
    (path/'frozen_fresh.json').write_bytes(frozen)
    summary=dict(model=name, profile=f'profile-{reasoning}', protocol=protocol(reasoning),
                 planned=9999, valid=9999, agreement_valid=1,
                 input_hashes={'fresh':hashlib.sha256(frozen).hexdigest()})
    summary['protocol']['case_scope']=scope
    (path/'summary.json').write_text(json.dumps(summary))
    (path/'planned_cases.json').write_text(json.dumps(plan))
    for part in range(1, 8):
        selected=[r for r in rows if r['part']==part]
        if selected:
            (path/f'part_{part}.json').write_text(json.dumps(dict(model=name, profile=summary['profile'], part=part, cases=selected)))
    return path


def fresh_plan():
    return [dict(id=f'F{part}.{i}', part=part, dataset='fresh', kind='synthetic',
                 text=f'Testo sintetico {part} {i}', criterion='Criterio esplicito',
                 expected_assessment='errata' if i==1 else 'coerente', rationale='Motivo atteso')
            for part in range(1, 8) for i in (1, 2)]


class ComparisonReportTests(unittest.TestCase):
    def test_recomputes_valid_and_planned_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=[case(0),case(3)]
            group=report_module().load_result('A',write_group(Path(tmp),plan,[row(plan[0])]))
            self.assertEqual(group['summary']['planned'],84)
            self.assertEqual(group['summary']['valid'],1)
            self.assertEqual(group['summary']['missing'],83)
            self.assertEqual(group['summary']['agreement_valid'],1)
            self.assertEqual(group['summary']['agreement_planned'],1/84)
            self.assertEqual(group['summary']['error_recall_planned'],0)

    def test_undefined_rates_stay_undefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            group=report_module().load_result('A',write_group(Path(tmp),[case(3)],[]))
            self.assertIsNone(group['summary']['agreement_valid'])
            self.assertIsNone(group['summary']['false_alarm_rate_valid_correct'])
            page=report_module().render_html([group],[])
            self.assertIn('non disponibile',page)
            self.assertIn('Mancante',page)

    def test_duplicate_foreign_and_tampered_rows_are_rejected(self):
        for mutation in ('duplicate','foreign','text','expected_assessment','observed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                plan=[case(0)]; rows=[row(plan[0])]
                if mutation=='duplicate': rows.append(copy.deepcopy(rows[0]))
                elif mutation=='foreign': rows[0]['id']='FOREIGN'
                elif mutation=='text': rows[0]['text']='Alterato'
                elif mutation=='expected_assessment': rows[0]['expected_assessment']='errata'
                else: rows[0]['observed']='errata'
                with self.assertRaises(ValueError):
                    report_module().load_result('A',write_group(Path(tmp),plan,rows))

    def test_duplicate_plan_and_tampered_baseline_are_rejected(self):
        for plan in ([case(0),case(0)], [dict(case(0),text='Alterato')]):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                report_module().load_result('A',write_group(Path(tmp),plan,[]))

    def test_reduced_scope_plan_cannot_inflate_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=write_group(Path(tmp),[case(0)],[row(case(0))])
            (directory/'planned_cases.json').write_text(json.dumps([case(0)]))
            with self.assertRaises(ValueError):report_module().load_result('A',directory)

    def test_frozen_fresh_hash_and_planned_identity_are_verified(self):
        for mutation in ('snapshot_hash','planned_label'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                directory=write_group(Path(tmp),fresh_plan(),[])
                filename='frozen_fresh.json' if mutation=='snapshot_hash' else 'planned_cases.json'
                path=directory/filename
                cases=json.loads(path.read_text())
                cases[0]['expected_assessment']='coerente'
                path.write_text(json.dumps(cases))
                with self.assertRaises(ValueError):report_module().load_result('A',directory)

    def test_statistical_context_is_checked_against_baseline_and_saved_rows(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            plan=[dict(case(0), statistical_context=[{'value': 99}])]
            report_module().load_result('A',write_group(Path(tmp),plan,[]))

    def test_saved_row_statistical_context_must_match_its_plan(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            plan=fresh_plan()[:1]
            saved=dict(row(plan[0]), statistical_context=[{'value': 99}])
            report_module().load_result('A',write_group(Path(tmp),plan,[saved]))

    def test_complete_rows_require_saved_assessments_but_model_errors_remain_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=[case(0)]
            saved=row(plan[0], observed='errata')
            saved['findings'].append({'proposal': 'errata', 'observed': 'Secondo motivo'})
            group=report_module().load_result('A',write_group(Path(tmp),plan,[saved]))
            self.assertEqual(group['summary']['valid'],1)
            self.assertEqual(group['summary']['matching_labels'],0)
        for findings in ([], [{'reason': 'Nessuna etichetta salvata'}]):
            with self.subTest(findings=findings), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                plan=[case(0)]
                saved=dict(row(plan[0]), findings=findings)
                report_module().load_result('A',write_group(Path(tmp),plan,[saved]))

    def test_escapes_all_dynamic_text_and_omits_raw_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=fresh_plan()[:1];plan[0]['text']='<script>alert(1)</script>'
            saved=row(plan[0],'errata');saved['findings'][0]['reason']='<img src=x onerror=boom>'
            saved['exchanges']=[{'reasoning_content':'SECRET_CHAIN_OF_THOUGHT'}]
            group=report_module().load_result('<script>name</script>',write_group(Path(tmp),plan,[saved]))
            page=report_module().render_html([group],[])
            self.assertNotIn('<script>',page)
            self.assertIn('&lt;script&gt;',page)
            self.assertIn('&lt;img src=x onerror=boom&gt;',page)
            self.assertNotIn('SECRET_CHAIN_OF_THOUGHT',page)

    def test_paired_gate_excludes_512_and_requires_identical_case_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=fresh_plan(); rows=[row(c,c['expected_assessment']) for c in plan]
            a=report_module().load_result('A',write_group(root/'a',plan,rows))
            b=report_module().load_result('B',write_group(root/'b',plan,rows,reasoning=512))
            self.assertEqual(report_module().paired_fresh([a,b]),[])
            c=report_module().load_result('C',write_group(root/'c',plan,rows))
            pairs=report_module().paired_fresh([a,c])
            self.assertEqual(len(pairs),1)
            self.assertEqual(pairs[0]['completed_pairs'],14)
            self.assertEqual(pairs[0]['left']['agreement_valid'],1)
            c['plan'][0]['criterion']='Criterio diverso'
            self.assertEqual(report_module().paired_fresh([a,c]),[])

    def test_pairs_count_only_completed_pairs_and_keep_time_caps_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=fresh_plan();rows=[row(c,c['expected_assessment']) for c in plan]
            a=report_module().load_result('A',write_group(root/'a',plan,rows))
            b=report_module().load_result('B',write_group(root/'b',plan,rows[:-1]))
            b['protocol']['per_part_seconds']=780
            result=report_module().paired_fresh([a,b])[0]
            self.assertEqual(result['completed_pairs'],13)
            self.assertEqual(result['planned_common'],14)
            self.assertEqual(result['left']['planned'],13)
            self.assertEqual(result['part_time_caps_seconds'],[180,780])
            b['protocol']['output_tokens']=768
            self.assertEqual(report_module().paired_fresh([a,b]),[])

    def test_paired_gate_requires_identical_statistical_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            left_plan=fresh_plan()
            right_plan=copy.deepcopy(left_plan)
            right_plan[0]['statistical_context']=[{'value': 99}]
            groups=[]
            for name, plan in [('A',left_plan),('B',right_plan)]:
                rows=[row(c,c['expected_assessment']) for c in plan]
                groups.append(report_module().load_result(name,write_group(root/name,plan,rows)))
            self.assertEqual(report_module().paired_fresh(groups),[])

    def test_minerva_checks_current_70_case_identity_and_recomputes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'minerva.json'
            saved=[dict(c,engine_status='complete',observed='coerente',findings=[{'proposal':'coerente','observed':'Spiegazione'}]) for c in VALIDATION_CASES]
            p.write_text(json.dumps(dict(cases=saved,model='Minerva',evaluation_profile='reference',metrics={'matching_labels':999})))
            group=report_module().load_minerva(p)
            self.assertEqual(group['summary']['planned'],70)
            self.assertEqual(group['summary']['valid'],70)
            self.assertEqual(group['summary']['matching_labels'],21)
            self.assertEqual(group['protocol']['output_tokens'],768)
            saved[0]['text']='Alterato';p.write_text(json.dumps(dict(cases=saved)))
            with self.assertRaises(ValueError):report_module().load_minerva(p)

    def test_cli_writes_html_and_safe_json_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=[case(0)]
            directory=write_group(root/'input',plan,[row(plan[0])])
            output=root/'report.html'
            report_module().main(['--result',f'Modello={directory}','--output',str(output)])
            self.assertIn('Concordanza sui validi',output.read_text())
            data=json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(data['groups'][0]['summary']['valid'],1)
            self.assertNotIn('exchanges',output.with_suffix('.json').read_text())

    def test_cli_refuses_sidecar_collision_before_either_output_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=write_group(Path(tmp),[case(0)],[row(case(0))])
            source=directory/'summary.json'
            original=source.read_bytes()
            output=directory/'summary.html'
            output.write_text('Existing HTML')
            with self.assertRaises(ValueError):
                report_module().main(['--result',f'A={directory}','--output',str(output)])
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(output.read_text(),'Existing HTML')

    def test_oversized_json_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'large.json'
            with p.open('wb') as handle:handle.truncate(32*1024*1024+1)
            with self.assertRaises(ValueError):report_module().read_json(p)


if __name__=='__main__':unittest.main()
