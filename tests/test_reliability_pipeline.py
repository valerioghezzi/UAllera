"""Real ZIP integration of cache isolation, evidence and comparison coverage."""
import io
import unittest
import zipfile
from jamovi_audit.pipeline import audit_zip
from jamovi_audit.report import compare_submissions
from test_pipeline import empty_omv, word


def delivery():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        for i,name in enumerate(['A','B']):
            for ext,body in [('omv',empty_omv()),('docx',word())]:
                info=zipfile.ZipInfo(f'Classe/{name}/file.{ext}',date_time=(2020+i,1,2,3,4,6))
                archive.writestr(info,body)
    return data.getvalue()


class ReliabilityPipelineTests(unittest.TestCase):
    def test_cached_results_match_uncached_findings_without_metadata_bleed(self):
        data=delivery();cached=audit_zip(data);fresh=audit_zip(data,use_cache=False)
        self.assertEqual(cached['cache_stats']['hits'],2)
        self.assertEqual(fresh['cache_stats']['hits'],0)
        self.assertEqual([r['findings'] for r in cached['reports']],[r['findings'] for r in fresh['reports']])
        self.assertEqual([r['metadata'] for r in cached['reports']],[r['metadata'] for r in fresh['reports']])
        self.assertNotEqual(cached['reports'][0]['metadata'],cached['reports'][1]['metadata'])
        self.assertTrue(all(r.get('evidence',{}).get('records') for r in cached['reports']))
        self.assertTrue(all(r['compatibility']['omv']['archive_contract_supported'] for r in cached['reports']))
        self.assertIn('Tracciabilità delle evidenze',cached['html'])
        self.assertIn('Compatibilità dei lettori',cached['html'])

    def test_nested_word_sources_participate_in_cross_checks(self):
        text='Ho controllato personalmente il bizzarro andamento mattutino delle risposte perché una distrazione insolita poteva spiegare la scelta inattesa nel questionario.'
        reports=[{'key':'a','label':'A','findings':[]},{'key':'b','label':'B','findings':[]}]
        parsed={k:{'docx':{'evidence_sources':[{'id':'nested','location':'Tabella annidata','text':text}]}} for k in ['a','b']}
        results=compare_submissions(reports,parsed)
        self.assertTrue(any(r['kind']=='testo_coincidente' for r in results))
        self.assertTrue(any('Tabella annidata' in str(r['evidence']) for r in results))

if __name__=='__main__':unittest.main()

class EvidenceReportSizeTests(unittest.TestCase):
    def test_unified_html_keeps_catalog_compact_while_forensic_export_keeps_records(self):
        import json
        from jamovi_audit.evidence import export_evidence_json
        from jamovi_audit.report import render_report
        records=[{'id':'EV-'+str(i).zfill(64),'finding_id':'CHECK.'+str(i),'origin':'deterministic','confidence':{'support_score':None,'missing':['source_anchor_not_available'],'semantic_confidence':None}} for i in range(1000)]
        bundle={'version':'1.0','records':records,'coverage':{'total_records':1000},'artifacts':[]}
        report={'key':'x','label':'X','files':{},'findings':[],'coverage':{},'evidence':bundle}
        html=render_report([report],[])
        self.assertLess(len(html),100000,'The offline HTML must not duplicate the full machine evidence catalog')
        self.assertEqual(len(json.loads(export_evidence_json(bundle))['records']),1000)
