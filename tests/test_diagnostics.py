"""Offline compatibility and allowlisted forensic diagnostics."""
import json
import copy
import unittest
from jamovi_audit import diagnostics
from jamovi_audit.report import ComparisonResults, compare_submissions
from jamovi_audit.metadata import metadata_cross_checks


class DiagnosticsTests(unittest.TestCase):
    def test_unknown_archive_and_word_omissions_remain_explicit(self):
        documents={'omv':{'sha256':'a'*64,'manifest':{'jamovi-Archive-Version':'12.0','Created-By':'jamovi future'},'errors':['unsupported']},'docx':{'sha256':'b'*64,'errors':[],'evidence_coverage':{'scope':'body','omissions':[{'kind':'header','reason':'not read'}]}}}
        result=diagnostics.compatibility_snapshot(documents)
        self.assertFalse(result['omv']['archive_contract_supported'])
        self.assertEqual(result['omv']['observed_archive'],'12.0')
        self.assertEqual(result['word']['omissions'][0]['kind'],'header')
        self.assertFalse(result['network_check_performed'])

    def test_known_archive_does_not_certify_application_version(self):
        result=diagnostics.compatibility_snapshot({'omv':{'manifest':{'jamovi-Archive-Version':'11.0','Created-By':'jamovi 99.0'},'errors':[]}})
        self.assertTrue(result['omv']['archive_contract_supported'])
        self.assertFalse(result['omv']['application_version_certified'])

    def test_comparison_snapshot_releases_raw_data_but_preserves_provenance(self):
        documents={'omv':{'sha256':'a'*64,'content_sha256':'b'*64,'columns':{'x':{'data':list(range(100))}},'analyses':[{'big':'payload'}],'metadata':{'author':'A'},'zip_metadata':{'file':{}},'errors':[]},'docx':{'paragraphs':[{'text':'legacy'}],'tables':[],'evidence_sources':[{'id':'source','text':'ordered','location':'P1'}],'metadata':{'creator':'A'},'formatting':{'unused':'big'}}}
        compact=diagnostics.comparison_snapshot(documents)
        self.assertNotIn('columns',compact['omv']);self.assertNotIn('analyses',compact['omv'])
        self.assertEqual(compact['omv']['metadata'],documents['omv']['metadata'])
        self.assertEqual(compact['docx']['evidence_sources'],documents['docx']['evidence_sources'])
        self.assertNotIn('formatting',compact['docx'])

    def test_forensic_export_excludes_config_credentials_and_full_documents(self):
        bundle={'version':'1.0','records':[],'artifacts':{},'coverage':{},'contradictions':[],'protocol':{},'limits':[]}
        result={'summary':{'student_count':1,'version':'test','secret':'forged'},'reports':[{'key':'x','label':'Student','evidence':bundle,'compatibility':{},'config':{'api_key':'PRIVATE'}}],'comparisons':[],'html':'WHOLE_DOC','config':{'api_key':'PRIVATE'}}
        exported=diagnostics.forensic_snapshot(result)
        text=json.dumps(exported)
        self.assertNotIn('PRIVATE',text);self.assertNotIn('WHOLE_DOC',text);self.assertNotIn('forged',text)
        self.assertEqual(exported['students'][0]['key'],'x')
        self.assertEqual(exported['schema_version'],'1.0')

    def test_forensic_nested_allowlists_remove_private_extras_without_losing_counts(self):
        secret={'config':{'api_key':'NESTED_PRIVATE_SENTINEL'},'api_key':'NESTED_PRIVATE_SENTINEL'}
        bundle={'version':'1.0','records':[],'artifacts':[],'coverage':{},'contradictions':[],'protocol':{},'limits':[]}
        compatibility={'reader_contract':'contract','network_check_performed':False,**secret,
            'omv':{'observed_archive':'11.0','archive_contract_supported':True,**secret},
            'word':{'scope':'body','reader_errors':['kept error'],**secret,
                    'omissions':[{'kind':'header','part':'word/header1.xml','location':'header',
                                  'reason':'kept omission',**secret}]}}
        cache={'hits':2,'misses':1,'bytes_accounting':'unique_deepcopy_graph+json+entry_overhead',**secret}
        comparisons=ComparisonResults()
        comparisons.append({'students':['a','b'],'labels':['A','B'],'kind':'testo_simile',
            'evidence':[{'component':'Passaggi','left':'citazione A','right':'citazione B',**secret}],
            'evidence_count':9,'corroborates':['testo_parziale'],'limits':'kept limits',**secret})
        comparisons.diagnostics={'pairs_total':1,**secret,
            'cutoffs':{'blocks_omitted':3,**secret},
            'thresholds':{'near_sequence_ratio':.72,**secret},
            'pairs_flagged_by_rule':{'testo_simile':1,**secret},
            'metadata':{'close_seconds':120,'timestamp_only_candidates':False,**secret}}
        result={'reports':[{'key':'a','label':'A','evidence':bundle,'compatibility':compatibility}],
                'summary':{'student_count':1,'cache_stats':cache},'cache_stats':cache,'comparisons':comparisons}
        before=copy.deepcopy(result)
        exported=diagnostics.forensic_snapshot(result)
        self.assertNotIn('NESTED_PRIVATE_SENTINEL',json.dumps(exported))
        self.assertEqual(exported['cache_stats']['hits'],2)
        self.assertEqual(exported['summary']['cache_stats']['misses'],1)
        self.assertEqual(exported['comparison_diagnostics']['cutoffs']['blocks_omitted'],3)
        self.assertEqual(exported['comparison_diagnostics']['thresholds']['near_sequence_ratio'],.72)
        self.assertEqual(exported['students'][0]['compatibility']['word']['omissions'][0]['reason'],'kept omission')
        self.assertEqual(exported['comparisons'][0]['evidence_count'],9)
        self.assertEqual(exported['comparisons'][0]['corroborates'],['testo_parziale'])
        self.assertEqual(result,before)

    def test_word_snapshot_drops_duplicate_legacy_views_and_run_payloads(self):
        source={'id':'node/1','location':'Paragrafo 1','text':'contenuto citabile','kind':'paragraph','runs':[{'unused':'big'}]}
        documents={'docx':{'sha256':'b'*64,'evidence_sources':[source],
                    'paragraphs':[{'text':'vista duplicata','runs':[{'unused':'big'}]}],
                    'tables':[{'rows':[['vista duplicata']]}],
                    'metadata':{'creator':'Autore','dataSet':{'unused':'big'}},
                    'metadata_summary':{'unused':'big'},'formatting':{'unused':'big'},'errors':[]}}
        compact=diagnostics.comparison_snapshot(documents)['docx']
        self.assertNotIn('paragraphs',compact)
        self.assertNotIn('tables',compact)
        self.assertNotIn('metadata_summary',compact)
        self.assertNotIn('dataSet',compact['metadata'])
        self.assertEqual(compact['evidence_sources'],[{'id':'node/1','location':'Paragrafo 1','text':'contenuto citabile'}])
        self.assertIn('runs',source)

    def test_legacy_snapshot_and_metadata_preserve_actual_comparison_results(self):
        text='Ho controllato il bizzarro andamento mattutino delle risposte perché una distrazione insolita poteva spiegare la scelta inattesa nel questionario.'
        reports=[{'key':key,'label':key,'findings':[],'paths':{'docx':[key+'/file.docx']}} for key in ('a','b')]
        docs={'docx':{'sha256':'d'*64,'paragraphs':[{'index':7,'text':text,'runs':[{'font':'unused'}]}],
                     'tables':[{'index':0,'rows':[[text]],'cells':[{'unused':'big'}]}],
                     'metadata':{'creator':'Nome specifico','modified':'2026-10-20T12:00:00Z','unused':{'big':'payload'}},
                     'zip_metadata':{'doc.xml':{'iso':'2026-10-20T11:00:00','unused':'big'}},'errors':[]},
              'omv':{'sha256':'a'*64,'content_sha256':'c'*64,'metadata':{'author':'Nome specifico'},
                     'zip_metadata':{'meta':{'iso':'2026-10-20T11:00:00'}},'columns':{'big':list(range(100))}}}
        parsed={key:copy.deepcopy(docs) for key in ('a','b')}
        compact={key:diagnostics.comparison_snapshot(value) for key,value in parsed.items()}
        original=compare_submissions(reports,parsed)
        reduced=compare_submissions(reports,compact)
        self.assertEqual(reduced,original)
        self.assertEqual(reduced.diagnostics,original.diagnostics)
        self.assertEqual(metadata_cross_checks(reports,compact,reduced),metadata_cross_checks(reports,parsed,original))
        self.assertEqual(compact['a']['docx']['paragraphs'],[{'index':7,'text':text}])
        self.assertEqual(compact['a']['docx']['tables'],[{'index':0,'rows':[[text]]}])

if __name__=='__main__':unittest.main()
