"""Evidence provenance tests use source documents, never model/service mocks."""
import copy
import importlib
import importlib.util
import json
import unittest


class EvidenceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('jamovi_audit.evidence'),
                             'Structured evidence API is missing')
        return importlib.import_module('jamovi_audit.evidence')

    def fixture(self):
        quote='La correlazione non dimostra causalità.'
        sources=[{'id':'node/1','text':quote,'location':'Paragrafo 1'},
                 {'id':'node/2','text':quote,'location':'Paragrafo 2'},
                 {'id':'node/3','text':'UNCITED_FULL_DOCUMENT_SECRET','location':'Paragrafo 3'}]
        documents={'omv':{'sha256':'a'*64}, 'docx':{'sha256':'d'*64,'evidence_sources':sources,
                    'text':'UNCITED_FULL_DOCUMENT_SECRET','evidence_coverage':{'scope':'body','omissions':[]}}}
        deterministic={'id':'CHECK.6','part':6,'source':'OMV–Word','status':'Errato',
                       'requirement':'Associazioni','observed':'Contraddizione da esaminare.',
                       'expected':'Interpretazione prudente.','location':'OMV e Word'}
        model={'id':'LOCAL.6.1','part':6,'source':'Word · motore locale','status':'Da verificare',
               'requirement':'Causalità','observed':'Proposta originale del modello.',
               'expected':'Controllo del tutor.','location':'Paragrafo 2',
               'proposal':'coerente','quote':quote,'source_id':'node/2','evidence_id':'E2',
               'char_start':0,'char_end':len(quote),'chunk':1}
        report={'findings':[deterministic],'linguistic':{'findings':[model],
                'chunks':[{'number':1,'status':'complete','evidence_ids':['E2']}],
                'manifest':{'source_document_sha256':'d'*64,'spans':[{
                    'source_id':'node/2','evidence_id':'E2','start':0,'end':len(quote),'location':'Paragrafo 2'}]},
                'protocol':{'version':'minerva-evidence-1.0','asset_verification':'external-unverified',
                            'config':{'api_key':'MODEL_SECRET'},'reported_runtime':{'version':'b11756','api_key':'MODEL_SECRET'}},
                'config':{'api_key':'MODEL_SECRET'}}}
        return report,documents

    def test_stable_records_bind_both_artifacts_and_do_not_mutate_input(self):
        api=self.api(); report,documents=self.fixture(); before=copy.deepcopy((report,documents))
        bundle=api.build_evidence_bundle(report,documents)
        self.assertEqual(bundle,api.build_evidence_bundle(report,documents))
        self.assertEqual((report,documents),before)
        self.assertEqual(bundle['version'],'1.0')
        deterministic=next(r for r in bundle['records'] if r['origin']=='deterministic')
        self.assertEqual({a['kind'] for a in deterministic['artifacts']},{'omv','docx'})
        self.assertEqual({a['sha256'] for a in deterministic['artifacts']},{'a'*64,'d'*64})
        self.assertEqual(deterministic['finding_text']['observed'],'Contraddizione da esaminare.')
        changed=copy.deepcopy(documents);changed['omv']['sha256']='b'*64
        other=api.build_evidence_bundle(report,changed)
        self.assertNotEqual(deterministic['id'],next(r for r in other['records'] if r['origin']=='deterministic')['id'])

    def test_identical_paragraphs_bind_the_selected_original_node(self):
        api=self.api(); report,documents=self.fixture()
        model=next(r for r in api.build_evidence_bundle(report,documents)['records'] if r['origin']=='model')
        self.assertEqual(len(model['anchors']),1)
        anchor=model['anchors'][0]
        self.assertEqual(anchor['source_id'],'node/2')
        self.assertEqual(anchor['location'],'Paragrafo 2')
        self.assertEqual(anchor['quote'],documents['docx']['evidence_sources'][1]['text'][anchor['start']:anchor['end']])
        confidence=model['confidence']
        self.assertEqual(confidence['support_score'],confidence['supported_checks']/confidence['eligible_checks'])
        self.assertIsNone(confidence['semantic_confidence'])
        self.assertEqual(confidence['calibration'],'not_available')

    def test_altered_quote_offsets_id_or_document_hash_cannot_produce_anchor(self):
        api=self.api()
        for alteration in ('quote','offset','id','hash','manifest_offset','chunk','incomplete_chunk','foreign_chunk_id'):
            with self.subTest(alteration=alteration):
                report,documents=self.fixture(); model=report['linguistic']['findings'][0]
                if alteration=='quote':model['quote']='Una correlazione dimostra causalità.'
                if alteration=='offset':model['char_start']=1
                if alteration=='id':model['evidence_id']='FORGED'
                if alteration=='hash':documents['docx']['sha256']='e'*64
                if alteration=='manifest_offset':report['linguistic']['manifest']['spans'][0]['end']-=1
                if alteration=='chunk':model['chunk']=2
                if alteration=='incomplete_chunk':report['linguistic']['chunks'][0]['status']='failed'
                if alteration=='foreign_chunk_id':report['linguistic']['chunks'][0]['evidence_ids']=['OTHER']
                record=next(r for r in api.build_evidence_bundle(report,documents)['records'] if r['origin']=='model')
                self.assertEqual(record['anchors'],[])
                self.assertTrue(record['confidence']['missing'])
                self.assertEqual(record['evaluation'],'unverifiable')

    def test_model_cannot_downgrade_missing_evidence_id_to_legacy_anchor(self):
        api=self.api()
        for mode in ('null_id','removed_id','removed_contract_fields'):
            with self.subTest(mode=mode):
                report,documents=self.fixture();row=report['linguistic']['findings'][0]
                report['linguistic']['manifest']['source_document_sha256']='WRONG'
                report['linguistic']['chunks'][0]['status']='failed'
                if mode=='null_id':row['evidence_id']=None
                elif mode=='removed_id':row.pop('evidence_id')
                else:
                    for field in ('evidence_id','source_id','char_start','char_end','chunk'):row.pop(field)
                record=api.build_evidence_bundle(report,documents)['records'][1]
                self.assertEqual(record['anchors'],[])
                self.assertEqual(record['evaluation'],'unverifiable')
                self.assertLess(record['confidence']['support_score'],1)

    def test_conflicts_reference_one_group_without_pairwise_duplication(self):
        api=self.api();report,documents=self.fixture()
        original=report['linguistic']['findings'][0]
        report['linguistic']['findings']=[dict(original,id=f'ROW{i}',proposal='errata' if i==19 else 'coerente') for i in range(20)]
        bundle=api.build_evidence_bundle(report,documents)
        self.assertEqual(len(bundle['contradictions']),1)
        conflict=bundle['contradictions'][0]
        rows=[r for r in bundle['records'] if r['origin']=='model']
        self.assertEqual(set(conflict['record_ids']),{r['id'] for r in rows})
        self.assertTrue(all(r['contradictions']==[conflict['id']] for r in rows))
        self.assertEqual(sum(len(r['contradictions']) for r in rows),20)

    def test_legacy_location_fallback_requires_unique_exact_occurrence(self):
        api=self.api()
        documents={'docx':{'sha256':'d'*64,'paragraphs':[{'text':'inizio È🧪 fine','location':'Paragrafo 1'}]}}
        model={'id':'legacy','part':1,'status':'Da verificare','proposal':'coerente',
               'quote':'È🧪','location':'Paragrafo 1'}
        report={'findings':[],'linguistic':{'findings':[model]}}
        anchor=api.build_evidence_bundle(report,documents)['records'][0]['anchors'][0]
        self.assertEqual((anchor['start'],anchor['end']),(7,9))
        documents['docx']['paragraphs'][0]['text']='È🧪 e È🧪'
        record=api.build_evidence_bundle(report,documents)['records'][0]
        self.assertEqual(record['anchors'],[])

    def test_legacy_overlapping_quote_occurrences_are_ambiguous(self):
        api=self.api()
        report={'linguistic':{'findings':[{'id':'legacy','part':1,'status':'Da verificare',
                 'proposal':'coerente','quote':'aa','location':'Paragrafo 1'}]}}
        documents={'docx':{'sha256':'d'*64,'paragraphs':[{'text':'aaa','location':'Paragrafo 1'}]}}
        self.assertEqual(api.build_evidence_bundle(report,documents)['records'][0]['anchors'],[])

    def test_conflicting_proposals_are_linked_without_resolving_their_truth(self):
        api=self.api(); report,documents=self.fixture()
        other=copy.deepcopy(report['linguistic']['findings'][0]);other.update(id='LOCAL.6.2',proposal='errata')
        report['linguistic']['findings'].append(other)
        bundle=api.build_evidence_bundle(report,documents)
        self.assertEqual(len(bundle['contradictions']),1)
        conflict=bundle['contradictions'][0]
        models=[r for r in bundle['records'] if r['origin']=='model']
        self.assertEqual(set(conflict['record_ids']),{r['id'] for r in models})
        self.assertEqual(conflict['resolution'],'unresolved')
        self.assertTrue(all(r['contradictions'] for r in models))
        self.assertTrue(all(r['confidence']['semantic_confidence'] is None for r in models))

    def test_coverage_separates_unverifiable_and_not_applicable_records(self):
        api=self.api();report,documents=self.fixture()
        report['findings'].append({'id':'N/A','part':0,'source':'OMV','status':'Non applicabile'})
        report['linguistic']['findings'].append({'id':'ABSTAIN','part':1,'status':'Non verificabile','proposal':'non_valutabile'})
        coverage=api.build_evidence_bundle(report,documents)['coverage']
        self.assertEqual(coverage['total_records'],4)
        self.assertEqual(coverage['evaluated'],2)
        self.assertEqual(coverage['unverifiable'],1)
        self.assertEqual(coverage['not_applicable'],1)

    def test_forensic_export_omits_documents_configs_and_unallowlisted_secrets(self):
        api=self.api();report,documents=self.fixture()
        bundle=api.build_evidence_bundle(report,documents)
        bundle['config']={'api_key':'INJECTED_SECRET'}
        bundle['records'][0]['config']={'api_key':'INJECTED_SECRET'}
        exported=api.export_evidence_json(bundle)
        value=json.loads(exported)
        self.assertEqual(value['version'],'1.0')
        self.assertNotIn('UNCITED_FULL_DOCUMENT_SECRET',exported)
        self.assertNotIn('MODEL_SECRET',exported)
        self.assertNotIn('INJECTED_SECRET',exported)
        self.assertIn('La correlazione non dimostra causalità.',exported)
        self.assertIn('b11756',exported)

    def test_tokenizer_identity_survives_structured_forensic_export(self):
        api=self.api();report,documents=self.fixture()
        protocol=report['linguistic']['protocol']
        protocol.update(tokenizer_sha256='f'*64,input_encoding='official-hf-tokenizer-json-stdlib')
        exported=json.loads(api.export_evidence_json(api.build_evidence_bundle(report,documents)))
        self.assertEqual(exported['protocol']['tokenizer_sha256'],'f'*64)
        self.assertEqual(exported['protocol']['input_encoding'],'official-hf-tokenizer-json-stdlib')

    def test_shared_provenance_facts_are_read_only_and_export_is_lossless(self):
        api=self.api();report,documents=self.fixture()
        report['findings'].append(dict(report['findings'][0],id='CHECK.OTHER'))
        bundle=api.build_evidence_bundle(report,documents)
        first=bundle['records'][0];second=bundle['records'][1]
        with self.assertRaises(TypeError):first['confidence']['semantic_confidence']=1.0
        with self.assertRaises(TypeError):first['artifacts'][0]['sha256']='forged'
        with self.assertRaises(TypeError):first['confidence']['checks'][0]['passed']=False
        self.assertIsNone(second['confidence']['semantic_confidence'])
        self.assertEqual(json.loads(api.export_evidence_json(bundle))['records'][0]['confidence']['missing'],list(first['confidence']['missing']))


if __name__=='__main__':unittest.main()
