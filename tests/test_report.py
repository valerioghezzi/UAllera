import unittest

from jamovi_audit.report import compare_submissions, render_report


def report(key, label='Rossi'):
    return {'key': key, 'label': label, 'paths': {'omv': ['a.omv'], 'docx': ['a.docx']}, 'coverage': {}, 'findings': []}


class ReportTests(unittest.TestCase):
    def test_unique_sections_and_escaped_hostile_text(self):
        r = report('x', '<script>alert(1)</script>')
        r['findings'] = [{'part': 1, 'source': 'OMV', 'status': 'Errato', 'requirement': '<img src=x onerror=1>', 'observed': '<x>', 'expected': 'a', 'location': 'a'}]
        html = render_report([r, report('y', r['label'])], [])
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('id="student-1"', html)
        self.assertIn('id="student-2"', html)
        self.assertIn('href="#student-2"', html)
        self.assertIn('Annotazioni personali non esposte', html)

    def test_common_data_and_required_analyses_not_similarity_evidence(self):
        parsed = {k: {'omv': {'columns': {'A': {'data': [1, 2]}}, 'analyses': [{'name': 'descriptives', 'options': {'vars': ['A']}}]}, 'docx': {'paragraphs': []}} for k in ['x', 'y']}
        self.assertEqual([], compare_submissions([report('x'), report('y')], parsed))

    def test_identical_full_file_hash_is_duplicate_evidence(self):
        parsed = {k: {'omv': {'sha256': 'a' * 64}, 'docx': None} for k in ['x', 'y']}
        matches = compare_submissions([report('x'), report('y')], parsed)
        self.assertEqual(1, len(matches))
        self.assertEqual('omv_identico', matches[0]['kind'])
        self.assertIn('sha256', str(matches[0]).lower())

    def test_recompressed_omv_same_internal_content_is_distinct_evidence(self):
        parsed={'x':{'omv':{'sha256':'a'*64,'content_sha256':'c'*64}},'y':{'omv':{'sha256':'b'*64,'content_sha256':'c'*64}}}
        matches=compare_submissions([report('x'),report('y')],parsed)
        self.assertEqual(1,len(matches))
        self.assertEqual('omv_contenuto_identico',matches[0]['kind'])
        self.assertEqual('c'*64,matches[0]['evidence'][0]['content_sha256'])
        self.assertIn('direzione',matches[0]['limits'])

    def test_equal_raw_archive_does_not_duplicate_the_internal_content_finding(self):
        parsed={k:{'omv':{'sha256':'a'*64,'content_sha256':'c'*64}} for k in ('x','y')}
        matches=compare_submissions([report('x'),report('y')],parsed)
        self.assertEqual(['omv_identico'],[m['kind'] for m in matches])

    def test_missing_or_different_content_hash_is_not_equivalence_evidence(self):
        for other in (None,'d'*64):
            parsed={'x':{'omv':{'sha256':'a'*64,'content_sha256':'c'*64}},'y':{'omv':{'sha256':'b'*64,'content_sha256':other}}}
            self.assertEqual([],compare_submissions([report('x'),report('y')],parsed))

    def test_assignment_and_questions_are_excluded(self):
        prompt = 'Spiegare perché gli item sono ordinali e come si possono interpretare i punteggi delle scale.'
        parsed = {k: {'docx': {'paragraphs': [{'index': 2, 'text': prompt}]}, 'omv': None} for k in ['x', 'y']}
        self.assertEqual([], compare_submissions([report('x'), report('y')], parsed))

    def test_distinctive_matching_passage_has_positions_and_limits(self):
        passage = 'Nel mio caso il piccolo orologio viola sul tavolo mi ha suggerito una metafora insolita della precisione psicometrica.'
        parsed = {k: {'docx': {'paragraphs': [{'index': 9, 'text': passage}]}, 'omv': None} for k in ['x', 'y']}
        matches = compare_submissions([report('x'), report('y')], parsed)
        self.assertEqual(1, len(matches))
        self.assertIn('9', str(matches[0]['evidence']))
        self.assertTrue(matches[0]['limits'])

    def test_partial_literal_copy_survives_changed_intro_and_paragraph_breaks(self):
        shared='il piccolo orologio viola sul tavolo mi ha suggerito una metafora insolita della precisione psicometrica durante questa lunga esercitazione'
        first=' '.join(shared.split()[:10]);second=' '.join(shared.split()[10:])
        parsed={'x':{'docx':{'paragraphs':[{'index':1,'text':'Introduzione originale: '+shared+' con una chiusa diversa.'}]}},
                'y':{'docx':{'paragraphs':[{'index':7,'text':'Premessa modificata: '+first},{'index':8,'text':second+' e una conclusione nuova.'}]}}}
        matches=compare_submissions([report('x'),report('y')],parsed)
        self.assertTrue(any(m['kind']=='testo_parziale' for m in matches))
        evidence=next(m for m in matches if m['kind']=='testo_parziale')['evidence'][0]
        self.assertIn('7',evidence['location_right'])
        self.assertGreaterEqual(evidence['matching_tokens'],12)

    def test_small_word_edits_produce_near_copy_evidence(self):
        a='Durante il laboratorio ho immaginato una biblioteca silenziosa dove ciascun volume rappresenta una sfumatura del carattere; questa immagine mi aiuta a ricordare quanto sia delicato riassumere una persona con pochi indicatori.'
        b='Durante questo laboratorio avevo immaginato una biblioteca tranquilla dove ogni volume rappresenta una sfumatura del carattere; tale immagine mi aiuta a ricordare quanto sia complesso riassumere una persona con pochi indicatori.'
        parsed={k:{'docx':{'paragraphs':[{'index':i,'text':t}]}} for k,i,t in [('x',2,a),('y',6,b)]}
        matches=compare_submissions([report('x'),report('y')],parsed)
        near=next(m for m in matches if m['kind']=='testo_simile')
        self.assertGreaterEqual(near['evidence'][0]['token_sequence_ratio'],.72)
        self.assertIn('non',near['limits'].lower())

    def test_majority_shared_passage_is_shown_with_cohort_frequency(self):
        passage='Nel mio caso il piccolo orologio viola sul tavolo mi ha suggerito una metafora insolita della precisione psicometrica.'
        reports=[report(str(i)) for i in range(4)]
        parsed={str(i):{'docx':{'paragraphs':[{'index':3,'text':passage if i<3 else 'Un testo indipendente breve.'}]}} for i in range(4)}
        matches=compare_submissions(reports,parsed)
        self.assertEqual(3,len(matches))
        self.assertEqual(3,matches[0]['evidence'][0]['document_frequency'])
        self.assertIn('consegna',matches[0]['limits'].lower())

    def test_known_assignment_embedded_subspan_and_common_numeric_answer_are_excluded(self):
        assignment='La biblioteca silenziosa offre una metafora insolita per ricordare quanto sia delicato riassumere una persona con pochi indicatori.'
        parsed={'assignment_text':assignment}
        for k,prefix in [('x','Commento uno: '),('y','Commento due: ')]:
            parsed[k]={'docx':{'paragraphs':[{'index':1,'text':prefix+assignment},{'index':2,'text':'Il campione adulto comprende 906 persone; la media della scala N è 3,20 e la deviazione standard è 1,10.'}]}}
        self.assertEqual([],compare_submissions([report('x'),report('y')],parsed))

    def test_shared_wrong_number_is_weak_cooccurrence_not_proof(self):
        reports=[report('x'),report('y'),report('z')]
        finding={'id':'WORD.NUM.adult_n','part':2,'status':'Errato','observed':'Il campione adulto è composto da 907 persone.','location':'Word §3'}
        reports[0]['findings']=[finding];reports[1]['findings']=[finding]
        matches=compare_submissions(reports,{})
        self.assertEqual(1,len(matches))
        self.assertEqual('errore_condiviso',matches[0]['kind'])
        self.assertIn('debole',matches[0]['limits'].lower())
        self.assertIn('comune',matches[0]['limits'].lower())

    def test_shared_missing_or_unknown_is_not_shared_error_evidence(self):
        for status in ('Mancante','Non verificabile'):
            a=report('x');b=report('y')
            a['findings']=b['findings']=[{'id':'a','status':status,'observed':'missing'}]
            self.assertEqual([],compare_submissions([a,b],{}))

    def test_report_discloses_rule_pair_counts_and_candidate_limits(self):
        html=render_report([report('x'),report('y')],compare_submissions([report('x'),report('y')],{}))
        self.assertIn('Copertura dei confronti',html)
        self.assertIn('Coppie segnalate per regola',html)
        self.assertIn('candidati',html)

    def test_resource_cutoffs_are_counted_instead_of_silent_omission(self):
        parsed={'x':{'docx':{'paragraphs':[{'index':i,'text':'breve frase autonoma'} for i in range(2001)]}},'y':{'docx':{'paragraphs':[]}}}
        matches=compare_submissions([report('x'),report('y')],parsed)
        self.assertEqual(1,matches.diagnostics['cutoffs']['blocks_omitted'])
        parsed['x']['docx']['paragraphs']=[{'index':1,'text':' '.join(['banana']*40005)}]
        matches=compare_submissions([report('x'),report('y')],parsed)
        self.assertEqual(5,matches.diagnostics['cutoffs']['tokens_omitted'])

    def test_quoted_question_does_not_hide_copied_answer_in_same_paragraph(self):
        answer='Nel mio caso il piccolo orologio viola sul tavolo mi ha suggerito una metafora insolita della precisione psicometrica.'
        parsed={'x':{'docx':{'paragraphs':[{'index':1,'text':'Come interpreti il risultato? '+answer}]}},'y':{'docx':{'paragraphs':[{'index':2,'text':answer}]}}}
        self.assertTrue(compare_submissions([report('x'),report('y')],parsed))


if __name__ == '__main__':
    unittest.main()
