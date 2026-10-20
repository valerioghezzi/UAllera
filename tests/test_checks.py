import copy
import math
import unittest

from jamovi_audit.checks import check_submission, _status_problem


def omv(analyses=None):
    return {'metadata': {}, 'columns': {}, 'analyses': analyses or [], 'errors': []}


def desc(nodes, **options):
    return {'id': 1, 'name': 'descriptives', 'ns': 'jmv', 'options': {'vars': ['O'], **options}, 'results': nodes}


def image(name='boxplot', **kwargs):
    return {'kind': 'image', 'name': name, 'path': ['O', name], 'title': name,
            'visible': True, 'image_exists': True, 'image_path': 'plot.png', **kwargs}


class ChecksTests(unittest.TestCase):
    def check(self, data, document=None, reference=None):
        return check_submission({'key': 'one', 'label': 'Rossi', 'omv': ['r.omv'], 'docx': ['r.docx']}, data, document, reference)

    def finding(self, report, identifier):
        return next(f for f in report['findings'] if f['id'] == identifier)

    def test_option_without_saved_output_is_missing(self):
        report = self.check(omv([desc([], box=True)]))
        self.assertEqual('Mancante', self.finding(report, 'P3.O.boxplot')['status'])

    def test_valid_image_is_verified_but_hidden_error_stale_and_missing_are_not(self):
        for change in ({}, {'visible': False}, {'error': 'failed'}, {'stale': True}, {'image_exists': False}):
            with self.subTest(change=change):
                report = self.check(omv([desc([image(**change)])]))
                status = self.finding(report, 'P3.O.boxplot')['status']
                self.assertEqual(status == 'Verificato', not change)

    def test_wrong_variable_output_does_not_satisfy_O(self):
        report = self.check(omv([desc([image()], vars=['N'])]))
        self.assertNotEqual('Verificato', self.finding(report, 'P3.O.boxplot')['status'])

    def test_corrupt_input_is_not_missing_requirements(self):
        report = self.check({'errors': ['unsupported'], 'analyses': [], 'columns': {}})
        self.assertEqual('Non verificabile', self.finding(report, 'P3.O.boxplot')['status'])

    def test_incomplete_pair_keeps_identity_and_seven_parts(self):
        report = check_submission({'key': 'x', 'label': 'No OMV', 'omv': [], 'docx': ['x.docx']}, None, {'paragraphs': [], 'text': ''})
        self.assertEqual('x', report['key'])
        self.assertEqual(set(range(1, 8)), {f['part'] for f in report['findings'] if f['part']})
        self.assertEqual('Mancante', self.finding(report, 'FILE.OMV')['status'])

    def test_numeric_word_incompatibility_and_rounding(self):
        ref = {'numeric_claims': [{'id': 'adult_n', 'part': 2, 'pattern': r'campione\s*(?:di|=)\s*(?P<value>\d+)', 'value': 906, 'location': 'dati/age', 'description': 'N adulti'}]}
        for value, expected in [('907', 'Errato'), ('906', 'Verificato')]:
            doc = {'paragraphs': [{'index': 4, 'text': 'Il campione di ' + value}], 'text': 'Il campione di ' + value}
            report = self.check(omv(), doc, ref)
            f = self.finding(report, 'WORD.NUM.adult_n')
            self.assertEqual(expected, f['status'])
            self.assertIn('4', f['location'])

    def test_reverse_and_score_values_are_independently_checked(self):
        data = omv()
        data['columns'] = {n: {'meta': {}, 'data': vals, 'levels': []} for n, vals in {
            'age': [17, 20, 42], 'A1': [1, 2, 3], 'A2': [2, 2, 2], 'A3': [3, 3, 3], 'A4': [4, 4, 4], 'A5': [5, 5, 5],
            'A1_Rev': [6, 5, 4], 'A': [4, 3.8, 3.6], 'A_controllo': [4, 3.8, 9]}.items()}
        report = self.check(data)
        self.assertEqual('Verificato', self.finding(report, 'P2.A1_Rev.values')['status'])
        self.assertEqual('Verificato', self.finding(report, 'P2.A.values')['status'])
        self.assertEqual('Errato', self.finding(report, 'P2.A_controllo.values')['status'])

    def test_questions_alone_are_not_verified_answers(self):
        doc = {'text': 'Parte 1\nPerché gli item sono ordinali?', 'paragraphs': [{'index': 0, 'text': 'Perché gli item sono ordinali?'}]}
        report = self.check(omv(), doc)
        self.assertFalse(any(f['status'] == 'Verificato' for f in report['findings'] if f['source'] == 'Word' and f['id'].startswith('WORD.Q')))

    def test_official_complete_enum_three_is_valid_error_four_is_not(self):
        for status, expected in [(3, 'Verificato'), (4, 'Errato'), (2, 'Non verificabile'), (0, 'Non verificabile')]:
            data=omv([desc([image(status=status)])])
            self.assertEqual(expected,self.finding(self.check(data),'P3.O.boxplot')['status'])

    def test_table_cannot_satisfy_a_required_graph(self):
        node={'kind':'table','name':'desc','path':['desc'],'visible':True,'columns':[{'name':'dep','cells':['N']},{'name':'mean[1]','cells':[3.]},{'name':'mean[2]','cells':[4.]}]}
        data=omv([{'id':1,'name':'ttestIS','options':{'vars':['N'],'group':'gender'},'results':[node]}])
        self.assertNotEqual('Verificato',self.finding(self.check(data),'P5.N.plot')['status'])

    def test_frequency_variable_can_be_identified_in_array_path(self):
        node={'kind':'table','name':'gender','path':['frequencies','gender'],'columns':[{'name':'counts','cells':[400,506]}]}
        data=omv([desc([node],vars=['gender'])])
        self.assertEqual('Verificato',self.finding(self.check(data),'P3.gender.freq')['status'])

    def test_reliability_item_table_requires_every_item_row(self):
        node={'kind':'table','name':'items','path':['items'],'columns':[{'name':'name','cells':['N1']},{'name':'itemRestCor','cells':[.4]}]}
        data=omv([{'id':1,'name':'reliability','options':{'vars':['N1','N2','N3','N4','N5']},'results':[node]}])
        self.assertNotEqual('Verificato',self.finding(self.check(data),'P7.N.items.itemRestCor')['status'])

    def test_unknown_module_is_not_treated_as_supported_missing(self):
        data=omv([{'name':'descriptives','ns':'custom-unrecognized','options':{'vars':['O']},'results':[]}])
        self.assertEqual('Non verificabile',self.finding(self.check(data),'P3.O.boxplot')['status'])

    def test_saved_score_provenance_requires_linked_reliability_not_column_name(self):
        data=omv([{'id':8,'name':'reliability','ns':'jmv','status':3,'options':{'vars':['N1','N2','N3','N4','N5']},'results':[]}])
        data['columns']['N']={'meta':{'outputAnalysisId':8,'outputOptionName':'meanScoreOV'},'data':[3.]}
        self.assertEqual('Verificato',self.finding(self.check(data),'P2.N.provenance')['status'])
        data['columns']['N']['meta']['outputAnalysisId']=99
        self.assertEqual('Non verificabile',self.finding(self.check(data),'P2.N.provenance')['status'])

    def test_explicit_word_format_errors_and_resolved_good_format(self):
        for fmt,status in [({'issues':[],'unknown':[],'complete':True,'runs_checked':4},'Verificato'),({'issues':['font=Arial'],'unknown':[],'complete':True},'Errato'),({'issues':[],'unknown':['font unresolved'],'complete':False},'Non verificabile')]:
            self.assertEqual(status,self.finding(self.check(omv(),{'formatting':fmt,'paragraphs':[]}),'WORD.format')['status'])

    def test_unambiguous_word_sample_claim_compares_own_data_without_reference(self):
        data=omv();data['columns']={'age':{'data':[17,18,40], 'meta':{}}}
        doc={'paragraphs':[{'index':2,'text':'Il campione adulto è composto da 3 persone.'}]}
        f=self.finding(self.check(data,doc),'WORD.NUM.adult_n')
        self.assertEqual('Errato',f['status'])
        self.assertIn('2',f['expected'])

    def test_numeric_claim_inside_word_table_is_checked_with_cell_location(self):
        ref={'numeric_claims':[{'id':'test','part':4,'pattern':r'Tscore_N\s*=\s*(?P<value>\d+,\d+)', 'value':62.368421}]}
        doc={'paragraphs':[],'tables':[{'index':0,'rows':[['Tscore_N = 62,37']]}]}
        f=self.finding(self.check(omv(),doc,ref),'WORD.NUM.test')
        self.assertEqual('Verificato',f['status'])
        self.assertIn('cella',f['location'])

    def test_completed_image_raw_stale_not_applicable_verifies_presence_only(self):
        data=omv([desc([image(status=3,stale=True,stale_applicable=False)])])
        result=self.check(data)
        f=self.finding(result,'P3.O.boxplot')
        self.assertEqual('Verificato',f['status'])
        self.assertEqual('Non verificabile',self.finding(result,'P3.O.boxplot.freshness')['status'])
        self.assertIn('aggiornamento',self.finding(result,'P3.O.boxplot.freshness')['requirement'].lower())

    def test_completed_table_raw_stale_not_applicable_verifies_presence_only(self):
        node={'kind':'table','name':'descriptives','path':['descriptives'],'status':3,'stale':True,'stale_applicable':False,
              'columns':[{'name':'name','cells':['O']},{'name':'mean','cells':[4.]}]}
        result=self.check(omv([desc([node])]))
        self.assertEqual('Verificato',self.finding(result,'P3.O.mean')['status'])
        self.assertEqual('Non verificabile',self.finding(result,'P3.O.mean.freshness')['status'])

    def test_applicable_stale_still_blocks_textual_result(self):
        for kind in ('html','preformatted','notice','syntax'):
            with self.subTest(kind=kind):
                self.assertEqual('Non verificabile',_status_problem({'kind':kind,'status':3,'stale':True,'stale_applicable':True})[0])

    def test_sample_z_does_not_reject_correct_filtered_out_missing_rows(self):
        data=omv();data['columns']={n:{'meta':{},'data':v} for n,v in {'age':[17,18,40],'N':[6,3,5],'Zscore_N':[None,-1/math.sqrt(2),1/math.sqrt(2)]}.items()}
        f=self.finding(self.check(data),'P4.Zscore_N.values')
        self.assertEqual('Verificato',f['status'])
        self.assertIn('Confrontate 2 righe',f['observed'])
        data['columns']['Zscore_N']['data'][1]=1
        self.assertEqual('Errato',self.finding(self.check(data),'P4.Zscore_N.values')['status'])

    def test_original_lowercase_id_is_supported_with_actual_source_path(self):
        data=omv();data['columns']['id']={'meta':{'measureType':'ID'},'data':[61789]}
        f=self.finding(self.check(data),'P1.ID.column')
        self.assertEqual('Verificato',f['status'])
        self.assertEqual('dati/id',f['location'])

    def test_original_decimal_id_resolves_own_T_claim(self):
        data=omv();data['columns']={'id':{'meta':{'measureType':'ID'},'data':[61789.0]},'Tscore_N':{'meta':{},'data':[62.36842105263158]}}
        doc={'paragraphs':[{'index':4,'text':'Per ID 61789 il punteggio T = 62,37.'}]}
        self.assertEqual('Verificato',self.finding(self.check(data,doc),'WORD.NUM.T_ID_61789')['status'])

    def test_actual_jmv_descriptives_dynamic_columns_are_recognized(self):
        node={'kind':'table','name':'descriptives','path':['descriptives'],'columns':[
            {'name':'stat[min]','cells':['Minimum']},{'name':'O[min]','title':'O','cells':[1.4]},
            {'name':'stat[sww]','cells':['Shapiro-Wilk W']},{'name':'O[sww]','title':'O','cells':[.95]},
            {'name':'stat[sw]','cells':['Shapiro-Wilk p']},{'name':'O[sw]','title':'O','cells':[.001]}]}
        r=self.check(omv([desc([node])]))
        for ident in ('P3.O.min','P3.O.sw','P3.O.swp'):
            self.assertEqual('Verificato',self.finding(r,ident)['status'])
        node['columns'][-1]['cells']=[]
        self.assertEqual('Mancante',self.finding(self.check(omv([desc([node])])),'P3.O.swp')['status'])

    def test_dynamic_descriptives_split_requires_both_groups_statistic(self):
        node={'kind':'table','name':'descriptives','path':['descriptives'],'columns':[
            {'name':'stat[meanmale]','cells':['Mean']},{'name':'var1[meanmale]','cells':['male']},{'name':'O[meanmale]','title':'O','cells':[4.]},
            {'name':'stat[meanfemale]','cells':['Mean']},{'name':'var1[meanfemale]','cells':['female']},{'name':'O[meanfemale]','title':'O','cells':[5.]}]}
        r=self.check(omv([desc([node],splitBy=['gender'])]))
        self.assertEqual('Verificato',self.finding(r,'P3.O.gender.mean')['status'])
        node['columns'][-1]['visible']=False
        self.assertEqual('Mancante',self.finding(self.check(omv([desc([node],splitBy=['gender'])])),'P3.O.gender.mean')['status'])

    def test_ttest_numeric_cells_must_belong_to_requested_variable_row(self):
        node={'kind':'table','name':'ttest','path':['ttest'],'columns':[{'name':'name','cells':['A','N']},
              {'name':'stat[stud]','cells':[2,None]},{'name':'df[stud]','cells':[30,None]},{'name':'p[stud]','cells':[.04,None]}]}
        data=omv([{'name':'ttestIS','options':{'vars':['N','A'],'group':'gender'},'results':[node]}])
        result=self.check(data)
        self.assertEqual('Verificato',self.finding(result,'P5.A.student')['status'])
        self.assertEqual('Mancante',self.finding(result,'P5.N.student')['status'])

    def test_reliability_item_statistics_require_numeric_cell_for_each_item(self):
        node={'kind':'table','name':'items','path':['items'],'columns':[{'name':'name','cells':['N1','N2','N3','N4','N5']},{'name':'itemRestCor','cells':[.4,.5,None,.3,.2]}]}
        data=omv([{'name':'reliability','options':{'vars':['N1','N2','N3','N4','N5']},'results':[node]}])
        self.assertEqual('Mancante',self.finding(self.check(data),'P7.N.items.itemRestCor')['status'])
        node['columns'][1]['cells'][2]=.3
        self.assertEqual('Verificato',self.finding(self.check(data),'P7.N.items.itemRestCor')['status'])

    def test_matrix_labels_do_not_replace_actual_pairwise_numeric_cells(self):
        names=['A','C','E','N','O','age']
        cols=[{'name':'.name['+stat+']','cells':names} for stat in ('r','rp','rciu','rcil','n')]
        node={'kind':'table','name':'matrix','path':['matrix'],'columns':cols}
        data=omv([{'name':'corrMatrix','options':{'vars':names,'pearson':True},'results':[node]}])
        self.assertEqual('Mancante',self.finding(self.check(data),'P6.pearson')['status'])
        for channel in ('r','rp','rciu','rcil','n'):
            for i,name in enumerate(names):
                cols.append({'name':name+'['+channel+']','cells':[None if j<=i else 906 if channel=='n' else .2 for j in range(len(names))]})
        self.assertEqual('Verificato',self.finding(self.check(data),'P6.pearson')['status'])
        next(c for c in cols if c['name']=='A[r]')['cells'][1]=None
        self.assertEqual('Mancante',self.finding(self.check(data),'P6.pearson')['status'])

    def test_welch_condition_ignores_other_group_namespace_or_variables(self):
        for change in ({'group':'education'},{'ns':'other'},{'vars':['A']}):
            node={'kind':'table','name':'eqv','path':['eqv'],'columns':[{'name':'name','cells':['N']},{'name':'f','cells':[1.]},{'name':'p','cells':[.8]}]}
            a={'name':'ttestIS','ns':change.get('ns','jmv'),'options':{'vars':change.get('vars',['N']),'group':change.get('group','gender')},'results':[node]}
            self.assertEqual('Non verificabile',self.finding(self.check(omv([a])),'P5.N.welch')['status'])

    def test_welch_condition_rejects_impossible_p_value(self):
        for p in (-.3,1.2):
            node={'kind':'table','name':'eqv','path':['eqv'],'columns':[{'name':'name','cells':['N']},{'name':'f','cells':[1.]},{'name':'p','cells':[p]}]}
            a={'name':'ttestIS','options':{'vars':['N'],'group':'gender'},'results':[node]}
            self.assertEqual('Non verificabile',self.finding(self.check(omv([a])),'P5.N.welch')['status'])

    def test_T_claim_must_parse_whole_number_including_exponent_and_percent(self):
        data=omv();data['columns']={'id':{'data':[61789.]},'Tscore_N':{'data':[62.368421]}}
        for token in ('62,37e4','62,37%','62,370,000','62,37e'):
            doc={'paragraphs':[{'index':1,'text':'ID 61789: T = '+token+'.'}]}
            self.assertNotEqual('Verificato',self.finding(self.check(data,doc),'WORD.NUM.T_ID_61789')['status'],token)
        doc={'paragraphs':[{'index':1,'text':'ID 61789: T = 6,237e1.'}]}
        self.assertEqual('Verificato',self.finding(self.check(data,doc),'WORD.NUM.T_ID_61789')['status'])

    def test_output_scores_allow_filtered_minor_missing_but_check_adults(self):
        data=omv();data['columns']={n:{'meta':{},'data':v} for n,v in {'age':[17,20],'N1':[1,2],'N2':[2,3],'N3':[3,4],'N4':[4,5],'N5':[5,6],'N':[None,4]}.items()}
        data['columns']['N']['meta']={'columnType':'Output'}
        data['columns']['adult_filter']={'meta':{'columnType':'Filter','formula':'age >= 18','active':True},'data':[0,1]}
        self.assertEqual('Verificato',self.finding(self.check(data),'P2.N.values')['status'])
        data['columns']['N']['data'][1]=3
        self.assertEqual('Errato',self.finding(self.check(data),'P2.N.values')['status'])

    def test_pooled_reliability_does_not_satisfy_separate_scale_requirements(self):
        names=['N'+str(i) for i in range(1,6)]+['O'+str(i) for i in range(1,6)]
        nodes=[{'kind':'table','name':'scale','path':['scale'],'columns':[{'name':'alpha','cells':[.9]}]},
               {'kind':'table','name':'items','path':['items'],'columns':[{'name':'name','cells':names},{'name':'itemRestCor','cells':[.4]*10}]}]
        data=omv([{'name':'reliability','options':{'vars':names},'results':nodes}])
        r=self.check(data)
        for s in ('N','O'):
            self.assertEqual('Mancante',self.finding(r,f'P7.{s}.alpha')['status'])
            self.assertEqual('Mancante',self.finding(r,f'P7.{s}.items.itemRestCor')['status'])

    def test_reference_values_align_by_unique_ID_after_student_sort(self):
        ref={'row_ids':[10,20],'column_values':{'Tscore_N':[40,60]}}
        data=omv();data['columns']={'id':{'meta':{},'data':[20.,10.]},'Tscore_N':{'meta':{},'data':[60,40]}}
        matching=[f for f in self.check(data,reference=ref)['findings'] if f['id']=='P4.Tscore_N.values']
        self.assertEqual('Verificato',matching[-1]['status'])
        data['columns']['Tscore_N']['data'][0]=61
        matching=[f for f in self.check(data,reference=ref)['findings'] if f['id']=='P4.Tscore_N.values']
        self.assertEqual('Errato',matching[-1]['status'])

    def test_reference_unknown_or_duplicate_ID_does_not_compare_positionally(self):
        ref={'row_ids':[10,20],'column_values':{'Tscore_N':[40,60]}}
        for ids in ([10,10],[10,30],[10,None],[]):
            data=omv();data['columns']={'id':{'meta':{},'data':ids},'Tscore_N':{'meta':{},'data':[40,60]}}
            matching=[f for f in self.check(data,reference=ref)['findings'] if f['id']=='P4.Tscore_N.values']
            self.assertEqual('Non verificabile',matching[-1]['status'],ids)

    def test_present_but_corrupt_or_unsupported_image_cannot_verify_graph(self):
        for valid,expected in ((False,'Errato'),(None,'Non verificabile'),(True,'Verificato')):
            node=image(image_valid=valid,image_issue='Risorsa di prova non decodificabile')
            self.assertEqual(expected,self.finding(self.check(omv([desc([node])])),'P3.O.boxplot')['status'])


if __name__ == '__main__':
    unittest.main()
