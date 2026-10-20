import io,json,struct,unittest,zipfile
from jamovi_audit.archive import safe_zip, inventory
from jamovi_audit.readers import read_omv, read_docx

def archive(entries):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for k,v in entries.items():
            info=zipfile.ZipInfo('fixture')
            info.filename=k
            info.orig_filename=k
            z.writestr(info,v)
    return b.getvalue()
def vi(n):
    if n<0:n+=1<<64
    b=b''
    while n>127:b+=bytes([(n&127)|128]);n>>=7
    return b+bytes([n])
def p(n,v):
    if isinstance(v,int):return vi(n<<3)+vi(v)
    if isinstance(v,float):return vi(n<<3|1)+struct.pack('<d',v)
    if isinstance(v,str):v=v.encode()
    return vi(n<<3|2)+vi(len(v))+v

def omv(analysis=b'',extra=None):
    fields=[{'name':'gender','dataType':'Integer','measureType':'Nominal'}, {'name':'N','dataType':'Decimal','measureType':'Continuous'}]
    e={'meta':'jamovi-Archive-Version: 11.0\nCreated-By: jamovi 2.7.4\n','metadata.json':json.dumps({'dataSet':{'rowCount':2,'fields':fields}}),'xdata.json':json.dumps({'gender':{'labels':[[1,'Woman'],[2,'Man']]}}),'data.bin':struct.pack('<ii',1,-2147483648)+struct.pack('<dd',4.4,float('nan'))}
    if analysis:e['01 descriptives/analysis']=analysis
    e.update(extra or {})
    return archive(e)

class ArchiveTests(unittest.TestCase):
    def test_recursive_directories_do_not_merge_homonyms(self):
        es={'download/A/Same/a.omv':b'a','download/A/Same/a.docx':b'b','download/B/Same/a.omv':b'c'}
        result=inventory(es)
        self.assertEqual(len(result),2);self.assertNotEqual(result[0]['key'],result[1]['key'])
        self.assertEqual(sum(len(r['docx']) for r in result),1)
    def test_recognized_format_subfolders_pair(self):
        r=inventory({'wrapper/Studente/Analisi/a.omv':b'a','wrapper/Studente/Relazione/b.docx':b'b'})
        self.assertEqual(len(r),1);self.assertEqual(r[0]['label'],'Studente');self.assertEqual(len(r[0]['docx']),1)
    def test_separate_partial_student_folders_never_pair(self):
        r=inventory({'wrapper/Student1/a.omv':b'a','wrapper/Student2/a.docx':b'b'})
        self.assertEqual(len(r),2);self.assertTrue(all(not (x['omv'] and x['docx']) for x in r))
    def test_flat_root_students_are_separate(self):
        r=inventory({'Rossi_Anna_relazione.omv':b'a','Rossi_Anna_relazione.docx':b'b','Verdi_Luca_relazione.omv':b'c'})
        self.assertEqual(len(r),2)
    def test_zip_size_limit(self):
        from unittest.mock import patch
        with patch('jamovi_audit.archive.MAX_FILE_SIZE',2),self.assertRaises(ValueError):safe_zip(archive({'a':b'123'}))
    def test_versions_preserved_with_warning(self):
        r=inventory({'wrapper/A/a.omv':b'a','wrapper/A/a2.omv':b'a','wrapper/A/a.docx':b'b'})[0]
        self.assertEqual(len(r['omv']),2);self.assertTrue(r['warnings'])
    def test_temporary_files_ignored_nested_zip_reported(self):
        r=inventory({'A/a.omv':b'a','A/~$a.docx':b'b','A/extra.zip':b'bad','__MACOSX/._a.docx':b'x'})[0]
        self.assertFalse(r['docx']);self.assertTrue(r['warnings'])
    def test_zip_valid_unicode(self):
        self.assertEqual(safe_zip(archive({'À folder/a.docx':b'a'})),{'À folder/a.docx':b'a'})
    def test_unsafe_paths(self):
        for name in ['../evil','/evil','C:/evil','a\\evil','a/../evil','A./x','CON','a:stream','a//b']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_zip(archive({name:b'x'}))
    def test_windows_collision(self):
        with self.assertRaises(ValueError):safe_zip(archive({'a/X':b'a','A/x':b'b'}))
    def test_symlink(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=0o120777<<16;z.writestr(i,'target')
        with self.assertRaises(ValueError):safe_zip(b.getvalue())
    def test_file_directory_collision(self):
        with self.assertRaises(ValueError):safe_zip(archive({'a':b'x','a/b':b'y'}))

class OMVTests(unittest.TestCase):
    def test_binary_columns_levels(self):
        r=read_omv(omv());self.assertFalse(r['errors']);self.assertEqual(r['columns']['gender']['data'],[1,None]);self.assertEqual(r['columns']['N']['data'],[4.4,None]);self.assertEqual(r['columns']['gender']['levels'][0],[1,'Woman'])
    def test_actual_table_options_and_missing_image(self):
        option=p(3,'N');opts=p(1,p(5,p(1,option)))+p(2,1)+p(3,'vars')
        col=p(1,'mean')+p(2,'Mean')+p(7,p(2,4.4))
        table=p(1,'descriptives')+p(3,3)+p(6,p(1,col))
        image=p(1,'plot')+p(3,3)+p(7,p(1,'01 descriptives/resources/missing.png'))
        group=p(1,'root')+p(3,3)+p(8,p(1,table)+p(1,image))
        a=p(2,1)+p(3,'descriptives')+p(4,'jmv')+p(6,opts)+p(7,group)+p(8,3)
        r=read_omv(omv(a));self.assertFalse(r['errors']);a=r['analyses'][0]
        self.assertEqual(a['options'],{'vars':['N']});self.assertEqual(len(a['results']),2)
        self.assertEqual(a['results'][0]['columns'][0]['cells'],[4.4]);self.assertFalse(a['results'][1]['image_exists'])
    def test_stale_applicability_matches_jamovi_client_semantics(self):
        table=p(1,'table')+p(5,1)+p(6,b'')
        image=p(1,'plot')+p(5,1)+p(7,p(1,'plot.png'))
        text=p(1,'text')+p(5,1)+p(10,'cached text')
        group=p(1,'root')+p(5,1)+p(8,p(1,table)+p(1,image)+p(1,text))
        a=p(2,1)+p(3,'descriptives')+p(7,group)+p(8,3)
        nodes=read_omv(omv(a))['analyses'][0]['results']
        self.assertTrue(all(n['stale'] for n in nodes))
        self.assertEqual([n.get('stale_applicable') for n in nodes],[False,False,True])
    def test_parent_visibility_error_stale_inherited(self):
        leaf=p(1,'table')+p(6,p(1,p(1,'x')+p(7,p(1,1)) ))
        root=p(1,'root')+p(4,p(1,'failed'))+p(5,1)+p(8,p(1,leaf))+p(15,3)
        a=p(2,1)+p(3,'descriptives')+p(7,root)+p(8,4)
        el=read_omv(omv(a))['analyses'][0]['results'][0]
        self.assertFalse(el['visible']);self.assertTrue(el['stale']);self.assertTrue(el['error']);self.assertEqual(el['status'],4)
    def test_configured_analysis_has_no_invented_results(self):
        a=p(2,1)+p(3,'descriptives')+p(6,p(1,p(4,1))+p(2,1)+p(3,'plots'))+p(8,3)
        self.assertEqual(read_omv(omv(a))['analyses'][0]['results'],[])
    def test_corruption_and_unsupported_version_errors(self):
        self.assertTrue(read_omv(b'bad')['errors'])
        self.assertTrue(read_omv(omv(extra={'meta':'jamovi-Archive-Version: 99.0'}))['errors'])
    def test_unsupported_column_type_is_error(self):
        meta={'dataSet':{'rowCount':2,'fields':[{'name':'x','dataType':'FutureType'}]}}
        self.assertTrue(read_omv(omv(extra={'metadata.json':json.dumps(meta),'data.bin':struct.pack('<ii',1,2)}))['errors'])
    def test_actual_image_resource_is_present(self):
        leaf=p(1,'plot')+p(7,p(1,'01 descriptives/resources/a.png'))
        a=p(2,1)+p(3,'descriptives')+p(7,leaf)+p(8,3)
        r=read_omv(omv(a,{'01 descriptives/resources/a.png':b'PNG'}))
        self.assertTrue(r['analyses'][0]['results'][0]['image_exists'])
    def test_empty_table_is_preserved_as_empty(self):
        a=p(2,1)+p(3,'descriptives')+p(7,p(1,'empty')+p(6,b''))+p(8,3)
        self.assertEqual(read_omv(omv(a))['analyses'][0]['results'][0]['columns'],[])
    def test_text_id_string_table(self):
        meta={'dataSet':{'rowCount':2,'fields':[{'name':'ID','dataType':'Text','measureType':'ID'}]}}
        r=read_omv(omv(extra={'metadata.json':json.dumps(meta),'data.bin':struct.pack('<ii',0,4),'strings.bin':b'abc\x00def\x00'}))
        self.assertEqual(r['columns']['ID']['data'],['abc','def'])
    def test_deep_protobuf_options_rejected(self):
        opts=p(1,p(3,'x'))
        for _ in range(70):opts=p(1,p(5,opts))
        a=p(2,1)+p(3,'descriptives')+p(6,opts)
        self.assertTrue(read_omv(omv(a))['errors'])
    def test_truncated_binary_is_error(self):
        self.assertTrue(read_omv(omv(extra={'data.bin':b'abc'}))['errors'])
    def test_corrupt_protobuf_is_error(self):
        self.assertTrue(read_omv(omv(b'\x3a\xff\xff'))['errors'])

class DocxTests(unittest.TestCase):
    def test_paragraphs_table_and_formatting(self):
        xml='''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:spacing w:line="480" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:t>Parte 1</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Mean 4,4</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
        r=read_docx(archive({'word/document.xml':xml}));self.assertFalse(r['errors']);self.assertEqual(r['paragraphs'][0]['text'],'Parte 1');self.assertEqual(r['tables'][0]['rows'],[['Mean 4,4']]);self.assertIn('Mean 4,4',r['text']);self.assertEqual(r['paragraphs'][0]['format']['line'],480)
    def test_inherited_word_formatting(self):
        xml='<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Test</w:t></w:r></w:p></w:body></w:document>'
        styles='<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:pPr><w:spacing w:line="480"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style></w:styles>'
        r=read_docx(archive({'word/document.xml':xml,'word/styles.xml':styles}))
        self.assertTrue(r['formatting']['complete']);self.assertFalse(r['formatting']['issues'])
    def test_docx_entities_rejected(self):
        r=read_docx(archive({'word/document.xml':'<!DOCTYPE x [<!ENTITY x "secret">]><x>&x;</x>'}))
        self.assertTrue(r['errors'])
    def test_corrupt_word(self):self.assertTrue(read_docx(archive({'word/document.xml':'<bad'}))['errors'])

if __name__=='__main__':unittest.main()
