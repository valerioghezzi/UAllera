"""Offline compatibility observations and explicitly scoped forensic export."""
from datetime import datetime, timezone
import json
import math
import platform
from .evidence import export_evidence_json

CONTRACT_VERSION='omv11-docx-body-1'
_METADATA_FIELDS=('creator','lastModifiedBy','author','lastSavedBy','created','dateCreated','creationDate',
                  'modified','dateModified','modificationDate')
_CACHE_COUNTERS=('hits','misses','bypassed','evictions','entries','bytes','encoded_bytes','max_entries','max_bytes')
_COMPARISON_KINDS=('testo_coincidente','testo_parziale','testo_simile','omv_identico','omv_contenuto_identico',
                   'errore_condiviso','metadata_autore_condiviso','metadata_correlazione_temporale')


def _mapping(value):
    return value if isinstance(value,dict) else {}


def _fields(value,names):
    """Only explicit scalar leaves cross an export/snapshot boundary."""
    return {key:item for key,item in _mapping(value).items() if key in names and
            (item is None or type(item) in (str,int,bool) or (type(item) is float and math.isfinite(item)))}


def _strings(value):
    return [item for item in value if isinstance(item,str)] if isinstance(value,(list,tuple)) else []


def _counters(value,names):
    return {key:item for key,item in _mapping(value).items() if key in names and type(item) is int and item>=0}


def _cache_stats(value):
    result=_counters(value,_CACHE_COUNTERS)
    if isinstance(_mapping(value).get('bytes_accounting'),str):
        result['bytes_accounting']=value['bytes_accounting']
    return result


def _compatibility(value):
    value=_mapping(value)
    result=_fields(value,('reader_contract','network_check_performed'))
    if 'omv' in value:
        omv=_mapping(value['omv'])
        result['omv']=_fields(omv,('observed_archive','supported_archive','archive_contract_supported',
                                   'declared_application','application_version_certified'))
        result['omv']['reader_errors']=_strings(omv.get('reader_errors'))
    if 'word' in value:
        word=_mapping(value['word'])
        result['word']=_fields(word,('scope',))
        result['word']['reader_errors']=_strings(word.get('reader_errors'))
        omissions=word.get('omissions',[])
        result['word']['omissions']=[_fields(row,('kind','part','location','reason'))
                                     for row in omissions if isinstance(row,dict)] if isinstance(omissions,list) else []
    if 'limits' in value:result['limits']=_strings(value['limits'])
    return result


def _comparison_diagnostics(value):
    value=_mapping(value)
    result=_counters(value,('pairs_total','literal_candidate_pairs','near_candidate_block_pairs','prompt_tokens_excluded'))
    result.update(_fields(value,('status','reason')))
    if 'cutoffs' in value:
        result['cutoffs']=_counters(value['cutoffs'],('blocks_omitted','tokens_omitted','repeated_shingle_positions_omitted',
            'literal_alignments_omitted','high_frequency_near_tokens_omitted','near_candidates_omitted'))
    if 'thresholds' in value:
        result['thresholds']=_fields(value['thresholds'],('literal_tokens','literal_lexical_words','document_max_blocks',
            'document_max_tokens','shingle_positions_per_document','near_min_tokens','near_window_tokens','near_window_overlap',
            'near_sequence_ratio','near_shared_lexical_words','near_posting_limit','literal_alignments_per_pair',
            'near_candidates_per_pair','snippets_per_rule_pair'))
    if 'pairs_flagged_by_rule' in value:
        result['pairs_flagged_by_rule']=_counters(value['pairs_flagged_by_rule'],_COMPARISON_KINDS)
    if 'metadata' in value:
        result['metadata']=_fields(value['metadata'],('close_seconds','timestamp_only_candidates','zip_timezone',
                                                     'zip_precision_seconds','filesystem_times_used'))
    return result


def _comparison(value):
    row=_fields(value,('kind','limits','evidence_count'))
    for field in ('students','labels','corroborates'):
        if field in value:row[field]=_strings(value[field])
    fields=('component','left','right','location_left','location_right','matching_tokens','matching_text',
            'document_frequency','class_size','frequency_basis','frequency_anchor','token_sequence_ratio',
            'shared_lexical_words','sha256','content_sha256','observed','requirement_id','gap_seconds',
            'precision_seconds','comparison_basis','threshold_seconds','meaning')
    evidence=value.get('evidence',[])
    row['evidence']=[_fields(item,fields) for item in evidence if isinstance(item,dict)] if isinstance(evidence,list) else []
    return row


def compatibility_snapshot(documents):
    omv=documents.get('omv') or {};word=documents.get('docx') or {}
    manifest=omv.get('manifest') or {};observed=manifest.get('jamovi-Archive-Version')
    coverage=word.get('evidence_coverage') or {}
    return _compatibility({'reader_contract':CONTRACT_VERSION,'network_check_performed':False,
            'omv':{'observed_archive':observed,'supported_archive':'11.0','archive_contract_supported':observed=='11.0',
                   'declared_application':str(manifest.get('Created-By','not declared'))[:200],
                   'application_version_certified':False,'reader_errors':list(omv.get('errors',[]))},
            'word':{'scope':coverage.get('scope','not available'),'omissions':coverage.get('omissions',[]),'reader_errors':list(word.get('errors',[]))},
            'limits':['Il formato archivio riconosciuto non certifica ogni versione o modulo jamovi.',
                      'Controllo offline delle versioni dichiarate; nessuna ricerca di aggiornamenti durante l’analisi.']})


def comparison_snapshot(documents):
    """Release numeric arrays after checks; retain exactly cross-check inputs."""
    output={}
    for kind,value in documents.items():
        if kind not in ('omv','docx'):continue
        if not isinstance(value,dict):
            output[kind]=None
            continue
        compact=_fields(value,('sha256','content_sha256'))
        if 'metadata' in value:
            # Non-scalar metadata remains invalid instead of exposing a later
            # date alias that the original inspector would not have selected.
            compact['metadata']={key:(item if item is None or type(item) in (str,int,float,bool) else False)
                                 for key,item in _mapping(value['metadata']).items() if key in _METADATA_FIELDS}
        if 'zip_metadata' in value:
            compact['zip_metadata']={key:({'iso':item.get('iso') if isinstance(item.get('iso'),str) else None}
                                        if isinstance(item,dict) else None)
                                     for key,item in _mapping(value['zip_metadata']).items()}
        if 'errors' in value:compact['errors']=_strings(value['errors'])
        if kind=='docx':
            sources=value.get('evidence_sources')
            if isinstance(sources,list):
                compact['evidence_sources']=[_fields(source,('id','text','location')) for source in sources if isinstance(source,dict)]
            else:
                compact['paragraphs']=[_fields(p,('index','text')) for p in value.get('paragraphs',[]) if isinstance(p,dict)]
                compact['tables']=[]
                for table in value.get('tables',[]):
                    if not isinstance(table,dict):continue
                    item=_fields(table,('index',))
                    item['rows']=[[str(cell) for cell in row] for row in table.get('rows',[])]
                    compact['tables'].append(item)
        output[kind]=compact
    return output


def forensic_snapshot(result):
    """No configurations, raw prompts, binaries or whole original documents."""
    fields=('student_count','planned_student_count','unexecuted_student_count','status','incomplete','comparison_status',
            'file_count','version','reference_available','linguistic','cache_stats','input_sha256')
    summary=result.get('summary') or {}
    students=[]
    for report in result.get('reports',[]):
        bundle=json.loads(export_evidence_json(report.get('evidence') or {}))
        student=_fields(report,('key','label'))
        student.update(evidence=bundle,compatibility=_compatibility(report.get('compatibility')))
        students.append(student)
    comparisons=[_comparison(row) for row in result.get('comparisons',[]) if isinstance(row,dict)]
    safe_summary=_fields(summary,tuple(field for field in fields if field!='cache_stats'))
    if 'cache_stats' in summary:safe_summary['cache_stats']=_cache_stats(summary['cache_stats'])
    return {'schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),
            'runtime':{'python':platform.python_version(),'platform':platform.platform()},
            'summary':safe_summary,'cache_stats':_cache_stats(result.get('cache_stats',{})),
            'comparison_diagnostics':_comparison_diagnostics(getattr(result.get('comparisons'),'diagnostics',{})),
            'students':students,'comparisons':comparisons,
            'limits':['Diagnostica locale; contiene identificativi e citazioni da custodire con i rapporti degli studenti.',
                      'Tracciabilità e punteggi di supporto non sono probabilità di frode o fiducia semantica calibrata.']}
