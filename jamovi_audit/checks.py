"""Conservative evidence checks for the seven-part BIG5 assignment.

No formula in a submission is executed. A verified saved output establishes its
presence, not the correctness of the estimator or the authorship of the file.
Optional reference numeric_claims: {id,part,pattern(named value),value,location,
 description}; reference column_values maps exact column names to row values.
"""
import math
import re
import statistics
from collections import Counter
from itertools import combinations

STATES = ('Verificato', 'Mancante', 'Errato', 'Da verificare', 'Non verificabile', 'Non applicabile')
SCALES = ('A', 'C', 'E', 'N', 'O')
ITEMS = tuple(f'{s}{i}' for s in SCALES for i in range(1, 6))
REVERSED = {'A1', 'C4', 'C5', 'E1', 'E2', 'O2', 'O5'}
QUESTIONS = {
 1: [('ordinal', 'Natura ordinale degli item e trattamento dei punteggi', 'Distinguere ordine delle risposte e trattamento della media di scala.')],
 2: [('filter', 'Numero esclusi e campione analizzato', 'Confrontare età, filtro attivo e conteggi.'), ('reverse', 'Inversione e conseguenze della mancata inversione', 'Spiegare orientamento comune degli item e alterazione del punteggio.')],
 3: [('raw', 'Limiti del punteggio grezzo e rappresentatività', 'Contestualizzare punteggio e campione.'), ('ceiling', 'Concentrazione verso il massimo e discriminazione', 'Usare distribuzioni e variabilità, senza dedurre dal solo valore medio.')],
 4: [('ids', 'Punteggi T degli ID 61789 e 61731 e percentuale sopra 60', 'Confrontare valori individuali e frequenze.'), ('norms', 'Standardizzazione sul campione e norme esterne', 'Distinguere i due riferimenti.')],
 5: [('differences', 'Direzione, significatività, grandezza delle differenze e norme per genere', 'Usare medie, ordine dei gruppi, p, IC e d; motivare le norme separate.')],
 6: [('associations', 'Associazioni, distinzione delle scale e Spearman per education', 'Interpretazione prudente; education ordinale.')],
 7: [('sem', 'Errore standard di misura con omega e intervallo individuale', 'SEM = DS × radice(1−omega); intervallo dichiarando livello e riferimento.'), ('precision', 'Precisione delle scale e item O con item-rest più bassa', 'Confrontare affidabilità, SEM e correlazioni item-rest.')]
}


def _flatten(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)
    else:
        yield value


def _num(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _cell_values(column):
    for value in column.get('cells', []):
        # Readers may retain typed cell dictionaries for unsupported types.
        if isinstance(value, dict):
            yield value.get('value', value.get('s', value.get('d')))
        else:
            yield value


def _table_columns(node):
    return [c for c in node.get('columns', []) if c.get('visible', True) and any(v is not None and v != '' for v in _cell_values(c))]


def _row_label(value):
    # Actual saved jmv rowNames are JSON-quoted strings; cell labels are plain.
    if isinstance(value,str) and len(value)>=2 and value[0]==value[-1]=='"':
        try:
            import json
            return json.loads(value)
        except ValueError:
            pass
    return value


def _rows_for_stat(node, columns, stat_name):
    tag=re.search(r'\[(.+)\]$',stat_name)
    preferred=('var['+tag.group(1)+']',) if tag else ()
    for name in preferred+('name','dep'):
        col=next((c for c in columns if c.get('name')==name),None)
        if col is not None:
            return [_row_label(v) for v in _cell_values(col)]
    return [_row_label(v) for v in node.get('row_names',[])]


def _reported_number(text, start, allow_percent=False):
    """Consume the complete numeric token, never validate a decimal prefix."""
    match=re.match(r'[+-]?(?:\d[\d.,\u00a0\u202f ]*|[.,]\d+)(?:[eE][+-]?\d+)?(?:\s*%)?',text[start:])
    if not match:return None,None,''
    raw=match.group().strip()
    tail=text[start+match.end():]
    if tail and ((not match.group()[-1].isspace() and (tail[0].isalnum() or tail[0]=='_')) or re.match(r'\s*[×*]\s*10',tail)):
        return None,None,raw+tail[:20]
    percent=raw.endswith('%')
    if percent and not allow_percent:return None,None,raw
    raw_number=raw[:-1].rstrip() if percent else raw
    # A sentence-ending dot/comma is punctuation, not another decimal digit.
    raw_number=raw_number.rstrip('., ')
    exponent_match=re.search(r'[eE]([+-]?\d+)$',raw_number)
    exponent=int(exponent_match.group(1)) if exponent_match else 0
    mantissa=raw_number[:exponent_match.start()] if exponent_match else raw_number
    if abs(exponent)>308:return None,None,raw
    if re.search(r'[ \u00a0\u202f]',mantissa):
        if not re.fullmatch(r'[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?',mantissa):return None,None,raw
        mantissa=re.sub(r'[ \u00a0\u202f]','',mantissa)
    if '.' in mantissa and ',' in mantissa:
        decimal='.' if mantissa.rfind('.')>mantissa.rfind(',') else ','
        grouping=',' if decimal=='.' else '.'
        integer,fraction=mantissa.rsplit(decimal,1)
        if not re.fullmatch(r'[+-]?\d{1,3}(?:'+re.escape(grouping)+r'\d{3})+',integer) or not fraction.isdigit():return None,None,raw
        normalized=integer.replace(grouping,'')+'.'+fraction
        decimals=len(fraction)
    elif mantissa.count('.')>1 or mantissa.count(',')>1:
        separator='.' if '.' in mantissa else ','
        if not re.fullmatch(r'[+-]?\d{1,3}(?:'+re.escape(separator)+r'\d{3})+',mantissa):return None,None,raw
        normalized=mantissa.replace(separator,'');decimals=0
    else:
        normalized=mantissa.replace(',','.')
        decimals=len(normalized.split('.')[1]) if '.' in normalized else 0
    value=_num(normalized+('e'+str(exponent) if exponent_match else ''))
    try:tolerance=.5*10.0**(exponent-decimals)+1e-10
    except OverflowError:return None,None,raw
    return value,tolerance,raw


def _status_problem(obj):
    status = str(obj.get('status', '')).lower()
    if obj.get('error') or status in ('error', 'failed', '4'):
        return 'Errato', 'Risultato in errore: ' + str(obj.get('error') or status)
    stale_not_applicable = obj.get('stale_applicable') is False and str(obj.get('kind', '')).lower() in ('table', 'image')
    if obj.get('stale') and not stale_not_applicable:
        return 'Non verificabile', 'Risultato obsoleto, da ricalcolare in jamovi.'
    if status in ('running', 'inited', 'none', 'waiting', 'rendering', '0', '1', '2', '5'):
        return 'Non verificabile', 'Calcolo non completato.'
    if obj.get('visible') is False:
        return 'Mancante', 'Output salvato ma nascosto.'
    return None


def check_submission(submission, omv, docx, reference=None):
    reference = reference or {}
    findings = []
    paths = {kind: submission.get(kind, []) for kind in ('omv', 'docx')}
    def add(identifier, part, source, status, requirement, observed, expected, location):
        findings.append(dict(id=identifier, part=part, source=source, status=status, requirement=requirement,
                             observed=str(observed), expected=str(expected), location=str(location)))
    for kind, parsed in [('OMV', omv), ('DOCX', docx)]:
        found = paths[kind.lower()]
        state = 'Mancante' if not found and parsed is None else 'Non verificabile' if parsed is None or parsed.get('errors') else 'Verificato'
        add('FILE.' + kind, 0, 'Word' if kind == 'DOCX' else 'OMV', state, 'File ' + kind + ' presente e leggibile',
            '; '.join(map(str, (parsed or {}).get('errors', []))) or str(found), 'Un solo file associato senza ambiguità.', str(found))
        if len(found) > 1:
            add('FILE.' + kind + '.ambiguous', 0, kind, 'Da verificare', 'Versione della consegna', found, 'Selezionare la versione corretta; nessuna associazione arbitraria.', found)
    for i, warning in enumerate(submission.get('warnings', [])):
        add('FILE.warning.' + str(i), 0, 'Inventario', 'Da verificare', 'Associazione e inventario', warning, 'Controllare i percorsi originali.', submission.get('key', ''))

    columns = (omv or {}).get('columns', {})
    analyses = (omv or {}).get('analyses', [])
    unreadable = omv is None or bool(omv.get('errors'))
    absent = 'Non verificabile' if unreadable else 'Mancante'
    def source_name(name):
        return 'id' if name == 'ID' and name not in columns and 'id' in columns else name
    def values(name):
        return columns.get(source_name(name), {}).get('data', [])
    def column_check(name, part, measure=None):
        actual_name = source_name(name)
        col = columns.get(actual_name)
        if col is None:
            state, obs = absent, 'Colonna non riscontrata.'
        elif not measure:
            state, obs = 'Verificato', 'Colonna salvata; ' + str(len(values(name))) + ' righe decodificate.'
        else:
            meta = col.get('meta', {})
            actual = meta.get('measureType', meta.get('measure_type'))
            norm = str(actual).lower()
            state = 'Non verificabile' if actual is None else 'Verificato' if norm in measure else 'Errato'
            obs = 'Livello di misura salvato: ' + str(actual)
        add(f'P{part}.{name}.column', part, 'OMV', state, f'Variabile {name}' + ('; livello di misura' if measure else ''), obs, '/'.join(measure or ['Colonna presente.']), 'dati/' + actual_name)
    for name, measures in [('ID', ('id', 'identifier')), ('gender', ('nominal',)), ('education', ('ordinal',)), ('age', ('continuous',))] + [(n, ('ordinal',)) for n in ITEMS]:
        column_check(name, 1, measures)
    for name in ('gender', 'education') + ITEMS:
        levels = columns.get(name, {}).get('levels', [])
        add(f'P1.{name}.levels', 1, 'OMV', absent if name not in columns else 'Da verificare' if levels else 'Non verificabile',
            f'Etichette dei livelli di {name}', str(levels)[:1500], 'Livelli e loro significato conformi alla consegna; confrontare le etichette.', 'dati/' + name + '/livelli')

    age = values('age')
    adults = [i for i, a in enumerate(age) if _num(a) is not None and _num(a) >= 18]
    filters = [c.get('meta', {}) for c in columns.values() if str(c.get('meta', {}).get('columnType', '')).lower() == 'filter']
    filters += (omv or {}).get('metadata', {}).get('dataSet', {}).get('filters', []) or []
    if isinstance(filters, dict):
        filters = list(filters.values())
    filt = [f for f in filters if isinstance(f, dict) and re.search(r'\bage\s*>=\s*18\b', str(f.get('formula', '')), re.I)]
    active = [f for f in filt if f.get('active', f.get('filterActive')) is True]
    state = 'Verificato' if active else 'Errato' if filt and all(f.get('active', f.get('filterActive')) is False for f in filt) else 'Non verificabile'
    add('P2.filter', 2, 'OMV', absent if unreadable else state, 'Filtro age >= 18 attivo', str(filt), 'Formula e stato attivo espliciti; conteggi calcolati dai dati non provano l’attivazione.', 'dati/filtri')
    add('P2.filter.counts', 2, 'OMV', 'Verificato' if age else 'Non verificabile', 'Conteggio indipendente del criterio adulti',
        f'Righe {len(age)}; adulti {len(adults)}; esclusi {len(age)-len(adults)}.', 'Conteggio del criterio age ≥ 18, distinto dall’applicazione effettiva del filtro.', 'dati/age')
    def calculated(name, part, expected, description):
        if active and str(columns.get(name,{}).get('meta',{}).get('columnType','')).lower()=='output':
            adult_set=set(adults)
            expected=[value if i in adult_set else None for i,value in enumerate(expected)]
            description+=' Colonna Output di jamovi: confronto delle righe adulte; valori delle righe filtrate fuori perimetro.'
        if name not in columns:
            state, obs = absent, 'Colonna non presente.'
        elif not expected or len(expected) != len(values(name)):
            state, obs = 'Non verificabile', 'Dati sorgente insufficienti o lunghezze diverse.'
        else:
            mismatches = [i+1 for i, (a, b) in enumerate(zip(values(name), expected)) if b is not None and (_num(a) is None or not math.isclose(_num(a), b, rel_tol=1e-7, abs_tol=1e-7))]
            n = sum(b is not None for b in expected)
            state = 'Errato' if mismatches else 'Verificato' if n else 'Non verificabile'
            obs = f'Confrontate {n} righe; discordanti {len(mismatches)}; prime righe: {mismatches[:12]}.'
        add(f'P{part}.{name}.values', part, 'OMV', state, f'Valori di {name}', obs, description + ' Tolleranza assoluta/relativa 1e-7; le righe senza sorgente valida non sono certificate.', 'dati/' + name)
    calculated('A1_Rev', 2, [7-_num(x) if _num(x) is not None else None for x in values('A1')], '7 − A1')
    for s in SCALES:
        names = [f'{s}{i}' for i in range(1,6)]
        rows = list(zip(*(values(n) for n in names))) if all(n in columns for n in names) else []
        expected = [sum(7-_num(v) if n in REVERSED else _num(v) for n, v in zip(names, row))/5 if all(_num(v) is not None for v in row) else None for row in rows]
        calculated(s, 2, expected, 'Media di cinque item orientati; inversi: ' + ', '.join(n for n in names if n in REVERSED))
        if s != 'A':
            meta = columns.get(s, {}).get('meta', {})
            provenance = {k: v for k, v in meta.items() if k in ('outputAnalysisId', 'outputOptionName', 'outputName', 'analysisId')}
            linked = [a for a in analyses if a.get('id') == meta.get('outputAnalysisId') and a.get('name', '').lower() == 'reliability' and a.get('ns', 'jmv') == 'jmv' and isinstance(a.get('options'),dict) and set(names) == set(_flatten(a['options'].get('vars', []))) and not _status_problem(a)]
            proven = len(linked) == 1 and meta.get('outputOptionName') == 'meanScoreOV'
            add(f'P2.{s}.provenance', 2, 'OMV', 'Verificato' if proven else 'Non verificabile', f'{s} salvato dall’analisi di affidabilità', provenance or 'Provenienza non conservata o non risolta.', 'Collegamento esplicito della colonna a meanScoreOV dell’analisi pertinente; una formula corretta non dimostra questa procedura.', 'dati/' + s + '/metadati')
    calculated('A_controllo', 2, [_num(x) for x in values('A')], 'Coincidenza riga per riga con A.')
    column_check('age_fasce', 2)
    add('P2.age_fasce.boundaries', 2, 'OMV', 'Non verificabile' if not columns.get('age_fasce') else 'Da verificare',
        'Fasce età 18–25, 26–40, oltre 40', str(columns.get('age_fasce', {}).get('meta', {})), 'Verificare formula e corrispondenza fra codici/etichette, inclusi i confini 25/26/40/41.', 'dati/age_fasce')

    nvals = values('N'); nums = [_num(nvals[i]) for i in adults if i < len(nvals) and _num(nvals[i]) is not None]
    samplemean = statistics.mean(nums) if nums else None
    samplesd = statistics.stdev(nums) if len(nums) > 1 else None
    adult_rows = set(adults)
    calculated('Zscore_N', 4, [(_num(x)-samplemean)/samplesd if i in adult_rows and _num(x) is not None and samplesd else None for i,x in enumerate(nvals)], 'Z = (N − media adulti) / DS campionaria adulti; confronto limitato alle righe adulte. Le righe escluse dal filtro sono fuori dal perimetro di questo confronto.')
    genders = values('gender')
    norms = {1: (2.99, 1.14), 2: (3.29, 1.22)}
    z_norm = [(_num(n)-norms[_num(g)][0])/norms[_num(g)][1] if _num(n) is not None and _num(g) in norms else None for n,g in zip(nvals,genders)]
    calculated('Zscore_N_norme', 4, z_norm, 'Norme gender 1: media 2.99, DS 1.14; gender 2: media 3.29, DS 1.22.')
    calculated('Tscore_N', 4, [50+10*z if z is not None else None for z in z_norm], 'T = 50 + 10 × Zscore_N_norme.')
    column_check('Tscore_N_fasce', 4)
    add('P4.Tscore_N_fasce.boundaries', 4, 'OMV', 'Da verificare' if 'Tscore_N_fasce' in columns else absent,
        'Fasce T: sotto 40; 40–60 inclusi; sopra 60', str(columns.get('Tscore_N_fasce', {}).get('meta', {})), 'Controllare codici e formula, soprattutto T = 40 e T = 60 nella fascia centrale.', 'dati/Tscore_N_fasce')

    def output(identifier, part, analysis_name, vars_, nodes, stats=(), split=False, options=None, text=None):
        requested = set(vars_)
        candidates = []
        unknown = False
        for a in analyses:
            if str(a.get('name', '')).lower() != analysis_name.lower():
                continue
            if a.get('ns', 'jmv') not in ('jmv', ''):
                unknown = True
                continue
            opts = a.get('options', {})
            if not isinstance(opts, dict):
                continue
            avars = set(str(x) for x in _flatten(opts.get('vars', [])))
            if not requested <= avars:
                continue
            by = set(str(x) for x in _flatten(opts.get('splitBy', [])))
            if split != ('gender' in by) and analysis_name.lower() == 'descriptives':
                continue
            if analysis_name.lower() == 'ttestis' and opts.get('group') != 'gender':
                continue
            if options and any(opts.get(k) != v for k,v in options.items()):
                continue
            candidates.append(a)
        matches = []
        for a in candidates:
            for node in a.get('results', []):
                context = ' '.join(map(str, node.get('path', []))) + ' ' + str(node.get('name', '')) + ' ' + str(node.get('title', ''))
                if any(re.search(r'(?<![\w])'+re.escape(n)+r'(?![\w])', context, re.I) for n in nodes):
                    matches.append((a,node))
        status, obs, location = absent, 'Analisi o output richiesto non riscontrato.', 'analisi/' + analysis_name
        if unknown and not candidates:
            status,obs = 'Non verificabile','Analisi con namespace non supportato: non si deduce l’assenza del requisito.'
        if candidates and not matches:
            # A supported analysis with no leaves is a real missing output.
            obs = 'Analisi pertinente configurata; output richiesto non salvato.'
        problems = []
        needs_image = identifier.endswith(('.plot', '.histogram', '.boxplot', '.heatmap'))
        accepted_node = None
        for a,node in matches:
            loc = 'analisi/' + str(a.get('id', '?')) + '/' + '/'.join(map(str,node.get('path', [])))
            problem = _status_problem(a) or _status_problem(node)
            if problem:
                problems.append((*problem,loc)); continue
            if needs_image and node.get('kind', '').lower() != 'image':
                continue
            if not needs_image and node.get('kind', '').lower() != 'table':
                continue
            if node.get('kind', '').lower() == 'image':
                if not node.get('image_exists'):
                    problems.append(('Mancante','Risorsa immagine inesistente.',loc)); continue
                # A multi-variable plot must itself identify the required variable.
                context = set(re.findall(r'[\w]+', ' '.join(map(str,node.get('path', []))) + ' ' + str(node.get('title', ''))))
                if len(requested)==1 and not requested <= context and len(set(_flatten(a['options'].get('vars',[])))) > 1:
                    problems.append(('Non verificabile','Immagine non attribuibile alla variabile richiesta.',loc)); continue
                status, obs, location = 'Verificato','Immagine presente e visibile: '+str(node.get('image_path')),loc
                accepted_node = node
                break
            cols = _table_columns(node)
            if not cols:
                problems.append(('Mancante','Tabella vuota o senza celle leggibili.',loc)); continue
            # Actual rows/columns must identify all requested variables, except
            # single-scale reliability where rows are explicitly scale/items.
            labels = set(str(x) for c in cols for x in _cell_values(c)) | {str(c.get('name')) for c in cols} | {str(c.get('title')) for c in cols} | {str(_row_label(v)) for v in node.get('row_names',[])}
            if 'frequencies' in nodes:
                labels |= set(map(str,node.get('path',[])))
            needs_item_rows = analysis_name.lower() == 'reliability' and 'items' in nodes
            if (analysis_name.lower() != 'reliability' or needs_item_rows) and not requested <= labels:
                problems.append(('Non verificabile','Tabella presente; righe/colonne non attribuibili a tutte le variabili richieste.',loc)); continue
            cnames = {str(c.get('name', '')).lower() for c in cols}
            missing_stats = [st for st in stats if not any(re.fullmatch(st, n, re.I) for n in cnames)]
            if analysis_name.lower() in ('ttestis','reliability'):
                missing_stats=[]
                for st in stats:
                    numeric_columns=[c for c in cols if re.fullmatch(st,str(c.get('name','')),re.I)]
                    complete=False
                    for col in numeric_columns:
                        cells=list(_cell_values(col))
                        if analysis_name.lower()=='reliability' and 'scale' in nodes:
                            complete=bool(cells) and _num(cells[0]) is not None
                        else:
                            rows=_rows_for_stat(node,cols,str(col.get('name','')))
                            complete=all(sum(label==variable for label in rows)==1 and any(label==variable and i<len(cells) and _num(cells[i]) is not None for i,label in enumerate(rows)) for variable in requested)
                        if complete:break
                    if not complete:missing_stats.append(st)
            elif analysis_name.lower()=='corrmatrix':
                byname={str(c.get('name','')):c for c in cols}
                channels=('r','rp','rciu','rcil','n') if (options or {}).get('pearson') else ('rho',)
                pairs=list(combinations(vars_,2)) if (options or {}).get('pearson') else [('education',v) for v in vars_ if v!='education']
                for channel in channels:
                    rows=[_row_label(v) for v in _cell_values(byname.get('.name['+channel+']',{}))]
                    for left,right in pairs:
                        present=False
                        for column,row in ((left,right),(right,left)):
                            cells=list(_cell_values(byname.get(column+'['+channel+']',{})))
                            if sum(label==row for label in rows)==1 and any(label==row and i<len(cells) and _num(cells[i]) is not None for i,label in enumerate(rows)):
                                present=True
                        if not present:missing_stats.append(f'{left}↔{right}[{channel}]')
            if analysis_name.lower() == 'descriptives' and any(n.startswith('stat[') for n in cnames):
                # jmv 2.7.4 encodes a statistic as adjacent stat[mean] and
                # O[mean] columns (or O[meanmale], O[meanfemale] when split).
                # Verify actual numeric cells for the requested variable, not
                # merely the presence of the statistic's label column.
                byname={str(c.get('name','')):c for c in cols}
                groups={str(v) for c in cols if str(c.get('name','')).startswith('var1[') for v in _cell_values(c) if v is not None}
                suffixes=groups if split else {''}
                missing_stats=[]
                for st in stats:
                    saved_stat={'sw':'sww','swp':'sw'}.get(st,st)
                    complete=bool(suffixes) and (not split or len(suffixes)>=2)
                    for suffix in suffixes:
                        stat_key=saved_stat+suffix
                        if 'stat['+stat_key+']' not in byname:
                            complete=False
                        for variable in requested:
                            col=byname.get(variable+'['+stat_key+']',{})
                            if not any(_num(v) is not None for v in _cell_values(col)):
                                complete=False
                    if not complete:missing_stats.append(st)
            if missing_stats:
                problems.append(('Mancante','Colonne richieste mancanti/nascoste/vuote: '+', '.join(missing_stats),loc)); continue
            status, obs, location = 'Verificato','Tabella presente con celle e colonne richieste; correttezza numerica non ricalcolata.',loc
            accepted_node = node
            break
        else:
            if problems:
                status,obs,location = sorted(problems,key=lambda x: {'Errato':0,'Non verificabile':1,'Mancante':2}.get(x[0],3))[0]
            elif candidates and any(_status_problem(a) for a in candidates):
                status,obs = next(_status_problem(a) for a in candidates if _status_problem(a))
            elif candidates and any(n.get('kind') == 'unknown' for a in candidates for n in a.get('results', [])):
                status,obs = 'Non verificabile','Componente risultato non supportata dal lettore.'
        add(identifier,part,'OMV',status,text or identifier,obs,'Output pertinente effettivamente salvato, visibile e leggibile, senza errore o calcolo incompleto; variabili '+', '.join(vars_)+'. La presenza è distinta dalla verifica di aggiornamento numerico.',location)
        if accepted_node is not None and accepted_node.get('stale_applicable') is False:
            add(identifier+'.freshness',part,'OMV','Non verificabile','Aggiornamento numerico di '+(text or identifier),
                'Presenza verificata. L’indicatore grezzo stale='+str(bool(accepted_node.get('stale')))+' non ha semantica di aggiornamento applicabile alle tabelle/immagini in questo formato jamovi.',
                'Riaprire e ricalcolare in jamovi per accertare la coerenza con dati e filtro correnti; il solo risultato salvato non certifica l’aggiornamento.',location)

    for name in ITEMS:
        for stat in ('min','max','missing'):
            output(f'P3.{name}.{stat}',3,'descriptives',[name],['descriptives','descriptivesT'],[stat],text=f'{name}: {stat} su adulti')
    for name in ('gender','education','age_fasce'):
        output(f'P3.{name}.freq',3,'descriptives',[name],['frequencies'],text=f'Frequenze di {name}')
    for split in (False,True):
        for s in SCALES:
            prefix=f'P3.{s}'+('.gender' if split else '')
            for stat in ('n','missing','mean','median','sd','min','max','skew','kurt','sw','swp'):
                output(prefix+'.'+stat,3,'descriptives',[s],['descriptives','descriptivesT'],[stat],split,text=f'{s}: {stat}'+(' separatamente per gender' if split else ''))
            for plot,names in [('histogram',['hist','histogram']),('boxplot',['box','boxplot'])]:
                output(prefix+'.'+plot,3,'descriptives',[s],names,split=split,text=f'{s}: {plot}'+(' per gender' if split else ''))
    output('P4.T.frequencies',4,'descriptives',['Tscore_N_fasce'],['frequencies'],text='Frequenze delle tre fasce T')
    for s in ('N','A'):
        for name, cols, nodes in [('student', [r'stat\[stud\]',r'df\[stud\]',r'p\[stud\]'],['ttest']), ('mean_difference',[r'md\[stud\]',r'cil\[stud\]',r'ciu\[stud\]'],['ttest']),('cohen_d',[r'es\[stud\]'],['ttest']),('descriptives',[r'mean\[1\]',r'mean\[2\]'],['desc']),('levene',['f','p'],['eqv']),('shapiro',['w','p'],['norm'])]:
            output(f'P5.{s}.{name}',5,'ttestIS',[s],nodes,cols,text=f'{s}: {name}; gruppi gender')
        output(f'P5.{s}.plot',5,'ttestIS',[s],['desc'],text=f'{s}: grafico descrittivo per gender')
        levene=[]
        for a in analyses:
            opts=a.get('options',{})
            if str(a.get('name','')).lower()=='ttestis' and a.get('ns','jmv') in ('jmv','') and isinstance(opts,dict) and opts.get('group')=='gender' and s in set(_flatten(opts.get('vars',[]))):
                for node in a.get('results',[]):
                    if node.get('name')=='eqv' and not (_status_problem(a) or _status_problem(node)):
                        cs=_table_columns(node); labels=next((list(_cell_values(c)) for c in cs if c.get('name')=='name'),[])
                        ps=next((list(_cell_values(c)) for c in cs if c.get('name')=='p'),[])
                        if s in labels and labels.index(s)<len(ps):
                            pvalue=_num(ps[labels.index(s)])
                            levene.append(pvalue if pvalue is not None and 0<=pvalue<=1 and labels.count(s)==1 else None)
        if len(levene)==1 and levene[0] is not None and levene[0]>=.05:
            add(f'P5.{s}.welch',5,'OMV','Non applicabile',f'Welch per {s} se Levene significativo',f'p Levene = {levene[0]}','Welch richiesto quando p < .05.','analisi/ttestIS/eqv')
        elif len(levene)==1 and levene[0] is not None:
            output(f'P5.{s}.welch',5,'ttestIS',[s],['ttest'],[r'stat\[welc\]',r'df\[welc\]',r'p\[welc\]'],text=f'Welch per {s}; Levene significativo')
        else:
            add(f'P5.{s}.welch',5,'OMV','Non verificabile',f'Condizione Welch per {s}','p di Levene non univocamente leggibile.','Controllare Levene prima di richiedere Welch.','analisi/ttestIS/eqv')
    for label,vs,opts,stats in [('pearson',list(SCALES)+['age'],{'pearson':True},[r'\.name\[r\]',r'\.name\[rp\]',r'\.name\[rciu\]',r'\.name\[rcil\]',r'\.name\[n\]']),('spearman',['education']+list(SCALES),{'spearman':True},[r'\.name\[rho\]'])]:
        output('P6.'+label,6,'corrMatrix',vs,['matrix'],stats,options=opts,text='Matrice '+label+' con variabili richieste')
    add('P6.pearson.stars',6,'OMV','Non verificabile','Asterischi di significatività Pearson','Annotazioni celle non validate in questa versione.','Controllare asterischi effettivi, IC 95% e N per ogni associazione.','analisi/corrMatrix/matrix')
    for s in ('N','O'):
        itemnames=[s+str(i) for i in range(1,6)]
        for stat in ('alpha','omega','mean','sd'):
            output(f'P7.{s}.{stat}',7,'reliability',itemnames,['scale'],[stat],text=f'{s}: {stat} della scala')
        for stat in ('itemRestCor','alpha','omega'):
            output(f'P7.{s}.items.{stat}',7,'reliability',itemnames,['items'],[stat],text=f'{s}: item-rest / eliminazione item: {stat}')
        output(f'P7.{s}.heatmap',7,'reliability',itemnames,['corPlot'],text=f'{s}: mappa di calore correlazioni item')
        add(f'P7.{s}.orientation',7,'OMV','Da verificare' if not unreadable else absent,f'Orientamento item di {s}','Rivedere revItems e possibili colonne già invertite.','Evitarne la doppia inversione; O2 e O5 inversi, N nessun inverso.','analisi/reliability/opzioni')
    for name,expected in reference.get('column_values',{}).items():
        calculated(name,4 if name.startswith(('Z','T')) else 2,expected,'Riferimento indipendente validato.')

    # Word evidence remains reviewable: a heading or copied question is never a pass.
    paragraphs=list((docx or {}).get('paragraphs',[]))
    for table in (docx or {}).get('tables',[]):
        for ri,row in enumerate(table.get('rows',[])):
            for ci,text in enumerate(row):
                paragraphs.append({'index':f"tabella {table.get('index', '?')}, riga {ri+1}, cella {ci+1}",'text':text})
    docerror=docx is None or bool(docx.get('errors'))
    for part, questions in QUESTIONS.items():
        for ident,question,criterion in questions:
            excerpts=[p for p in paragraphs if re.search(r'\b'+str(part)+r'\b',str(p.get('text',''))) or any(w in str(p.get('text','')).lower() for w in {'ordinal','inversion','campione','punteggi','affidabil','correlaz'})]
            response=' | '.join(f"§{p.get('index','?')}: {p.get('text','')}" for p in excerpts[:5])
            add(f'WORD.Q{part}.{ident}',part,'Word','Non verificabile' if docerror else 'Da verificare',question,response or 'Nessuna risposta attribuibile automaticamente; esaminare la parte pertinente.',criterion,'paragrafi Word; pagine non determinate')
    formatting=(docx or {}).get('formatting',{})
    format_status = 'Errato' if formatting.get('issues') else 'Verificato' if formatting.get('complete') else 'Non verificabile'
    add('WORD.format',0,'Word',format_status,'Times New Roman 12; interlinea doppia',str(formatting)[:2000], 'Distinguere stili ereditati e formattazione esplicita.','Word/stili e paragrafi')
    claims=list(reference.get('numeric_claims',[]))
    claimed_ids={c.get('id') for c in claims}
    def own_claim(identifier,part,pattern,value,location,description):
        if identifier not in claimed_ids:
            claims.append(dict(id=identifier,part=part,pattern=pattern,value=value,location=location,description=description))
    if age and not unreadable:
        own_claim('adult_n',2,r'\bcampione\s+(?:adulto|analizzato|filtrato)\s*(?:(?:è|e)\s+)?(?:(?:composto|costituito)\s+da\s+|di\s+|[=:]\s*)(?P<value>\d+)\b',len(adults),'dati/age, criterio ≥18','N del campione adulto dichiarato')
        own_claim('excluded_n',2,r'\b(?:esclusi|escluse)\s*(?:dal\s+filtro\s*)?(?:[=:]\s*|sono\s+)?(?P<value>\d+)\b',len(age)-len(adults),'dati/age, criterio ≥18','Numero esclusi dichiarato')
    for row,identifier in enumerate(values('ID')):
        if _num(identifier) not in (61789,61731):
            continue
        identifier = int(_num(identifier))
        if row<len(values('Tscore_N')) and _num(values('Tscore_N')[row]) is not None and not unreadable:
            own_claim('T_ID_'+str(identifier),4,r'\bID\s*'+str(identifier)+r'\b[^\n.;]{0,80}?\b(?:T|Tscore_N|punteggio\s+T)\s*(?:[=:]|(?:è|e)\s+(?:pari\s+a)?)\s*(?P<value>[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))',values('Tscore_N')[row],'dati/Tscore_N, ID '+str(identifier),'Punteggio T dichiarato per ID '+str(identifier))
    for claim in claims:
        pattern=claim.get('pattern','')
        try:
            regex=re.compile(pattern,re.I)
        except re.error:
            continue
        matches=[]
        for p in paragraphs:
            for m in regex.finditer(str(p.get('text',''))):
                value=m.groupdict().get('value')
                if value:
                    actual,tolerance,raw=_reported_number(str(p.get('text','')),m.start('value'),claim.get('unit')=='percent')
                    expected=_num(claim.get('value'))
                    comp=m.groupdict().get('op','=')
                    ok=(expected is not None and actual is not None and tolerance is not None and (expected<actual if comp=='<' else expected<=actual if comp=='<=' else abs(expected-actual)<=tolerance))
                    matches.append(('Non verificabile' if actual is None or expected is None else 'Verificato' if ok else 'Errato',p,raw,tolerance))
        for j,(state,p,raw,tolerance) in enumerate(matches):
            add('WORD.NUM.'+claim.get('id','claim')+('' if j==0 else '.'+str(j)),claim.get('part',0),'OMV–Word',state,claim.get('description','Affermazione numerica'),p.get('text',''),f"{claim.get('value')}; tolleranza arrotondamento {tolerance if tolerance is not None else 'numero/unità non risolvibile'}; token numerico completo {raw}.",f"Word §{p.get('index','?')} ↔ {claim.get('location','riferimento validato')}")
    counts=Counter(f['status'] for f in findings)
    return dict(key=submission.get('key',''),label=submission.get('label',submission.get('key','')),findings=findings,
                coverage={'total':len(findings),'by_status':dict(counts),'automatic':counts['Verificato']+counts['Errato']+counts['Mancante']+counts['Non applicabile'],
                          'limitations':['Output statici non ricalcolati; nessuna inferenza sull’autore.','Concetti, formule equivalenti, stili ereditati e provenienza dei punteggi richiedono revisione.','La presenza di un output non certifica che il filtro adulti fosse applicato al momento del calcolo.']},paths=paths)
