"""Single escaped, self-contained report and conservative cross-submission evidence."""
from collections import Counter, defaultdict
from html import escape
from itertools import combinations
import json
import re
import unicodedata


def _normal(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text)).casefold()).strip()


# Deliberately omit unavoidable technical vocabulary from the lexical-content
# gate. These words still participate in a match, but cannot alone create one.
_STOP = set("il lo la i gli le un uno una di del della delle dei degli da dal dalla a al alla alle ai e ed o per con su nel nella nello nei negli non che come si sono essere stato stata questo questa questi quelle quanto sia anche più meno mio mia mi ho ha è fra tra durante caso ogni ciascun tale hanno può si cui se sul sullo sulla".split())
_TECH = set("campione adulto adulti analizzato filtrato composto costituito comprende persone partecipanti soggetti media mediana scala scale punteggio punteggi deviazione standard item items ordinali ordinale variabile variabili gender education age statistica statistiche descrittive frequenza frequenze correlazione correlazioni significatività significativo significativa pearson spearman shapiro wilk levene alpha alfa omega cronbach cohen intervallo confidenza percentuale test student welch risultato risultati valore valori minimo massimo mancanti asimmetria curtosi affidabilità neuroticismo amicalità coscienziosità estroversione apertura formula formule".split())
_TOKEN = re.compile(r"\w+",re.UNICODE)


class ComparisonResults(list):
    """List-compatible evidence plus transparent candidate-search coverage."""
    def __init__(self):
        super().__init__()
        self.diagnostics={}


def _tokens(text):
    return [m.group().casefold() for m in _TOKEN.finditer(unicodedata.normalize('NFKC',str(text)))]


def _lexical(tokens):
    return {w for w in tokens if w and len(w)>=4 and w.isalpha() and w not in _STOP and w not in _TECH}


def _document_blocks(document):
    result=list((document or {}).get('paragraphs',[]))
    for table in (document or {}).get('tables',[]):
        for ri,row in enumerate(table.get('rows',[])):
            for ci,value in enumerate(row):
                result.append({'index':f"tabella {table.get('index','?')}, riga {ri+1}, cella {ci+1}",'text':str(value)})
    return result


def _prepared(document, excluded, prompt_shingles, diagnostics):
    blocks=[];stream=[];positions=[]
    rawblocks=_document_blocks(document)
    if len(rawblocks)>2000:
        diagnostics['cutoffs']['blocks_omitted']+=len(rawblocks)-2000
    for p in rawblocks[:2000]:
        text=str(p.get('text','')).strip();norm=_normal(text);ts=_tokens(text)
        instruction=bool(re.fullmatch(r'parte\s*\d+\s*[.:—-]?',norm) or re.match(r'^(?:\d+[.)]\s*)?(?:spiegare|calcolare|descrivere|riportare|confrontare|interpretare|indicare|verificare|discutere|determinare|eseguire|costruire|perché|quale|quali|come si)\b',norm))
        if not ts or norm in excluded or (instruction and '?' not in norm):
            stream.append(None);positions.append(None);continue
        remaining=max(0,40000-len(stream))
        if len(ts)>remaining:
            diagnostics['cutoffs']['tokens_omitted']+=len(ts)-remaining
            ts=ts[:remaining]
        if not ts:continue
        # Remove actual supplied prompt spans, even when embedded in a response.
        masked=set()
        normalized=unicodedata.normalize('NFKC',text)
        questions=[(m.start(),m.end()) for m in re.finditer(r'[^.!?]+[.!?]?',normalized) if m.group().endswith('?')] if '?' in normalized else []
        if questions:
            for ti,token in enumerate(_TOKEN.finditer(normalized)):
                if ti>=len(ts):break
                if any(start<=token.start()<end for start,end in questions):masked.add(ti)
        for i in range(max(0,len(ts)-7)):
            if tuple(ts[i:i+8]) in prompt_shingles:
                masked.update(range(i,i+8))
        if masked:
            diagnostics['prompt_tokens_excluded']+=len(masked)
        bid=len(blocks)
        tokens=[None if i in masked else t for i,t in enumerate(ts)]
        blocks.append({'text':text,'tokens':tokens,'location':f"Word §{p.get('index','?')}", 'lexical':_lexical(tokens)})
        for token in tokens:
            stream.append(token);positions.append(bid)
    return {'blocks':blocks,'tokens':stream,'positions':positions}


def _snippet(doc,start,length):
    bids=[b for b in doc['positions'][start:start+length] if b is not None]
    if not bids:return '', 'Word: posizione non risolta'
    unique=list(dict.fromkeys(bids))
    return ' / '.join(doc['blocks'][b]['text'] for b in unique)[:1200], ' → '.join(doc['blocks'][b]['location'] for b in unique)


def compare_submissions(reports, parsed):
    """Indexed literal/near text matches, exact archives and weak shared errors.

    No semantic paraphrase or AI attribution is inferred. Candidate thresholds and
    resource cutoffs are exposed via the list-compatible result.diagnostics and
    rendered into HTML. Known prompts are removed before indexing.
    """
    from difflib import SequenceMatcher
    result=ComparisonResults()
    diag={'pairs_total':len(reports)*(len(reports)-1)//2,'literal_candidate_pairs':0,'near_candidate_block_pairs':0,
          'prompt_tokens_excluded':0,'cutoffs':{'blocks_omitted':0,'tokens_omitted':0,'repeated_shingle_positions_omitted':0,'literal_alignments_omitted':0,'high_frequency_near_tokens_omitted':0,'near_candidates_omitted':0},
          'thresholds':{'literal_tokens':8,'literal_lexical_words':3,'document_max_blocks':2000,'document_max_tokens':40000,'shingle_positions_per_document':4,'near_min_tokens':20,'near_window_tokens':400,'near_window_overlap':80,'near_sequence_ratio':.72,'near_shared_lexical_words':4,'near_posting_limit':600,'literal_alignments_per_pair':100,'near_candidates_per_pair':200,'snippets_per_rule_pair':5}}
    result.diagnostics=diag
    excluded=set()
    for field in ('assignment_text','excluded_texts'):
        value=parsed.get(field,[]) if isinstance(parsed,dict) else []
        if isinstance(value,str):value=value.splitlines()
        excluded.update(_normal(t) for t in value if t)
    prompt_shingles={tuple(ts[i:i+8]) for text in excluded for ts in [_tokens(text)] for i in range(max(0,len(ts)-7))}
    docs={r['key']:_prepared((parsed.get(r['key'],{}) or {}).get('docx'),excluded,prompt_shingles,diag) for r in reports}
    bykey={r['key']:r for r in reports};order={r['key']:i for i,r in enumerate(reports)}
    def emit(pair,kind,evidence,limits):
        a,b=pair
        result.append({'students':[a,b],'labels':[bykey[a].get('label',a),bykey[b].get('label',b)],'kind':kind,'evidence':evidence[:5],'evidence_count':len(evidence),'limits':limits})
    # Index eight-token phrases across paragraph boundaries. Repetition positions
    # are bounded; omitted positions are disclosed, not silently certified absent.
    shingle_index=defaultdict(dict)
    for key,doc in docs.items():
        ts=doc['tokens']
        for i in range(max(0,len(ts)-7)):
            sh=tuple(ts[i:i+8])
            if None in sh or len(_lexical(sh))<3:continue
            positions=shingle_index[sh].setdefault(key,[])
            if len(positions)<4:positions.append(i)
            else:diag['cutoffs']['repeated_shingle_positions_omitted']+=1
    alignments=defaultdict(dict)
    for sh,members in shingle_index.items():
        if len(members)<2:continue
        for a,b in combinations(sorted(members,key=order.get),2):
            pair=(a,b)
            for x in members[a]:
                for y in members[b]:
                    delta=x-y
                    if delta in alignments[pair]:continue
                    if len(alignments[pair])<100:alignments[pair][delta]=(x,y,len(members))
                    else:diag['cutoffs']['literal_alignments_omitted']+=1
    diag['literal_candidate_pairs']=len(alignments)
    for pair,seeds in alignments.items():
        a,b=pair;at=docs[a]['tokens'];bt=docs[b]['tokens'];segments=[]
        for x,y,freq in seeds.values():
            anchor=' '.join(at[x:x+8])
            while x>0 and y>0 and at[x-1] is not None and at[x-1]==bt[y-1]:x-=1;y-=1
            end=0
            while x+end<len(at) and y+end<len(bt) and at[x+end] is not None and at[x+end]==bt[y+end]:end+=1
            if end<8 or len(_lexical(at[x:x+end]))<3:continue
            if any(x>=z[0] and x+end<=z[0]+z[2] for z in segments):continue
            segments.append((x,y,end,freq,anchor))
        segments.sort(key=lambda z:(-z[2],z[0],z[1]));evidence=[]
        for x,y,n,freq,anchor in segments:
            ls,ll=_snippet(docs[a],x,n);rs,rl=_snippet(docs[b],y,n)
            evidence.append({'component':'Sequenza letterale coincidente','left':ls,'right':rs,'location_left':ll,'location_right':rl,'matching_tokens':n,'matching_text':' '.join(at[x:x+n])[:1200], 'document_frequency':freq,'class_size':len(reports),'frequency_basis':'ancora letterale di 8 token','frequency_anchor':anchor})
        if evidence:
            identical=any(_normal(e['left'])==_normal(e['right']) for e in evidence)
            emit(pair,'testo_coincidente' if identical else 'testo_parziale',evidence,'Coincidenza letterale da esaminare nel contesto; la frequenza nella classe non è una probabilità di copiatura. Anche un passaggio condiviso dalla maggioranza resta visibile: può derivare dalla consegna, da fonti comuni o da scambi. Nessuna attribuzione di autore, direzione o origine AI. Sono mostrati al massimo cinque segmenti per coppia; il totale rilevato è conservato.')
    # Near-copy candidates use an inverted lexical index rather than all
    # paragraph pairs. Frequent words are retained for scoring after candidacy.
    blocks={};lex_index=defaultdict(list)
    for key,doc in docs.items():
        for i,block in enumerate(doc['blocks']):
            ts=block['tokens']
            # Bound the sequence alignment cost while retaining long answers
            # through overlapping windows instead of dropping whole paragraphs.
            for wi,start in enumerate(range(0,len(ts),320)):
                window=ts[start:start+400]
                lex=_lexical(window)
                if len(window)>=20 and None not in window and len(lex)>=4:
                    ident=(key,i,wi)
                    text=block['text'] if len(ts)<=400 else ' '.join(window)
                    location=block['location']+('' if len(ts)<=400 else f', token {start+1}–{start+len(window)}')
                    blocks[ident]={'tokens':window,'text':text,'location':location,'lexical':lex}
                    for token in lex:lex_index[token].append(ident)
    overlap=Counter()
    for token,posting in lex_index.items():
        if len(posting)>600:
            diag['cutoffs']['high_frequency_near_tokens_omitted']+=1;continue
        for left,right in combinations(posting,2):
            if left[0]==right[0]:continue
            if order[left[0]]>order[right[0]]:left,right=right,left
            overlap[(left,right)]+=1
    near_candidates=defaultdict(list)
    for (left,right),shared in overlap.items():
        if shared>=4:near_candidates[(left[0],right[0])].append((shared,left,right))
    for pair,candidates in near_candidates.items():
        candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
        if len(candidates)>200:diag['cutoffs']['near_candidates_omitted']+=len(candidates)-200
        evidence=[]
        for shared,left,right in candidates[:200]:
            diag['near_candidate_block_pairs']+=1
            a=blocks[left];b=blocks[right]
            if a['tokens']==b['tokens']:continue
            if min(len(a['tokens']),len(b['tokens']))/max(len(a['tokens']),len(b['tokens']))<.57:continue
            ratio=SequenceMatcher(None,a['tokens'],b['tokens'],autojunk=False).ratio()
            if ratio>=.72:
                evidence.append({'component':'Passaggi con modifiche di parole','left':a['text'][:1200],'right':b['text'][:1200],'location_left':a['location'],'location_right':b['location'],'token_sequence_ratio':round(ratio,4),'shared_lexical_words':shared})
        if evidence:emit(pair,'testo_simile',evidence,'Rapporto di sequenza dei token = 2 × token allineati / token totali; soglia 0,72 su almeno 20 token per passaggio. È una misura testuale, non probabilità di copiatura o origine AI. Risposte indipendenti, fonti comuni e consegna possono produrre somiglianze. Ricerca limitata ai candidati con almeno quattro parole lessicali condivise; omissioni e soglie sono dichiarate nella copertura.')
    # Whole-file evidence never reduces to matching prescribed data/options.
    for a,b in combinations(reports,2):
        pair=(a['key'],b['key']);left=parsed.get(pair[0],{}) or {};right=parsed.get(pair[1],{}) or {}
        ao=left.get('omv') or {};bo=right.get('omv') or {};h=ao.get('sha256')
        if h and h==bo.get('sha256'):
            emit(pair,'omv_identico',[{'component':'OMV completo','sha256':h,'observed':'File identici byte per byte, anche se rinominati.'}],'Identità del file completo: non determina autore, intenzione, direzione o numero di scambi. Dati comuni e risultati prescritti non sono indizi distintivi.')
        elif ao.get('content_sha256') and ao['content_sha256']==bo.get('content_sha256'):
            emit(pair,'omv_contenuto_identico',[{'component':'Tutte le componenti interne OMV','content_sha256':ao['content_sha256'],'observed':'Nomi e byte non compressi di tutte le componenti interne coincidono; archivi esterni eventualmente ricompressi o rinominati.'}],'Equivalenza del contenuto interno completo; non determina autore, intenzione, direzione o numero di scambi. La semplice identità dei dati e risultati prescritti non produce questo rilievo.')
    # Shared errors are explicitly weak. Generic missing/error-reader findings and
    # identical required metadata are never converted into integrity signals.
    error_index=defaultdict(dict)
    for r in reports:
        for f in r.get('findings',[]):
            ident=str(f.get('id',''));obs=str(f.get('observed',''))
            if f.get('status')!='Errato' or not re.search(r'\d',obs) or not (ident.startswith('WORD.NUM.') or ident.endswith('.values')):continue
            error_index[(ident,_normal(obs))][r['key']]=f
    for (ident,observed),members in error_index.items():
        # Common class-wide mistakes are correction information rather than a
        # distinctive cross-submission clue; textual matches remain unaffected.
        if len(members)>max(2,len(reports)//2):continue
        for a,b in combinations(sorted(members,key=order.get),2):
            fa=members[a];fb=members[b]
            emit((a,b),'errore_condiviso',[{'component':'Stesso rilievo numerico','requirement_id':ident,'left':fa['observed'],'right':fb['observed'],'location_left':fa.get('location',''),'location_right':fb.get('location',''),'document_frequency':len(members),'class_size':len(reports)}],'Segnale debole di cooccorrenza: lo stesso errore può essere comune o derivare dalla stessa difficoltà, da arrotondamenti o da una fonte comune. Non costituisce prova di copiatura, autore o direzione; confrontare spiegazioni e calcoli originali.')
    diag['pairs_flagged_by_rule']={kind:len({tuple(c['students']) for c in result if c['kind']==kind}) for kind in sorted({c['kind'] for c in result})}
    return result


def render_report(reports, comparisons, metadata=None):
    metadata=metadata or {}
    def e(value):
        if isinstance(value,(dict,list,tuple)):
            value=json.dumps(value,ensure_ascii=False,sort_keys=True,default=str)
        return escape(str(value if value is not None else ''),quote=True)
    out=['<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rapporto esercitazioni jamovi</title>',
         '<style>body{font:16px Georgia,serif;max-width:1500px;margin:2em auto;padding:0 1em;color:#111;background:white}table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #777;padding:.5em;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eee}a{color:#111}small{font-size:.85em}.evidence{white-space:pre-wrap}section{margin:3em 0}h1,h2,h3,h4{font-family:Arial,sans-serif}details{margin:1em 0}summary{cursor:pointer}tr{break-inside:avoid}@media print{body{max-width:none;font-size:10pt}section{break-before:page}details{display:block}a{text-decoration:none}}</style></head><body>',
         '<h1>Rapporto delle esercitazioni jamovi</h1>',
         '<p>Supporto alla revisione del tutor. Completezza, correttezza ed evidenze da approfondire restano distinte. Nessuna bocciatura automatica, percentuale di origine AI o certificazione di lavoro umano. La partecipazione alle attività non è deducibile dai file.</p>',
         '<p>Confronti OMV: identità del file completo e, quando disponibile, identità di tutte le componenti interne dopo decompressione. Annotazioni personali non esposte dal lettore non vengono confrontate separatamente. La coincidenza dei soli dati, valori, formule e ordine prescritti non viene segnalata come anomalia.</p>',
         '<p>Metadati della scansione: '+e(metadata)+'</p>', '<h2>Riepilogo e indice</h2>',
         '<table><thead><tr><th>Studente</th><th>Identificativo locale</th><th>Completezza</th><th>Correttezza</th><th>Da approfondire</th><th>Copertura</th></tr></thead><tbody>']
    for i,r in enumerate(reports,1):
        counts=Counter(f.get('status') for f in r.get('findings',[]))
        out.append(f'<tr><td><a href="#student-{i}">{e(r.get("label"))}</a></td><td>{e(r.get("key"))}</td><td>{counts["Mancante"]} mancanti</td><td>{counts["Errato"]} errori; {counts["Verificato"]} requisiti verificati</td><td>{counts["Da verificare"]} da verificare; {counts["Non verificabile"]} non verificabili</td><td>{len(r.get("findings",[]))} controlli; nessun voto automatico</td></tr>')
    out.append('</tbody></table>')
    diagnostics=getattr(comparisons,'diagnostics',{})
    rule_pairs={kind:len({tuple(c.get('students',[])) for c in comparisons if c.get('kind')==kind}) for kind in sorted({c.get('kind','') for c in comparisons})}
    out.append('<h2>Copertura dei confronti</h2><p>Coppie segnalate per regola (una coppia può comparire in più regole): '+e(rule_pairs)+'</p>')
    out.append('<p>Ricerca dei candidati e limiti effettivi: '+e(diagnostics or 'Diagnostica dei candidati non disponibile in questo rapporto ricostruito.')+'</p>')
    out.append('<p>La ricerca usa sequenze letterali di almeno 8 token e almeno 3 parole lessicali; i passaggi con modifiche richiedono almeno 20 token, quattro parole lessicali condivise e un rapporto di sequenza di almeno 0,72. Parole tecniche inevitabili, numeri e formule non bastano da soli. Un esito senza rilievi non certifica autonomia; parafrasi, testo molto breve e candidati omessi dai limiti dichiarati possono sfuggire. La frequenza nella classe resta visibile anche per coincidenze diffuse.</p>')
    for i,r in enumerate(reports,1):
        out.append(f'<section id="student-{i}"><h2>{e(r.get("label"))}</h2><p>Identificativo locale: {e(r.get("key"))}</p><p>File esaminati: {e(r.get("paths",{}))}</p><p>Copertura: {e(r.get("coverage",{}))}</p>')
        if r.get('metadata'):
            out.append('<h3>Metadati osservati</h3><p>Proprietà modificabili dei documenti e degli archivi; nessuna data del filesystem dopo estrazione viene usata per dedurre la cronologia.</p>')
            for kind,summary in r['metadata'].items():
                if not isinstance(summary,dict):continue
                out.append('<h4>'+e(kind.upper())+'</h4><table><thead><tr><th>Proprietà</th><th>Valore osservato</th><th>Fonte, fuso e precisione</th></tr></thead><tbody>')
                authors=summary.get('authors',[])
                for author in authors:
                    out.append('<tr><td>'+e(author.get('field','autore'))+'</td><td>'+e(author.get('value','sconosciuto'))+'</td><td>'+e(author.get('source','fonte non risolta'))+'</td></tr>')
                if not authors:out.append('<tr><td>Autore / ultima modifica</td><td>Sconosciuto: proprietà non disponibile</td><td>Non si deduce un autore dal nome del file.</td></tr>')
                for date_name,date in summary.get('dates',{}).items():
                    if not isinstance(date,dict):continue
                    out.append('<tr><td>'+e(date_name)+'</td><td>'+e(date.get('raw') or 'Sconosciuto')+'</td><td>'+e({'source':date.get('source'),'timezone':date.get('timezone','unknown'),'precision_seconds':date.get('precision_seconds'),'valid':date.get('valid',False)})+'</td></tr>')
                if summary.get('zip_entries'):
                    out.append('<tr><td>Orologi delle componenti interne ZIP</td><td>'+e(summary['zip_entries'])+'</td><td>ZIP entry; fuso sconosciuto; precisione 2 secondi. Non certificano il salvataggio.</td></tr>')
                out.append('<tr><td>Voce nello ZIP delle consegne</td><td>'+e(summary.get('source_zip_metadata') or 'Sconosciuta')+'</td><td>Metadato della voce esterna; distinto dagli orari interni e dalle proprietà Word.</td></tr></tbody></table><p><strong>Limiti:</strong> '+e(summary.get('limits','Metadati modificabili, non autenticati; non provano paternità o direzione degli scambi.'))+'</p>')
        linguistic=r.get('linguistic')
        if isinstance(linguistic,dict):
            out.append('<h3>Analisi linguistica — proposte per il tutor</h3><p>Stato: '+e(linguistic.get('status','non disponibile'))+'; provider: '+e(linguistic.get('provider','sconosciuto'))+'; modello: '+e(linguistic.get('model','sconosciuto'))+'; rubrica: '+e(linguistic.get('rubric_version','non specificata'))+'. Ogni rilievo di questa sezione è una proposta da valutare e non modifica automaticamente gli esiti dei controlli deterministici.</p>')
            proposals=linguistic.get('findings',[])
            if proposals:
                out.append('<table><thead><tr><th>Proposta del modello</th><th>Evidenze e posizione</th></tr></thead><tbody>')
                for finding in proposals:
                    if isinstance(finding,dict):
                        title=finding.get('requirement',finding.get('title',finding.get('category','Rilievo da verificare')))
                        out.append('<tr><td>'+e(title)+'</td><td>'+e(finding)+'</td></tr>')
                    else:out.append('<tr><td>Rilievo da verificare</td><td>'+e(finding)+'</td></tr>')
                out.append('</tbody></table>')
            else:out.append('<p>Nessuna proposta disponibile. Questo esito non certifica correttezza, autonomia o origine del testo.</p>')
            out.append('<p><strong>Limiti dell’analisi linguistica:</strong> '+e(linguistic.get('limits','Valutazione del tutor necessaria; nessuna probabilità di origine AI o giudizio di colpevolezza.'))+'</p>')
        out.append('<h3>Riepilogo per fonte</h3><ul>')
        for source in ('OMV','Word','OMV–Word'):
            counts=Counter(f['status'] for f in r.get('findings',[]) if f.get('source')==source)
            out.append('<li>'+e(source)+': '+e(dict(counts) if counts else 'Nessun confronto deterministico eseguibile.')+'</li>')
        out.append('</ul>')
        for part in range(8):
            fs=[f for f in r.get('findings',[]) if f.get('part',0)==part]
            out.append('<h3>'+('File, inventario e formato' if part==0 else 'Parte '+str(part))+'</h3>')
            if not fs:
                out.append('<p>Nessun controllo disponibile per questa parte: non verificabile.</p>');continue
            out.append('<table><thead><tr><th>Requisito e fonte</th><th>Esito</th><th>Evidenza osservata</th><th>Atteso / azione del tutor</th><th>Posizione</th></tr></thead><tbody>')
            for f in fs:
                out.append('<tr><td>'+e(f.get('id',''))+'<br>'+e(f.get('requirement',''))+'<br><small>'+e(f.get('source',''))+'</small></td><td>'+e(f.get('status','Non verificabile'))+'</td><td class="evidence">'+e(f.get('observed',''))+'</td><td>'+e(f.get('expected',''))+'</td><td>'+e(f.get('location',''))+'</td></tr>')
            out.append('</tbody></table>')
        relevant=[c for c in comparisons if r.get('key') in c.get('students',[])]
        out.append('<h3>Evidenze da approfondire con altre consegne</h3>')
        if not relevant:
            out.append('<p>Nessuna evidenza rilevata dai confronti implementati. Questo esito non certifica autonomia o paternità.</p>')
        for c in relevant:
            out.append('<h4>'+e(c.get('kind','Confronto'))+' — '+e(c.get('labels',c.get('students',[])))+'</h4>')
            if c.get('corroborates'):
                out.append('<p>Elemento accessorio associato ai rilievi di contenuto: '+e(c['corroborates'])+'. Non costituisce da solo un’attribuzione di paternità.</p>')
            if c.get('evidence_count',0)>len(c.get('evidence',[])):
                out.append('<p>Mostrate '+e(len(c.get('evidence',[])))+' di '+e(c['evidence_count'])+' evidenze rilevate per questa regola e coppia.</p>')
            for ev in c.get('evidence',[]):
                if isinstance(ev,dict) and 'left' in ev:
                    out.append('<table><tr><th>'+e(c.get('labels',['A','B'])[0])+' — '+e(ev.get('location_left'))+'</th><th>'+e(c.get('labels',['A','B'])[1])+' — '+e(ev.get('location_right'))+'</th></tr><tr><td>'+e(ev['left'])+'</td><td>'+e(ev['right'])+'</td></tr></table>')
                    if 'document_frequency' in ev:
                        basis='Ancora del passaggio (8 token)' if ev.get('frequency_basis') else 'Passaggio'
                        out.append('<p>'+basis+' presente in '+e(ev['document_frequency'])+' di '+e(ev['class_size'])+' documenti esaminati.'+(' Ancora: '+e(ev.get('frequency_anchor',''))+'.' if ev.get('frequency_basis') else '')+'</p>')
                    if 'matching_tokens' in ev:
                        out.append('<p>Token consecutivi coincidenti: '+e(ev['matching_tokens'])+'. Sequenza: '+e(ev.get('matching_text',''))+'</p>')
                    if 'token_sequence_ratio' in ev:
                        out.append('<p>Rapporto di sequenza dei token: '+e(ev['token_sequence_ratio'])+'; misura testuale, non probabilità di copiatura.</p>')
                    extra={k:v for k,v in ev.items() if k not in {'component','left','right','location_left','location_right','document_frequency','class_size','matching_tokens','matching_text','frequency_basis','frequency_anchor','token_sequence_ratio'}}
                    if extra:out.append('<p>Dettagli dell’evidenza: '+e(extra)+'</p>')
                else:
                    out.append('<p>'+e(ev)+'</p>')
            out.append('<p><strong>Limiti:</strong> '+e(c.get('limits',''))+'</p>')
        counts=Counter(f.get('status') for f in r.get('findings',[]))
        out.append('<h3>Quadro finale per il tutor</h3><p>Completezza: '+e(counts['Mancante'])+' requisiti mancanti. Correttezza: '+e(counts['Errato'])+' contraddizioni riscontrate. Evidenze da approfondire: '+e(len(relevant))+' confronti; '+e(counts['Da verificare'])+' controlli demandati al tutor. I requisiti non verificabili sono '+e(counts['Non verificabile'])+' e non costituiscono giudizi negativi sullo studente.</p></section>')
    out.append('</body></html>')
    return ''.join(out)
