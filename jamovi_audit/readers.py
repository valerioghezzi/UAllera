"""Offline OMV (jamovi 2.7.4 archive 11) and DOCX readers; never execute content."""
import hashlib
import json
import math
import re
import struct
import zlib
import xml.etree.ElementTree as ET
from .archive import safe_zip
from .protobuf import fields, val, string, signed, MAX_DEPTH


def _error(f,n):
    b=val(f,n)
    if b is None:return ''
    e=fields(b)
    return ': '.join(s for s in (string(e,1),string(e,2)) if s)


def _options(data,depth=0):
    f=fields(data,depth);values=[]
    for _,b in f.get(1,[]):
        o=fields(b,depth+1)
        if 1 in o:v=signed(val(o,1))
        elif 2 in o:v=val(o,2)
        elif 3 in o:v=string(o,3)
        elif 4 in o:v={0:False,1:True,2:None}.get(val(o,4))
        elif 5 in o:v=_options(val(o,5),depth+1)
        else:v=None
        values.append(v)
    if val(f,2,0):
        names=[b.decode('utf-8') for _,b in f.get(3,[])]
        if len(names)!=len(values) or len(set(names))!=len(names):raise ValueError('Nomi opzioni protobuf ambigui')
        return dict(zip(names,values))
    return values


def _cell(data):
    f=fields(data)
    if 1 in f:return signed(val(f,1))
    if 2 in f:
        v=val(f,2);return v if math.isfinite(v) else None
    if 3 in f:return string(f,3)
    return None


def _image_validity(content, path):
    """Validate PNG chunks, checksums and bounded scanline framing, offline."""
    signature=b'\x89PNG\r\n\x1a\n'
    if not content.startswith(signature):
        if path.lower().endswith('.png'):
            return False,'Firma PNG assente o non valida'
        return None,'Formato immagine non supportato per la verifica'
    pos=8;header=None;idat=[];ended=False;palette=False
    try:
        while pos<len(content):
            if pos+12>len(content):raise ValueError('Chunk PNG troncato')
            length=struct.unpack_from('>I',content,pos)[0]
            end=pos+12+length
            if end>len(content):raise ValueError('Lunghezza chunk PNG oltre la risorsa')
            kind=content[pos+4:pos+8];payload=content[pos+8:pos+8+length]
            crc=struct.unpack_from('>I',content,pos+8+length)[0]
            if zlib.crc32(kind+payload)&0xffffffff!=crc:raise ValueError('CRC PNG non valido')
            if header is None and kind!=b'IHDR':raise ValueError('IHDR PNG non iniziale')
            if kind==b'IHDR':
                if header is not None or length!=13:raise ValueError('IHDR PNG duplicato o non valido')
                header=struct.unpack('>IIBBBBB',payload)
                width,height,depth,colour,compression,filtering,interlace=header
                permitted={0:(1,2,4,8,16),2:(8,16),3:(1,2,4,8),4:(8,16),6:(8,16)}
                if not (0<width<2**31 and 0<height<2**31):raise ValueError('Dimensioni PNG non valide')
                if colour not in permitted or depth not in permitted[colour] or compression or filtering or interlace not in (0,1):raise ValueError('Parametri IHDR PNG non validi')
            elif kind==b'PLTE':
                if idat or not length or length>768 or length%3:raise ValueError('Palette PNG non valida')
                palette=True
            elif kind==b'IDAT':idat.append(payload)
            elif kind==b'IEND':
                if length or not idat or not any(idat):raise ValueError('IEND o IDAT PNG non valido')
                if end!=len(content):raise ValueError('Dati dopo IEND PNG')
                ended=True;break
            elif kind[:1].isupper():return None,'Chunk PNG critico non supportato'
            pos=end
        if header is None or not ended:raise ValueError('IHDR, IDAT o IEND PNG mancante')
        width,height,depth,colour,_,_,interlace=header
        if colour==3 and not palette:raise ValueError('Palette PNG indicizzata assente')
        channels={0:1,2:3,3:1,4:2,6:4}[colour]
        passes=[(0,0,1,1)] if not interlace else [(0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2)]
        rows=[];expected=0
        for x,y,dx,dy in passes:
            pw=max(0,(width-x+dx-1)//dx);ph=max(0,(height-y+dy-1)//dy)
            if pw and ph:
                stride=(pw*channels*depth+7)//8+1;rows.append((stride,ph));expected+=stride*ph
        if expected>64*1024*1024:return None,'PNG oltre il limite di verifica delle righe'
        dec=zlib.decompressobj();raw=dec.decompress(b''.join(idat),expected+1)
        if len(raw)!=expected or not dec.eof or dec.unused_data or dec.unconsumed_tail:raise ValueError('Dati compressi o righe PNG non validi')
        offset=0
        for stride,count in rows:
            for row in range(count):
                if raw[offset+row*stride]>4:raise ValueError('Filtro riga PNG non valido')
            offset+=stride*count
        return True,''
    except (ValueError,struct.error,zlib.error) as e:return False,str(e)


def _flatten(data,entries,parent,depth=0):
    f=fields(data,depth)
    name=string(f,1)
    path=parent['path']+[name or '?']
    visibility=val(f,15,0)
    visible=parent['visible'] and visibility in (0,2)
    own_status=val(f,3,0)
    status=parent['status'] if parent['status'] in (1,2,4,5) else (own_status or parent['status'])
    err=parent['error'] or _error(f,4)
    stale=parent['stale'] or bool(val(f,5,0))
    base={'path':path,'name':name,'title':string(f,2),'status':status,'visible':visible,'stale':stale,'error':err}
    for number,kind in ((6,'table'),(7,'image'),(8,'group'),(9,'array'),(10,'preformatted'),(11,'syntax'),(12,'html'),(13,'outputs'),(24,'notice')):
        if number not in f:continue
        b=val(f,number)
        if kind in ('group','array'):
            node=fields(b,depth+1);res=[]
            if kind=='array' and val(node,8,0) and 9 in node:
                res.extend(_flatten(val(node,9),entries,base,depth+1))
            for _,child in node.get(1,[]):res.extend(_flatten(child,entries,base,depth+1))
            return res
        # jamovi 2.7.4 Table/Image inherit stale=True without updating it.
        # Its own client ignores that field for these types. Preserve the raw
        # flag, but do not misrepresent it as a freshness verdict.
        out=dict(base,kind=kind,stale_applicable=kind not in ('table','image'))
        if kind=='table':
            node=fields(b,depth+1);cols=[]
            for _,cb in node.get(1,[]):
                c=fields(cb,depth+2)
                cols.append({'name':string(c,1),'title':string(c,2),'visible':visible and val(c,15,0) in (0,2),'cells':[_cell(x) for _,x in c.get(7,[])],'type':string(c,3),'format':string(c,4)})
            out['columns']=cols
            out['row_names']=[x.decode('utf-8') for _,x in node.get(2,[])]
        elif kind=='image':
            image=fields(b,depth+1);path_=string(image,1)
            exists=bool(path_ and path_ in entries)
            valid,issue=_image_validity(entries[path_],path_) if exists else (False,'Risorsa immagine assente')
            out.update(image_path=path_,image_exists=exists,image_valid=valid,image_issue=issue,width=val(image,2,0),height=val(image,3,0))
        elif kind in ('preformatted','syntax'):out['text']=b.decode('utf-8')
        elif kind=='html':out['text']=string(fields(b,depth+1),1)
        elif kind=='notice':
            node=fields(b,depth+1);out['text']=string(node,1);out['notice_type']=val(node,2,0)
            if out['notice_type']==0:out['error']=out['text'] or 'Errore nel risultato'
        return [out]
    return []


def _analysis(data,path,entries):
    f=fields(data)
    a={'id':val(f,2,0),'name':string(f,3),'ns':string(f,4),'title':string(f,18),'status':val(f,8,0),'options':_options(val(f,6,b'')),'results':[],'error':_error(f,9),'source_path':path,'revision':val(f,11,0)}
    if 7 in f:
        a['results']=_flatten(val(f,7),entries,{'path':[],'status':a['status'],'visible':True,'stale':False,'error':a['error']})
    return a


def read_omv(data: bytes) -> dict:
    result={'metadata':{},'columns':{},'analyses':[],'errors':[],'manifest':{},'sha256':hashlib.sha256(data).hexdigest()}
    try:
        es=safe_zip(data,'OMV')
        content_hash=hashlib.sha256()
        for entry_name in sorted(es):
            encoded_name=entry_name.encode('utf-8')
            content_hash.update(struct.pack('<Q',len(encoded_name)))
            content_hash.update(encoded_name)
            content_hash.update(struct.pack('<Q',len(es[entry_name])))
            content_hash.update(es[entry_name])
        result['content_sha256']=content_hash.hexdigest()
        raw=es.get('meta',es.get('META-INF/MANIFEST.MF',b'')).decode('utf-8')
        result['manifest']={k.strip():v.strip() for line in raw.splitlines() if ':' in line for k,v in [line.split(':',1)]}
        if result['manifest'].get('jamovi-Archive-Version')!='11.0':
            result['errors'].append('Versione archivio OMV non collaudata: '+result['manifest'].get('jamovi-Archive-Version','assente'))
        meta=json.loads(es['metadata.json']);result['metadata']=meta
        dataset=meta['dataSet'];rows=dataset['rowCount'];fs=dataset['fields']
        if not isinstance(rows,int) or not 0<=rows<=2000000 or len(fs)>10000:raise ValueError('Limite dati OMV superato')
        xdata=json.loads(es.get('xdata.json',b'{}'));binary=es['data.bin'];strings=es.get('strings.bin');pos=0
        for field in fs:
            name=field['name']
            if name in result['columns']:raise ValueError('Nome colonna duplicato: '+name)
            dt=field.get('dataType')
            if dt not in (None,'Integer','Decimal','Text'):raise ValueError('Tipo dati colonna non supportato: '+str(dt))
            decimal=dt=='Decimal' or (dt is None and field.get('measureType')=='Continuous')
            fmt,width=('<d',8) if decimal else ('<i',4)
            size=width*rows
            if pos+size>len(binary):raise ValueError('Dati binari OMV troncati: '+name)
            values=[v[0] for v in struct.iter_unpack(fmt,binary[pos:pos+size])];pos+=size
            out=[]
            for v in values:
                if (decimal and not math.isfinite(v)) or (not decimal and v==-2147483648):out.append(None)
                elif dt=='Text' and field.get('measureType')=='ID':
                    if strings is None:raise ValueError('strings.bin assente per colonna testo ID')
                    if v<0 or v>=len(strings):raise ValueError('Offset strings.bin non valido')
                    end=strings.find(b'\x00',v)
                    if end<0:raise ValueError('Testo strings.bin non terminato')
                    out.append(strings[v:end].decode('utf-8'))
                else:out.append(v)
            result['columns'][name]={'meta':field,'data':out,'levels':xdata.get(name,{}).get('labels',[])}
        if pos!=len(binary):result['errors'].append('Dimensione data.bin inattesa: byte residui')
        for path in sorted(es):
            if re.match(r'^\d{2,} .+/analysis$',path):
                try:result['analyses'].append(_analysis(es[path],path,es))
                except (ValueError,TypeError,UnicodeError,struct.error) as e:result['errors'].append(f'Analisi non leggibile {path}: {e}')
        result['analyses'].sort(key=lambda a:(a['id'],a['source_path']))
    except (ValueError,KeyError,TypeError,UnicodeError,struct.error,AttributeError) as e:
        result['errors'].append('OMV non leggibile: '+str(e))
    return result


W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def _xml(data):
    if len(data)>16*1024*1024:raise ValueError('XML troppo grande')
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():raise ValueError('DTD XML non consentito')
    return ET.fromstring(data)

def _text(node):
    text=[]
    for e in node.iter():
        if e.tag==W+'t':text.append(e.text or '')
        elif e.tag==W+'tab':text.append('\t')
        elif e.tag in (W+'br',W+'cr'):text.append('\n')
    return ''.join(text)

def _props(node):
    out={}
    if node is None:return out
    for e in node.iter():
        if e.tag==W+'rFonts':
            font=e.get(W+'ascii') or e.get(W+'hAnsi')
            if font:out['font']=font
        elif e.tag==W+'sz':
            try:out['size']=int(e.get(W+'val'))/2
            except (ValueError,TypeError):pass
        elif e.tag==W+'spacing':
            try:out['line']=int(e.get(W+'line'));out['line_rule']=e.get(W+'lineRule','auto')
            except (ValueError,TypeError):pass
    return out


def _docx_evidence(body, parts):
    """Saved body paragraph text in XML order, independent of legacy summaries.

    IDs address zero-based XML element child paths within this document; callers
    must also bind them to the document hash. Drawing/textbox content is omitted,
    never folded into its surrounding paragraph. No normalization or sampling is
    applied. The manifest does not claim coverage of Word's rendered pages.
    """
    sources, omissions, counts = [], [], {}
    part = 'word/document.xml'
    def omit(kind, location, reason, source_part=part):
        omissions.append(dict(kind=kind, part=source_part, location=location, reason=reason))
    nonbody = {}
    if '[Content_Types].xml' in parts:
        for declaration in _xml(parts['[Content_Types].xml']):
            match = re.fullmatch(r'application/vnd\.openxmlformats-officedocument\.wordprocessingml\.(header|footer|footnotes|endnotes|comments)\+xml',
                                 declaration.get('ContentType', ''))
            name = declaration.get('PartName', '').lstrip('/')
            if match and name in parts:
                nonbody[name] = match.group(1)
    for name in sorted(parts):
        match = re.fullmatch(r'word/(header[^/]*|footer[^/]*|footnotes|endnotes|comments)\.xml', name)
        if match:
            value = match.group(1)
            nonbody.setdefault(name, 'header' if value.startswith('header') else
                               'footer' if value.startswith('footer') else value)
    for name, kind in sorted(nonbody.items()):
        omit(kind, name, 'Parte esterna al corpo Word: testo non esaminato.', name)
    def number(context, kind):
        key = (context, kind)
        counts[key] = counts.get(key, 0) + 1
        return counts[key]
    excluded = {'drawing', 'pict', 'object', 'txbxContent', 'altChunk', 'subDoc', 'del', 'moveFrom'}
    inline = {'r', 'hyperlink', 'sdt', 'sdtContent', 'customXml', 'smartTag',
              'ins', 'moveTo', 'dir', 'bdo', 'fldSimple'}
    markers = {'pPr', 'rPr', 'sdtPr', 'customXmlPr', 'smartTagPr', 'bookmarkStart',
               'bookmarkEnd', 'proofErr', 'permStart', 'permEnd', 'commentRangeStart',
               'commentRangeEnd', 'lastRenderedPageBreak', 'fldChar'}
    def paragraph_text(node, path):
        text, pending = [], [(node, path)]
        while pending:
            current, current_path = pending.pop()
            local = current.tag.removeprefix(W)
            if current is node or local in inline:
                pending.extend((child, current_path + '/' + str(i))
                               for i, child in reversed(list(enumerate(current))))
            elif local == 't':
                text.append(current.text or '')
            elif local == 'tab':
                text.append('\t')
            elif local in ('br', 'cr'):
                text.append('\n')
            elif local not in markers:
                omit(local.rsplit('}', 1)[-1], current_path,
                     'Contenuto non supportato escluso dal testo citabile del paragrafo.')
        return ''.join(text)
    structural = {'sdt', 'sdtContent', 'customXml'}
    properties = markers | {'sectPr', 'tblPr', 'tblGrid', 'trPr', 'tcPr'}
    pending = [(child, 'word/document.xml#body/' + str(i), '')
               for i, child in reversed(list(enumerate(body)))]
    while pending:
        node, path, context = pending.pop()
        local = node.tag.removeprefix(W)
        if local == 'p':
            index = number(context, 'p')
            location = (context + f', paragrafo {index}') if context else f'Paragrafo {index}'
            sources.append(dict(id=path, text=paragraph_text(node, path),
                                location=location, kind='paragraph'))
            continue
        if local in excluded:
            omit(local, path, 'Contenuto non supportato escluso dal corpo citabile.')
            continue
        if local == 'tbl':
            index = number(context, 'tbl')
            context = (context + f', tabella annidata {index}') if context else f'Tabella {index}'
        elif local == 'tr':
            context += f', riga {number(context, "tr")}'
        elif local == 'tc':
            context += f', cella {number(context, "tc")}'
        elif local in properties:
            continue
        elif local not in structural:
            omit(local.rsplit('}', 1)[-1], path, 'Elemento del corpo Word non supportato: testo non esaminato.')
            continue
        pending.extend((child, path + '/' + str(i), context)
                       for i, child in reversed(list(enumerate(node))))
    return sources, dict(scope='body_paragraphs_and_table_paragraphs_in_document_order', omissions=omissions)


def read_docx(data: bytes) -> dict:
    result={'paragraphs':[],'tables':[],'text':'','formatting':{},'metadata':{},'errors':[],'sha256':hashlib.sha256(data).hexdigest(),
            'evidence_sources':[], 'evidence_coverage':{'scope':'unavailable','omissions':[]}}
    try:
        es=safe_zip(data,'DOCX');root=_xml(es['word/document.xml']);styles={};defaults={};default_style=None
        if 'word/styles.xml' in es:
            style_root=_xml(es['word/styles.xml']);defaults=_props(style_root.find(W+'docDefaults'))
            for s in style_root.findall(W+'style'):
                sid=s.get(W+'styleId');styles[sid]=s
                if s.get(W+'type')=='paragraph' and s.get(W+'default') in ('1','true'):default_style=sid
        def style_format(sid,seen=None):
            seen=set() if seen is None else seen
            if sid in seen or len(seen)>32:return {},False
            if sid not in styles:return {},sid is None
            seen.add(sid);s=styles[sid];base=s.find(W+'basedOn');fmt={};resolved=True
            if base is not None:fmt,resolved=style_format(base.get(W+'val'),seen)
            fmt.update(_props(s.find(W+'pPr')));fmt.update(_props(s.find(W+'rPr')))
            return fmt,resolved
        body=root.find(W+'body')
        if body is None:raise ValueError('Corpo Word assente')
        result['evidence_sources'],result['evidence_coverage']=_docx_evidence(body,es)
        chunks=[];runs=[]
        def paragraph(p,index,location):
            pp=p.find(W+'pPr');ps=pp.find(W+'pStyle') if pp is not None else None
            sid=ps.get(W+'val') if ps is not None else default_style
            sf,resolved=style_format(sid);fmt=dict(defaults);fmt.update(sf);fmt.update(_props(pp))
            rr=[]
            for r in p.findall('.//'+W+'r'):
                if not _text(r):continue
                rf=dict(fmt);rp=r.find(W+'rPr');rs=rp.find(W+'rStyle') if rp is not None else None
                if rs is not None:
                    extra,ok=style_format(rs.get(W+'val'));rf.update(extra);resolved=resolved and ok
                rf.update(_props(rp));rr.append(rf);runs.append(rf)
            return {'index':index,'text':_text(p),'location':location,'style':sid,'format':fmt,'runs':rr,'style_resolved':resolved}
        pindex=0
        for elem in body:
            if elem.tag==W+'p':
                item=paragraph(elem,pindex,f'Paragrafo {pindex+1}');pindex+=1;result['paragraphs'].append(item);chunks.append(item['text'])
            elif elem.tag==W+'tbl':
                ti=len(result['tables']);rows=[];cells=[]
                for ri,tr in enumerate(elem.findall(W+'tr')):
                    row=[]
                    for ci,tc in enumerate(tr.findall(W+'tc')):
                        paras=[paragraph(p,pi,f'Tabella {ti+1}, riga {ri+1}, cella {ci+1}, paragrafo {pi+1}') for pi,p in enumerate(tc.findall(W+'p'))]
                        text='\n'.join(p['text'] for p in paras);row.append(text);cells.append({'row':ri,'column':ci,'text':text,'paragraphs':paras})
                    rows.append(row)
                result['tables'].append({'index':ti,'rows':rows,'cells':cells});chunks.extend('\t'.join(row) for row in rows)
        result['text']='\n'.join(chunks)
        issues=[];unknown=[]
        for i,r in enumerate(runs):
            for key,want in (('font','Times New Roman'),('size',12),('line',480)):
                if key not in r:unknown.append(f'Run {i+1}: {key} non risolto')
                elif r[key]!=want or (key=='line' and r.get('line_rule','auto')!='auto'):issues.append(f'Run {i+1}: {key}={r[key]}')
        result['formatting']={'issues':issues,'unknown':unknown,'runs_checked':len(runs),'complete':bool(runs) and not unknown,'fonts':sorted({r['font'] for r in runs if 'font' in r}),'sizes':sorted({r['size'] for r in runs if 'size' in r})}
        if 'docProps/core.xml' in es:
            result['metadata']={e.tag.rsplit('}',1)[-1]:e.text or '' for e in _xml(es['docProps/core.xml'])}
    except (ValueError,KeyError,TypeError,UnicodeError,ET.ParseError,AttributeError) as e:
        result['errors'].append('DOCX non leggibile: '+str(e))
        result['evidence_sources']=[]
        result['evidence_coverage']={'scope':'unavailable','omissions':[]}
    return result
