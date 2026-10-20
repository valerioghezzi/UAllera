"""Stdlib tokenizer for the pinned official Minerva Hugging Face graph.

encode(prompt, tokenizer_path) preserves exact rendered text: postprocessing is
explicitly disabled, matching tokenizers.encode(..., add_special_tokens=False).
Use OfficialTokenizer(path) for repeated calls. No prompt/token cache, external
service, optional package, or model weights are used.

The supported graph is NFKC9 -> ByteLevel(GPT-2 regex, per-span prefix space)
-> BPE. Raw, non-normalized added tokens are extracted before normalization.
Unicode normalization/property tables are pinned below, independent of Python's
Unicode version. Unsupported graphs and changed tokenizer assets fail closed.
"""
import base64
from bisect import bisect_right
import hashlib
import heapq
import json
from pathlib import Path
import re
import zlib

TOKENIZER_SIZE = 3667107
TOKENIZER_SHA256 = '0d8b77024a99bb31ba2b4fb6e0efed0910335ad77ac2d922ca09ed0a05f65f4d'
TOKENIZER_SOURCE_URL = ('https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0/'
    'resolve/d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d/tokenizer.json')
ORACLE_VERSION = 'tokenizers 0.22.1'
NORMALIZATION_UNICODE_VERSION = '9.0.0'

_SPECIAL_IDS = {'<s>': 0, '</s>': 1, '<unk>': 2, '<|start_header_id|>': 51200,
                '<|end_header_id|>': 51201, '<|eot_id|>': 51202}
_ADDED_TOKENS = [dict(id=value, content=content, single_word=False, lstrip=False,
    rstrip=False, normalized=False, special=True) for content, value in _SPECIAL_IDS.items()]
_BYTELEVEL = dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True)
_NORMALIZER = {'type': 'Sequence', 'normalizers': [{'type': 'NFKC'}]}
_POST_PROCESSOR = dict(type='TemplateProcessing',
    single=[{'SpecialToken': {'id': '<s>', 'type_id': 0}}, {'Sequence': {'id': 'A', 'type_id': 0}}],
    pair=[{'SpecialToken': {'id': '<s>', 'type_id': 0}}, {'Sequence': {'id': 'A', 'type_id': 0}},
          {'SpecialToken': {'id': '<s>', 'type_id': 1}}, {'Sequence': {'id': 'B', 'type_id': 1}}],
    special_tokens={'<s>': {'id': '<s>', 'ids': [0], 'tokens': ['<s>']}})
_MODEL_SETTINGS = dict(type='BPE', dropout=None, unk_token=None, continuing_subword_prefix=None,
    end_of_word_suffix=None, fuse_unk=False, byte_fallback=True, ignore_merges=False)
_SPECIAL_PATTERN = re.compile('|'.join(re.escape(value) for value in sorted(_SPECIAL_IDS, key=len, reverse=True)))
_WHITESPACE = frozenset(range(9, 14)) | {0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | frozenset(range(0x2000, 0x200B))
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _byte_encoder():
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    values = dict(zip(printable, printable))
    next_codepoint = 256
    for value in range(256):
        if value not in values:
            values[value] = next_codepoint
            next_codepoint += 1
    return tuple(chr(values[value]) for value in range(256))


_BYTE_ENCODER = _byte_encoder()


def _compose(left, right):
    # Unicode's algorithmic Hangul composition is outside the lookup tables.
    if 0x1100 <= left < 0x1113 and 0x1161 <= right < 0x1176:
        return 0xAC00 + ((left - 0x1100) * 21 + right - 0x1161) * 28
    if 0xAC00 <= left < 0xD7A4 and (left - 0xAC00) % 28 == 0 and 0x11A8 <= right < 0x11C3:
        return left + right - 0x11A7
    return _COMPOSE.get((left, right))


def _nfkc9(text):
    """Unicode9 compatibility decomposition, canonical ordering, recomposition."""
    ordered = []
    marks = []
    def flush():
        if marks:
            marks.sort(key=lambda value: _CCC.get(value, 0))  # Stable for equal CCC.
            ordered.extend(marks)
            marks.clear()
    for character in text:
        point = ord(character)
        if 0xAC00 <= point < 0xD7A4:
            offset = point - 0xAC00
            values = [0x1100 + offset // 588, 0x1161 + offset % 588 // 28]
            if offset % 28:
                values.append(0x11A7 + offset % 28)
        else:
            values = _DECOMPOSE.get(point, (point,))
        for value in values:
            if _CCC.get(value, 0):
                marks.append(value)
            else:
                flush()
                ordered.append(value)
    flush()
    result = []
    starter_index = None
    starter = None
    previous_class = 0
    for point in ordered:
        combining_class = _CCC.get(point, 0)
        combined = None
        if starter is not None and (previous_class == 0 or previous_class < combining_class):
            combined = _compose(starter, point)
        if combined is not None:
            result[starter_index] = combined
            starter = combined
        else:
            if combining_class == 0:
                starter_index = len(result)
                starter = point
            result.append(point)
            previous_class = combining_class
    return ''.join(chr(value) for value in result)


def _in_ranges(point, ranges, starts):
    index = bisect_right(starts, point) - 1
    return index >= 0 and point <= ranges[index][1]


def _category(character):
    point = ord(character)
    if point in _WHITESPACE:
        return 'W'
    if _in_ranges(point, _LETTER_RANGES, _LETTER_STARTS):
        return 'L'
    if _in_ranges(point, _NUMBER_RANGES, _NUMBER_STARTS):
        return 'N'
    return 'O'


def _pieces(text):
    """The exact ordered GPT-2 regex, hand-scanned without Unicode regex extensions."""
    index = 0
    length = len(text)
    while index < length:
        contraction = next((value for value in _CONTRACTIONS if text.startswith(value, index)), None)
        if contraction is not None:
            yield contraction
            index += len(contraction)
            continue
        start = index
        body = index + 1 if text[index] == ' ' else index
        matched = False
        for category in ('L', 'N', 'O'):
            if body < length and _category(text[body]) == category:
                end = body + 1
                while end < length and _category(text[end]) == category:
                    end += 1
                yield text[start:end]
                index = end
                matched = True
                break
        if matched:
            continue
        end = index + 1
        while end < length and _category(text[end]) == 'W':
            end += 1
        # \s+(?!\S) greedily leaves the final whitespace for the next branch
        # when a non-whitespace follows; the fallback \s+ handles a single one.
        if end < length and end - index > 1:
            end -= 1
        yield text[index:end]
        index = end


def _validate_graph(graph):
    expected = dict(version='1.0', truncation=None, padding=None, added_tokens=_ADDED_TOKENS,
        normalizer=_NORMALIZER, pre_tokenizer=_BYTELEVEL, post_processor=_POST_PROCESSOR,
        decoder=_BYTELEVEL)
    if not isinstance(graph, dict) or set(graph) != set(expected) | {'model'}:
        raise ValueError('Grafo tokenizer Minerva non supportato.')
    if any(graph.get(key) != value for key, value in expected.items()):
        raise ValueError('Grafo tokenizer Minerva non supportato.')
    model = graph['model']
    if not isinstance(model, dict) or set(model) != set(_MODEL_SETTINGS) | {'vocab', 'merges'}:
        raise ValueError('Grafo BPE Minerva non supportato.')
    if any(model.get(key) != value for key, value in _MODEL_SETTINGS.items()):
        raise ValueError('Grafo BPE Minerva non supportato.')
    vocab = model['vocab']
    merges = model['merges']
    if (not isinstance(vocab, dict) or not vocab or not isinstance(merges, list) or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0 or
                not isinstance(token, str) or not token for token, value in vocab.items()) or
            len(set(vocab.values())) != len(vocab)):
        raise ValueError('Vocabolario BPE Minerva non supportato.')
    for token, value in _SPECIAL_IDS.items():
        if token in vocab and vocab[token] != value:
            raise ValueError('ID speciali Minerva non supportati.')
    ranks = {}
    for rank, pair in enumerate(merges):
        if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(part, str) for part in pair) or
                any(part not in vocab for part in pair) or ''.join(pair) not in vocab):
            raise ValueError('Merge BPE Minerva non supportato.')
        key = (vocab[pair[0]], vocab[pair[1]])
        if key in ranks:
            raise ValueError('Merge BPE Minerva duplicato.')
        ranks[key] = (rank, vocab[''.join(pair)])
    return vocab, ranks


class OfficialTokenizer:
    """Load the public pinned asset; encode exact text without automatic BOS/EOS."""
    def __init__(self, path):
        with Path(path).open('rb') as source:
            data = source.read(TOKENIZER_SIZE + 1)
        if len(data) != TOKENIZER_SIZE:
            raise ValueError('Dimensione tokenizer Minerva diversa dalla fonte ufficiale.')
        if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
            raise ValueError('SHA-256 tokenizer Minerva diverso dalla fonte ufficiale.')
        try:
            graph = json.loads(data)
        except (ValueError, UnicodeError):
            raise ValueError('Grafo tokenizer Minerva non valido.') from None
        self._vocab, self._ranks = _validate_graph(graph)

    def _bpe(self, piece):
        symbols = []
        for value in piece.encode('utf-8'):
            character = _BYTE_ENCODER[value]
            token_id = self._vocab.get(character)
            if token_id is not None:
                symbols.append(token_id)
            else:
                # Official graph: byte_fallback=true, unk_token=None. A fallback
                # succeeds only if every encoded-byte token exists. The official
                # vocabulary has no such tokens; notably an isolated CR is dropped.
                fallback = [self._vocab.get(f'<0x{byte:02X}>') for byte in character.encode('utf-8')]
                if all(token is not None for token in fallback):
                    symbols.extend(fallback)
        count = len(symbols)
        if count < 2:
            return symbols
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        heap = []
        def push(left, right):
            if left < 0 or right < 0:
                return
            rule = self._ranks.get((symbols[left], symbols[right]))
            if rule is not None:
                heapq.heappush(heap, (rule[0], left, right, symbols[left], symbols[right], rule[1]))
        for index in range(count - 1):
            push(index, index + 1)
        while heap:
            _, left, right, left_id, right_id, merged = heapq.heappop(heap)
            if (not alive[left] or not alive[right] or following[left] != right or
                    symbols[left] != left_id or symbols[right] != right_id):
                continue
            symbols[left] = merged
            alive[right] = False
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            push(previous[left], left)
            push(left, following[left])
        return [value for index, value in enumerate(symbols) if alive[index]]

    def _ordinary(self, text):
        if not text:
            return []
        normalized = _nfkc9(text)
        if not normalized.startswith(' '):
            normalized = ' ' + normalized
        result = []
        for piece in _pieces(normalized):
            result.extend(self._bpe(piece))
        return result

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Il tokenizer Minerva richiede testo Unicode.')
        if any(0xD800 <= ord(character) <= 0xDFFF for character in text):
            raise ValueError('Il testo contiene caratteri Unicode non validi.')
        result = []
        offset = 0
        # Added tokens with normalized=False must be extracted from RAW text.
        # Normalized fullwidth role markers remain ordinary BPE text afterward.
        for match in _SPECIAL_PATTERN.finditer(text):
            result.extend(self._ordinary(text[offset:match.start()]))
            result.append(_SPECIAL_IDS[match.group()])
            offset = match.end()
        result.extend(self._ordinary(text[offset:]))
        return result


def encode(prompt, tokenizer_path):
    """Stateless convenience API; reuse OfficialTokenizer for repeated requests."""
    return OfficialTokenizer(tokenizer_path).encode(prompt)


# Normalization data source: unicode-normalization-alignments 0.1.12,
# https://crates.io/crates/unicode-normalization-alignments/0.1.12
# tables.rs SHA256: 83736e2488e7f5e513317e035dfd476611248227efe0a456fddb8a641fe9ade6
# The source declares Unicode 9.0.0. Extracted data: 814 CCC entries,
# 5722 fully decomposed mappings (compatibility overrides canonical), and 940
# compositions including the 12 astral pairs. Hangul is handled algorithmically.
#
# tables.rs: Copyright 2012-2018 The Rust Project Developers
# LICENSE-MIT: Copyright (c) 2015 The Rust Project Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Regex L/N ranges: all Unicode scalar values were classified against
# tokenizers 0.22.1 ByteLevel(add_prefix_space=False), using literal A+c+B and
# 1+c+2 boundaries. 677 letter ranges and 144 number ranges were observed.
# Oracle range-file SHA256:
# 0a1e5b694f047a9ffdda0af975ba691460d030cdf77d90222ca0c76b2fdcdce3
# Regex source: huggingface/tokenizers v0.22.1,
# tokenizers/src/pre_tokenizers/byte_level.rs, source SHA256:
# f9b5a747a0ad936f0c49cff8f32bc8f166126704ab02d5af0505ad032fe6d99a
# This snapshot avoids Python
# Unicode-property drift; the complete tokenizer is separately oracle-tested.
# Compressed payload below is JSON data only, never executable serialization.
_TABLES_SHA256 = '8767576ec1fd97b2f9c83a1f9f525985708b56c57ab41a51cfd3b88ea46745e4'
_TABLES_SIZE = 132394
_TABLES_B85 = (
    'c-'
    'nNm$&wyR&MfvWy{<*f!+jN6uYPGMllIs2?h3#iLl!Bs?o(v(8!|H@FgP5J|NWo;`Jey!&;R=0|2d=m=YO?ce%_zT`%`;=TC=Re-'
    '`2Y4<ex`d&3@n4-'
    '2eGswe0tj(Kdg(R@;rA@;<rSuippS*NonVec<WCzGiRx_V;Oh`M2M`dU~h*x$kw}*VpqtJYD$d_BFel_s5IxzII7P{<$>y=iT0yZ'
    '2gKl<=1t+U$1Vrvip6PS>IRP`CeJOzN=NcvZ2@B<hRFuxwl<kUG4fS*L$U2>ARH8OWCE~g9~UKA2r^e&$4H;;=^_+r|8Yp)^2P5%'
    ';XtxWx9Xx^0hm|uA%jF(t>vXb*cV9kKCo+PnpGD{T^=h53V<srPaJFen~?=TRx?g<NK!XYS*;xxh1{RzQ46S`9<#vhkgG(`ki`JM'
    'c%k&Wlp!3wpsi2)vt5dKfJ4r$7H+g%*9Wub=u7wyi%V2s_)<BpXJJI5ANm9>n-'
    'K`W8HFhzB;$vz2eUmyQYoXaJ4H=7`AfP({t_goK<{#Zf?)_Memn&=yKY#IMvp;&N=c@_+f4BTt6aLFBTifH_iLJ8E>t#a_7HYF0X'
    'DEzaK;ILZ0vN1HJ8$o}ErfY{%(e($DgX_GKOW;vDoU8}sTRvbVekdVSAzIh(c@wqNfU)Ayop?R`tyH~!i-'
    'n=jkD8s8Vk*S(K2aTRZe`1b6&ubZ6KA_w&9X6PqwGVKfBVsBC96XfRNP<E_^Ua#NW>AqNJ+8_6*%U5l~ZtYwBzT5)8Q+NDv`fqow'
    'Wni=U!+J|#=VJS1ysN))^(~BpKkj$uCoWZ1dBE_`c(U!59*?;u1JQf(6g6$LwcPAI8yLsm4gLGju$NOzdy~E6hI_j~xWkUHEb;a?'
    'VRNcY^T+DjG*<mR%pH$9PEjn|@jtD*OXEgWA5h>p!m%8G#C9~_Enz9%^3HH>$@{s_J3T$qb6Rqi{_K`m!amhLhU=NW<yT9$;QJ|W'
    '=z9lz$6n1h$8vk{4x6vXo&J=c0optqZ}oitz@wO}^^bdt>IT7W;d21)z|Iac@78eI$DSEm{leaeNqXNF9<Z&jI3?}YoBZ0^v7`OO'
    's4o$V-P{ex7WM=8qK9j?uKD3U>KN4&%a`mpHU0`jG(!U&GcOk(TyZD9{q&=K*+*@$18chc6b$tgin-CdzTFm{a#x=vuf4LJ{L*40'
    'M0lqcG{QO*-t>@H95ui@t>=gZ3zIS8=nh#M4gR{jFZaYXwi-f4eW8B))FbEy<gLVvkjd`v)Rw{|KQ1-)Y-'
    '~f{BhO0OUT9}8ySmRB%=m;JX1K<~4m+`01fe_K_)gcp(s%o|7Q<I@9z8sJan`_bIw_cN&-'
    '%XDH^MucJ&tkMx0aS+rT{AM|M1W0Xx5Q}6S(#Fxn{3MUpG9O-'
    '_SaKaqkq^R}T%abg!Lkx=K7W4;Bz+3=L#g#82|R%D2V$hIz*(fYBIn>XPLeaWNf+O-+WH-'
    '0(#wnW6A3damTi)?`C&JrQ%Kec!XWDDjMHvgPrdTlkut_R~%u;>nYwI6I#7h^-'
    'j=e%5$uvAIU~I}9^&>PhBec{I6v1$Tsh@ol5eFk8L?CxR?JnqrBcJFg#K<&o9wOiadz1O)p<gV!A%+*i}zEeFd_cf;0vPRO6k@B{'
    'Rms+_7Kp>x}t{w)0G`yhY(vIQ%iU=NcU9<_y6@^e%?ta|RC<NPjv)Ee@nJ-'
    '*{>cRE~QDEoyI2){3%05dJ8u!oCSIjo93QFAO^!~K>>*M_BQIf{wZo#LM~P8yM9mD_XhKe)bUcmQ}l+q=b0CqQ7QWX|)-'
    'lMIya$7dJasc5*3UAp2?jY3DX-'
    '2%$7S>eEP;9284;vlT$X>N&pqD4?!BWAxXXO8}g;Sb={NEIS$7;VNHgz4P1=gcb}i^YHulF1Q@Ajq%yMF6phX9+Xc4XF^jo`SVFM'
    'J31AzJ2|ZPw6SB((90=evc^mUc13DN`*1)8Xunl=d!~vx>;`9Lc`G_eD_TnHtbn@If??S^azC{Uu}6w!|{dJs$HD&mQS1cjp5)*x'
    'D>Gr6Rx=g_5?5dF40d_fJ5^Zq;}6TmcT;t+3>Z0{v(B+HRYv#F~yDC^04gvHz6UP1U={91EyeU?tF2gZp7m^+)E0Dr*9rP!Ij25S'
    '=Ep0@=&fKCu`h|@!5wDyXF-B#elx;R;+YA9t(-'
    'H>0vel$yqCXbO2YyZ>;yhEwtpg;i?yZ_Y9t|m`6A7?JJ$69MZttzYAEA^7I?Y%Z=?Q&<tRQlTzny9xP0}7%)YI8ot2JYGjrfAKek'
    'JwXlrU5~v8K)pw3a`u&Y7Pyw+^ewycTThn?9W6T6~q$U7x6cP1MH)otZqNlJyXXJfY0|9kws`!PTl8AHM-oITPE*CcZ9dq{v$@4{'
    '8rs%4KkN&HEJiU1j(o>`or_YVT3kM>Y2tN=Z^3s($`cwiasylnUZnyakXvKCbrF3S9XNSq`ff}~>b&5!U2)y^L>N{JvxrZ8R7`i^'
    'xnc~WbOanWcGof5|sNRxJF6sNa!>?=QL^$ZFOCU_|T`jCa!h0gd_FG$lLA-rZn4`Q6Qwf5-'
    '*i&GB<TEQ}K9Sn2oPq}=rVn0B;a=mUy_k^0!$3>T`qtBY0uM~-UUkFY6;0JdYPB4mx!t-'
    'R^|q8Rk91yFr2ajClmu>fiy@!MSTGq~KS2Y~_eh349XUC=DVJKcveP9w0CrkuMA|ZPaZ<(?*+nE)arkRG8q?vNenxn^48`;S3wK6'
    'J0)Qw9&$wX_!q6@EEAqya9|q7r(uMA1nv<UzVej2VN>uSNhbK(=3G*7+NV9i#J$$$Fbbexp>{1h6thNDn%mqDBA&@WkLX^jMatmS'
    'N;;i<Pzp$MAg^@dmrKlQ;$i^YI8~}%0XV?5YlDl`<nw2<S*i&dA3j>MmDR*4~y+sx2&_p-)NT+(>H|1ksj>9FZk-qTi2@h&vNoM2'
    '_6&JiN3sCP^=ITvUno{5$wL`VoRqmn;%i1Drf)zs5D?C(y;xB;<L)Qqa$$pFw_b^BnD|FLFs)686h60Es8Xv({i!XVW1z;zK%U-'
    '6FS6HoLr)YQjlJ|7EzMWDUmJgV}_WGvY5}#-'
    'tvjv3E$_)fB%$4I~&vxEOAxp~Q<6y7Yx~FpOPR(&>I?E$$W*`M9X$*c#TA`HxdS=yi*|px4s<(UH9Auz0c6L;8<w3IOSTA-'
    '+i}+6tV+y|gaFy_1)h!GClRh7<nCP+4X!_{34k$)5;-jmRg4sxG*VmwW1hPu5K=pKlCkwc10R3rdbdDi`JAwN%*)~|E7U-'
    'W^GdJLBJLK$pV)O1XikWWzdoHwxS17R`Blj!;u&t=LjmXWTZ2uCx6o2>d%xSwM)F%N4BVyI4LAU$aeo>#D>5qYZ&Q!{pvGJoY=~n'
    'QWoIp!bc;#NCj-KlFzN1=0;CF_jJEsL`7FRg^ver6IHW+{ymR>-'
    'FQCcdx@OsCuFnn7XJSp|W?pq_u`l(8HO4@IM1<#r)>#2f%k_mI<?Oj}rC+SGxRn&0nA>2HKa^4ahB=A>9kls#WtiSw8jmrE{%7iJ'
    '-^%!4S6IVC9Td>n|39-'
    'a;I*0}9aig?OBt$h=iad2_vzU=ax~B%cL>zPj^=Wec_W%3ypa0|k{Gb2(|N4J~1M^}#S*bsLJfryE_o_RzO8!U6y5qlE@0P#5AN}'
    'Vv`OVfy_0N5=;d^{Pr|ZkASkFIQK4WjWsXu%DJb(OOK}wf=!gwz)_zH#l{`A^w{jU_B^$h-bL3o||1#-'
    '0hyeRU$<NUm{$_tZK`}555kzX_=smOJ<AE`X*MaVPybNkjezAJnG-};H-'
    '0pBHey<~jiLZtFX0x|st#eDv}MC0AqUhg+UZ?xWb4O?zZ&%%2_{W(QGR<njb@4@VK<OAOZ(TA}1Ex3;!_wio^uH@g(l+WXLynJu?'
    'HE+M!XDj9I>cI`XyT`@8{nl{tdV>D6mx9YjPl*-()G80T>$NCs^5-'
    'pLEx0zZM)G=HllOi(ihrK@E>|y2|M|Wv*ya6Szf}F&yWeFGL>HmzBJ?dwrBTxR+INS0@+rxs$!&^hYPb}!T<@z!srpzd-'
    'n98WTMOIrSdMPF_bpn`dn@xIX?;fe*Sl_dMw-6;d4K<}rYQDY2d+TUig&ZmeDMQy>7FI;xO=4-yxVxYNjJ{XMQEE#`SzB4x-'
    'dh21ZvW&ES0OTtA{1;3akEU^Ce&U_dBpGS1<Byf%)0WM{nnz&+VKaTV;^N{?zr1an%cxNaW9*_4Bf!X7bNl(z&p>M-'
    'TCwXX1%3z1?y5;OFYaKLfpSFM3(B1bF1{6Q&-'
    '?_r}U6rWg5w#J3H&iuy=ih=S9iN!TCx*?%;y;0EdX)h|zxmvud}wHzM)_kfBn&3le=sp>`Fzsu59e$CZeS@wY&yJ=4S#a2k-'
    'kJ#+DEqHT!&g7<X_g%MsePPXyl-'
    '>9ueUklId7h7>e}B6hn^rTaXHM?5N49m>KBJ*KVol$)@`BlM;GU5y3kOK<bj@^twi@C;M+}4<p~-`&X6%Jau%-*v)Xu-'
    '|5JmC+Ji}y0_dVdEH}!rtf85V?>Q&cw=bdJu=qk0k__!Cot=hV-'
    't&fv$cw0sv!yNPH{TX{sKO(H?#`99^_|Ep>37YzNcBehsShu(QJaXzK-'
    '$K)Lv{88j&su2hUGH8Gs+`<Axu9FKqd)J7@$U+}iT}Xg-'
    '%Z2JzBr2RW5J!=J?#|f;;i?AVs~9|*Ogvw9^Izpr~#4Z26K7zxhYczi)jvMR~%dUxqjY%^^NawtIHXeFKe2`z3uvzP6z;clX|`4R'
    '_-'
    '+0`VLqz1I`Di=c&C&Lj6L8Uw8({jhY(UcM$Z*YE=48&X5hKk4c}mc%$ceaWNNrMpsOthOUS4{o$nrfIfe~0hKRQMD6Xle!1t4=H2'
    'f%?>o%(>(az#y*O`rw4Q4kcHZ7$){BWl`to7*Uwt0lu0FWI2QEvb8>~H3s%K~M{t7?hlv>P2`$2j8TVM9f%*B3J;?!H{w+wn6$w8'
    'UXt_`lVOgWUgw!05TEOxo~A}ksO)^5R$tsbsWbFsHz@5}if*o3)oG3v)I{MeB9T--tL9)VoMTDA0U)>Gf`{3{&aEh*GWF5rb*jf*'
    'J=92WDx%019<ZOy|qdrx}ngM9(;!EU^6`|W0THtp{k8sf2cnt7R<4;2Sgze8h%VW(p=c)ICRJIt_L0irYAJkQd^jl7LAuu)(hxAH'
    'X1AYi?ok<i~=-'
    'wEo>;S}Y|2Np+nWzjQ;AZ69!XK<+zS#s01ar=eOqm%bl)y*MsafRF87W7qli2@*h1P;ZylU1&WHa0zF*S|mS?C0U_%U)rZZEGz0!'
    '3IMJbNZ|eV&-RWH;h^tYF2b3Y>PZ)-'
    '>VILXP2L+c*48qt?p<k9`Z*(pr7@28DFDO&wGys7Y+8TGj7v;uyh}+%6*_FyGf46Fyk`JpwHYF!5|FrsOrnw3wjt2Gh%<clYybWe'
    's<oo#!A~ZqLv5#sHa3wR&r#F{nGEjnw|9{AugTZhAA0~_P^<U9+>~F1jj$e_L#n4;707mhTiBqxr|`xyIN@s?DfW(6G?<|7>q}R@'
    'nS3erBG(ohg91*2+8I7<8;r8*hf>ww`3(;1-4SkQM4v1@#P#AfNy3&IE%2++npmq**=>F-'
    '%7<6*K>+xF&mMN9v44ja>gq{#!?V?ZnN~3e;tJJFF!gAGGW}X3@?apcDy|;ly@OB87~&sWKb&se}onG(@RId0k5*;Ng>#%h&A8#g'
    '#l3eqz4{(9X$wk=g~!M>gzmJdG11QKz#V&Zp7;ktQvyEo$^Z@(RaRajJuM%-nN?eTH2~t^)fKNK(G775Ehv>m?>;C1xEhdYyNp1-'
    'QJQFZ*z%i0$#*BVOBkY92P+y-R|&Tbc^~ahlXGyI##=3xxvxRKO_u>0lSg4S}^fqDE;6UlpPwri;XkF7twFb!_-'
    'FC|HT~)=#X{E&rbqy;JDvYEEDhRhz#$sP`s+`8n-'
    'fA!8s7*#_SM|+cu+&{Jrj>hDjiGP5$1`_h8CcaF)8$)V*fd02brL!@Js+)MThVmxaiEcTem}oCiJP67TA}U@@La3a^CVN|wgMx}&'
    '^kKLir?!)`TzAn(UCv_dXTZ(P`r2LTrx-I3%50oO**T+B!xt*uK;HZ4l4dVgvrm*E%-k@hqGu`pBX$$wtJFp-'
    '9j>X+(4u9kO@ikr>ST`h!`!c(_!>MZqy-MB{sC2d(^n$#>%xfZC*Pt?hP4Y<U-'
    '7FHdE4bNB(L!|cPv$H@8rgchgs7HSq4&zV}j`}_{73yhT{KA||kD|ubk#lLI2n|NIUj<hEO!K@0!9^RubaossiwWLcX}%%Z(PoG|'
    'kdnT@dmz*8NaDL>g7Q^!!#T^ME%)j$F{^u&jI5z{WTil)>P*-'
    'ggYowfNInp(NrK`C#JfJ&^7eLz<_Y6t?8WmVJ4SLVS?Bm<pXyuG{wwOT%(npgSu;J112E?8+~ZqzIx1pY7T}7c-~RarxKRQcd6-'
    '^8(o=@I%>iHi^7jN+GwcAT_WAEh$6m1!3Vn#y%iiOh-Mr^QV6d5lX?D@lmT29r1&_=bv6=Mhc0O2f*|OI)9+IbI;r>K`b$L03Al<'
    'UVu(8u^M>NK|@8=E};2A4@pIA1aP9^PcUN-<mdFI=~Bz;-'
    'kQsL0MRhOtJsry7+@jJVOiSR`)5@M8j5oI2WGG*&uM%B1t*tloo`NbEDH`BKu!0+M-'
    '&E6wgJ_}fj8x?|^S~&hANN^mdgZ%mBl^Wb?d0oKW{UaO+KJN=$tQzXou>5Dh12ZzZ9w`ny+4qb9O;+Mrx6+f!GXhNd+#M|<z1Kos'
    '_*Ml^{yj`zvW>lpW67AOBc0p|#1D8aIYoA|H=^2s)K-wP0{hXpS6l2=%l{R5kjk2QR^9*gJ%>ruW&$Pdw%`g6^-'
    '|17Err3FfkUOPu#((H2k=`dxGrW$&x=cNEWf)7JY3&<vPl@b*@`P`UDB6{xMyAW$rW)=B^+QlTCX_U_XO&o5x9V6=Dn81h(y7)Fs'
    '4}*151Zj-4W=)(2u@|bQPPxIUTi{slghLW*G&3@iLqV3$?$M2*Cdc=1%gixX<TXxOb2tz2(uw7MMv{+yq-ILk=<`i|qz3(E^W^U*'
    'aK`rcJ_;*ULi*C+<6-<-;9umicGkxy`&5{B<dnQS|`c^?uqdF*mBrosg-{-'
    '59{=Efn_!eehDOl2kp9*Z{iV1Mg~<G3Y)8#MiBCH8tpw&MSB<1R^-'
    '74Nhmc)3Ie;Kn0899@^EVp>(m+0<ScmA(}9dg9g%2(Ry%mm1u@ugaE!~KNf0Cy=(09!U0&iw>BL>pEgo(r_0~F-'
    'q^K$E|$K)e9Zd^vkx8selfdMmOR(Q346xD($l+BGOoKEY4g^|ZK%E67D=2-'
    'IPh4YsgQ1iw{#DJ;P9;(vsE0gmq`Kf(a2vNIKa_UJyw#dLTIQ!5=<o=_&b~lWt#v~S~sUYQTThd*iZ*uMM*Uc=FnfP%Lr(UDE^^V'
    '*{xP7;T$~NE(VeyuHWwi7v6sT?H(s1u#lQ)T|{qnBM32w<%P=ff<bC9)<@&ET8S#j^X4b&S(m3>E$P9|m}_j?uLl_!5~pVw8$cm1'
    'yd7YwOT*m0hvlOd8zNkM;g#7?1h>!2<x0sp3=2AgML_)>iS~ih)CYnXUmM0{*>;L8)SlP7F3XJr;KAKihk(I<_rj&eF*qxh9;Twb'
    'gHScxT$XQPhoYu+=VgX}P&19l;u-z&tsBO<9mLWHK5FMRj*v~B*+K;NVz!|s*#Kp1;muN<X6VVkp}zQbAou8G9M$RGj)4_k<-'
    '!Fm3rvBAzmsI=0Ck&}3Fy5dOM0>Owp4Mc(^`fE2;{gETBiZTM1+CbZMtwXR>J^^Ye&rhJgy>cV<WG67r$A9phIg6j|5h+@JMN^;y'
    'ba6pF8OF-w9z=jJ$c-!^#-'
    '7pl|`e9_jeFVJU_kd$oA`1v9+=4wqcP0oXl(NLITjbY~k;BjB=TdE6){KE4Rbj!;EBh+uEP2$$`9a<-_~T0V<dbu-'
    '8xl_+3AE%uh3a0IJ?F=z6a`CDa@>qd356@JY|i1Icx!*JB&Ht^tvizs$oN&{3c<Aj6FqT1{h2mq*C#mh&-'
    '%11jY4!huJ%TEDgj@s3UM5f0VjE%kX(p-S=0~PkSd0tXk4J+8cw}t}49{V8-'
    '+Hn5W9@jGz`>$Pd5UWcd;mQ0)U3?Yul0>BMGV6aecoh_EY5?7SXGCBxyx<c?wRP?QgVv6=?HcfCU=f{)lLsu?ZrHLnfG$IX8aFW@'
    '^cOeSE$Ag#nNUb98+Wkml%udP+8``D!?aoLvm#X57Qy4#_Hmhw*Y+vZTEKuB_+`m_zyQB*>ZrOjR9NWH&aNL=jeLmad<X~o;5p|e'
    '7<>%H<N0WoxJFc85CNd}yp@;+&;e=Oy&0L~iK&1*UO=YH1bpuqP)8*|BLLiRlDE)mpx%p=6Y<LHe&yt97K5^y@42hC!1Ky4?w4}B'
    '<vYM%$6)R~tkP!~M=J@AigEyeah&RoY_3?yddlt}KSuDWkS)?b)e3^w)gCIssuxbt!T<uA&2k=O0$Sg=>OYPp9<1D*RJqf|?BiHC'
    '?4>g1u)s5L-ZcP3R*b5{FwkCrp-bTD<mmUpp1m;4!U@EDqu5bbeF4FcHM%{Ub&@(AS_Sn)>Y$Yj7mx(rk+T2<wfx%_IT)KL*zOs8'
    'Q5W)3W?W0_gkO7tU*)h&9Cg1yB$=DUmyXn2!hORg3}{8w=G5=*wEO}gT_Fap-'
    '<^nVfjuqO?m7(8HK5nl{}kdUNc6Y5?f0YF0|T{0(pTf^G&MC{UMtQJHGic!aF3xLSWr?pVARCL`LS6zuu|){>L-'
    '9VX+|{fu~`SRJQW3JDY3JLS6Q^y=LNuZ-(XK%Ab)ExLId>kqS?kGBFh$)4-'
    '6;tIqh!LH)O&u)kWC=TTsCwrotMgi2RY!0cYZ=Xh#Cii|3@edV0v+Z*5cs2o2>Fej<g}1w23w-'
    'VY;Z$ce>{0q`U{iU9K>xS%HQg<p*8fqt|F#OWx}T>&J$77Jfkqqto|*;broqD9W(ER<tBLjXDbGV%rUYyonU$L$`KS5Z)~EGX8ZS'
    '`=l8W5LJ#&0V2YFR7BNIVya^o~UE$vI0$T>JK(bwX6xtR!U>N3MJQ<Js@t>f=4We+VoRMLZk6hQ3G1y?uAxT)vS0Y2j;RiM-'
    '*_C;M1tDXee(tFK@HdTuXxZ@$K_qZ+EOg9rOY~Eqoo}(g6?%iZGH<0?DY&Yo!*{8~clj8AXDi@?4{Q^4MOXG{pZ5FcS<0Ls+^E4='
    'G0FGAjeXv2C)8Q9Z%|XojSrbTT!2pC~9?qud^kqZn|H`6&eeW^nA(G1^V`?68N346TxeH*Sx#j@Qx<?;?QF=X#tZYAzD??dYX;b>'
    'XM^IYu2|{^EYN52Gji{qkJZUb$0{VRV@>Nwe{!4SwuBQRBKI;(MoAk3?VBD{=rDO0(Rv`2#x!Bp|~dHABgs$RH(M*BF1gSA_0j9u'
    '4)E7mFJ0mNl)iXoWXx$S6*orD5?4!bbv`Kr4{)u-'
    'fq`fRku6KpMfdGPV(j6lL~9YAZu9<kyYN?Nf}}jb!Hi=s)iicsXXX9AkxPBa_hupHJD)5*rJHgMNa7KtqjF<9rrTx1pxUE8iovOa'
    'QbFk}0I9&LCn%5U~$xdy;QgvNGY-'
    '1he7aGZ9$*CelK{TWtr}2((yXtKB{Cgcq!d@eME0JJ_|VPQcX5iHz{Xt1L$r5iU1WY~Pb=Mp{s!&!}e7wvz_Hd!EMaK(w-'
    '3gRfmq4YH6HEUlNp8u+iWJupthc=Wp3AOtlEEBf1(djc>Z1(;QHJt!;D0FV|(*nou_4n!j`xu9`w;5LC49&)7{xzaVQW-'
    ';#TwqoLf%@Fl9E2#AyE21H@mbrFKs2+;j#49CbgdacGB254;FB<ENMUjfPr9>72BWorwpk0<p4ZYJPzE?SMZ#l6X&B<jFKhd5VAh'
    'CeniDVn&smu5b<_*0XE!S`*1n;ajT2X8#BD(|?u(*{J9HQJJJ7Xayl9bvPIA--aO_q<xg4SXsmj$S<o1FtTc=p9Ixob$R5fW!?mJ'
    'a!djIYPw3lLxL{TjlD!07Gpf=U4CM|4ONwYok~H{Hxn4I<nC0u&KKg$b!s@823Y!`$a)00BqQm*E)^-'
    '_w8#=}MyQ1K&abCirBPb(O-eo$<9Wtt6(u0M1MI@-'
    '~M4e9FDi6U>x@E&vfdzzy>72C1tTc+ERpq)5O@NG|An6hs{JYPqSe?S~lK4D6Voj-^T$Vw-'
    '!K^Si*Qv}(rQbXH#V13N$xMKKt9tdQ{1D3J$^JX=cBB(OK8!KZ}ZCR6!sM@N=GFI&eP3!LSy6+j4Rufy?fWIO=42yyPn5kbdTiBg'
    '8R0wt@fk%zT9pk#NE3+-'
    'F`hahO31sW7w91B%m)og|*x3|P_ag+n4nR&a%hX=tO&elDS+a+?!CifbL`fW*IMs`<X1z3}&l>qSHT4@%>PL`kn+IQs6fgh^3k&X'
    'gSe$hZvkR1!$KnhivAJ~B_+`-x%II{wTS-'
    '6h6<Gm3cOvE!~8Ly7qp!X)E)pO)ELT{6LW%xB5lDiGv!kIwMkPiMT>}XQ_woR|BF*ii}OWVb61tjHn)V%)yOH)W!)J0fMNG4la+I'
    '{>!U~2EOPtYq9Eh0p;08eMmW7}|J>P>j)OxnZJY|sauK_#z&toi+m1*Gv>C2^2U$K$zz6_54A?&{#|sULHoImIFc?S<~%vN217Dy'
    ')7FE{L%6jH!r%$g@ZTM3wEqk(X^Y(_uUIp!~eMQxL6MAxHx@rVPliXaNU`7@f>xmZX!7(Qlp+^)9amZX=i)pPSX^Wq8c_-ef%4iL'
    '!MWR%7izosI7>ww3*l#e8tI>v?8{CWk@v^eotKXGYUsu`_55MpMsHXHjt?<8QNNrqh8(DAoK7{M~LFk)<a)Zh{D_4NnEtP_^d($>'
    '-Z(gC2G%toDIqbk}z8QbYPrG=H)PSR30+N@iCf+<fN}G1DPI`B~{7Z9z7Rh#PndiMk&6oqYzZLpQV$7{Ct0#RY+`Z$_BAX3+5>k='
    'q+t#o9rNFed}_V-lZ1$`irhL9B>hkYv-4<_6M%FFzJVpsKvBGYMqflOO0&>8X_5j?@B)^Us3DsHr-'
    'J2A7Tom*qe%_r9Tr744#Ld2;m$9i=R~srRpmwWTN3(q<kOAQdk|GGyt1ErJW<_Ea;Bssrru=5tZ&qwZpblLTKPs0wfO8%!qp(k4t'
    'QIiOngHtQd#VV|3=hvv>S-_&L+Hrm`Md^x_bzllkzZlEpsxq|M!k~f!14uFl*Ps-A9(nOTVn5jtm-'
    'iM&>bWc=zjN5}C`Tc0=lfeFN$J?lgD-0k$YUA&9!uzpqa>l`De#^RPCv{YO+C@RCByMu#w-d`+4m&78yLeF-fO41%<>TP^&64{(6'
    '{EQ8MkA3Wj;PeU2n@cdd8ODZiF)$<aEeO?IqgQThc3*^z4c;eVH1$k8}1kT^0mh4njg;Zv*E-'
    'kj@3*EZ<NIyGQX)SKW=P~CET^VkOv{yXfTMlbaZvV_o2X4r3s6{`fS#R<0K{2I9^<3k*HPzMzL0^ibig>1~e6a(KgOt(a!>Cb9MA'
    'CZ>R}aD?1GKM2!Y*u3i5_KJ8_ocO%dPlXP9%Mo1XP7{1fN)T|8$p9M-'
    '|4Ky!=d65QRBCxNWo{Flj@q8CDa_Npif(?xBogU}aqE0GAB|3K$r$iDD=VCbb(a#3xjnR^m&OBls5>f;-'
    'n9LhVmUHO>$7M$$MAPj#Otf^aV~s^S=M8#YMBbe0Eq)$wduJOFx_{oCd$ARnbl;JX7bfJ~1PDZ*IpQW*7~!C;IM#E568E||#AiCZ'
    'Yw0?hH^~v(BM~BDt%k>rENuX?XvUuE5EbapgClmkpYG06V7Yq#bS<%2uz}RHH(--'
    'um4vrNPu{6|?7c|p{4+EUA8T0cYD14z03>+Y44!VTRDw)!K@(q{T+enq7y#>N4yDJUukOsK!ZZMox#=|Za*Ph3iG^mC<;WSlo<@L'
    '|ZAB>1LWI}}o3Op{+D`X|4fJ*pe1d4_5T!kxX*(@1m@sQu+D_P@4@RQ>+oR*OH_#ys{H>P_Ku95HvEyYeYBlQ=2=|_~;Y8Cka%Yn'
    'JKw>c*<_D4W{g~0}kIJ;$BtxfB;55Sfs4xIj0O>toDwKj*TDg&GftWs!B8rPb$5+E4xwC#ra6rFyoE$T7M#nM?vQa7J{fy|djYvr'
    '>=cMKc+PF?ZWo=#$vzLe277Mh>)_z0L;1cH|R)H|idXdRGsHTPDu69v@L}}HtS)tlda`^mNPz>57Bf)r_=&O0I;YR4f%D<4M@x;H'
    'S%P79z&Q4E?twg75VBizxMd}{xjy*&OY0UfmhRn<l(GN7Mx4R)hyx=&Ux})jbb~+Bw0rm1gJVe3>Yc8iiOSNjrf=05O;%VNXq401'
    'UxP<mZ6B~6Sv3mE6ydmn}Ez*GX5x>bwC?M2cSnvZyn4><-'
    '9{(hweA%hpjKOS~i#O(g^R|}dAy)r}{jU0VR`bNl8<?Xz1L`BFPPYz1<TEAadwK(Z36yRyN6@(pQWz@$g9pSO>7n3C`;@uKr#vO)'
    'ulrX$y_4RBL<T_5`(fPML0ZDLgb4-t;Km*06MA4^eSafMRDg9iyiRD(dlODz%+L_x4y%DL7g4Yhs>^#grx&s-'
    'FWGP4_8XhP<zrBqDA$n)d?`|<=g{D{pNkNp1z`>r0R|$D7N?gr91uY_co`Qdg#4V5i8u0#^`ldT94IZ@^tGTi*m`7%B<lxL)52E0'
    '9fSNpQre3_aTmrwehLt98#@OEv+8P{1+bNDcfr&*Az_1`*(O_i40bmY3!Zts+X^J$GxkBM-'
    '1TfK^gCF<yEQs#a8|tArXw=ws|;j8$H`yC&)dx^O-'
    '?=oN9{$?_Tf;jxS1>tSMlP>&72F}fMI`n;Zt{Xxq}sI=g#ouzNkXO9%Xvvp1g8{zRZpa8stCTqcj#>``}`p2k17Ess&(&Z1b#qM~'
    'tU=+<t+p%Fn=kUc9zal!rD+^p2dh0{Q7E3H4?n|0ak5hds)-'
    '*_bPi0Bf<$_tpfl(`^iNV4A>|SQR8<2g|!j>$%>Ya8jdbu3<D_XB~d+wzMmpU5b+2Iu601klsZKNmq#t*w?Ghn9HRF(}cgH_Jkg6'
    'Gose$=>8D$Q+-'
    'vtL_qdqPFg8KgwaR_N|PzkNJ4T`Ex<~*jcLF{0qrIVIT*RK87~6ys4&#v>}Te|>Qhm9R}FKiri2>0An1ll08L>EAZY_~k4`RL^Rx'
    '<S1IjU0m7m2Rv0zy((uO_cp+uNfQdeV#nAPG47hF^0*x#O8r=&2(^{zJfVHt2@iN6k?aRiX9Z0Iz14~cB){fI-'
    '4WI#I3Y~PThH)gj&H6pM*L1BXja@vzrhCqp{hEg4CxMNqu?(Bi}i`|zTN^8yBn-$>YwUC2u4aw*kNh5by-'
    'Z=*F_gQ~FZpVyJ`|h9GH(W5Q3n^Y+<xp(Tl+IwOH-'
    '&nJfyW3y#ThI*h2fdQM8a8DG<Tylb|GdIfSrw}=2De!xBjga1^aLug5)78L?;SSv-Rl%cWMu`#{k}MaCX<p-'
    'EuJV#XNU6u`;WEwB!a5-'
    '$EO(CP}DGq9iHwdl~bVS#as>3}&H$6l2wk^WJl)ne>m^)PUB4kZ93kT&&*Yku~7l3o!&;=0xmtfe`z+Y62oGxowxc+8~r6ZF$C~L'
    'mTs;IV*bNQ8ZA)_<r=1n-uEzU1z+Z5$CYkd9!oD8iqRjqEv^%p$^KKHSS!1wa-TtmzTQ0<OFVK_DaU~{V=u{tA)=V^I_SMVE#O#b'
    'IAL@PSR3ri!D-b@m($l^o7s}>-'
    '$5@*Y}~NqB;knUv?=j3cawIcP*o9EqFuLlV2zsB8ADcAe*jUKmbv+cZhiiE^%oC*g1=iw}VuIJlEbMOa&f1P)j_QsQXD75A?!b9_'
    '#vHNCuGCN~F60g-'
    'i<1&5uNh6@}QU2)b5dtMxUn+O{{RaibiQb7e@^C`wRMsUJFq?Zw^RB8`X<7^h0?qtIE*!!_<3_I<?juxO`czuYON#;SicAqK@(Pz'
    'ZvjtyWV6C28L%lEIL_pY7X93+jSQNTss{5d`N_ECwvf&|Gw^VcUTm)cd98=PVYAS}!VhsROvM36-'
    'IZ6^&{>EV2WBZjJB1t+8|WXM%z@=vt0|JAoXoj!4{qVp4xW8>oN7c?KPg+Im7$rDmgBNmMlmv;h>TMxa>~ex?5HCK;zWue4Y1^+u'
    '&?pCG*^c~j!$(p9gewq@9xMTeKsVVIcNO*rlt@Hi41ejnEBCNE|p$0#WVW8JAB9<Rf@NjJ?v^=?N)(HoMRr4<w=`L%}33RoASq-'
    '*&Jioz3>HOTwCA1&1Sz=@hoSwQSKF(`@}s#JwxiAt2XP@TUWY{}!d1`K|On1ba=;Go#rgHfJX$(DP=KlR4-'
    'Y#G5pUkxgHa&@?aMxh2nN_B3uruf#iJ3g!sILc|rfkv-'
    '0YM%ND217dm9uay6>3?tL1*vy>5_?+5>3U!zq8H+=PlFhjgKOg}$qU0r3u7K~fFP#+xa%Gth{)cLl|Um@poD9_nA{gHnVZQ0jQLI'
    '@G2caiXeEZ|rs!<*GFA}D^m96Ws8_i+Ynj&a3pEIt+Tld_Ng_OjHWE?DQZ_>F+FWaq3Sz{URl1r0o@aHU3nJ~g&8dR+w1)C}X6t_'
    '1l!!bXVhzq|hmo~5#m%_%YQWH0DA^%_SlRh#oNCxwY8q*^qW!yCVWK&p*5K6C!c%h$sX354cuy#(1|VilZU`N#1t6IfBD7qu+e`&'
    'uWW~5qN&TAa5NhDm3HopQ7X22-tLl@LLgFw4#P?{lFeB@=%;{ZEF3E&;mf+Z3z^M%p<I2GW;+LuM3Hv%CxE-L<HX*dZUwGH0*-'
    'zNE!*csw_5s~VKxpoq*8WMm<3Uih(!xku-'
    'U!C59=|lZzA0${8&G(Ygp<&6^isR&tXRMo&%+yejRM~t#c<h7XxN|AJvTaWdN+P^N!pqEIH<E{0pNtB^!q0}6PE%@Xj&6HeUiwH*'
    '4se786h_WFd14pdI;E4R$X2N)vZc5$7Yw=nx#ZqaK3CwuwV-'
    '`v~nguDbmjBHk|5IRR8TX1Mh!!$l2VNbq2VCX%zl2X>BKYMd>X9EV8&$*`U6<KCTU2IU$0O2H6AA;MOkeSRCw?o30Y+O=0`rVG+x'
    'lfX_M`0y^Zl-'
    '~M);`W)6(O4#nP911ww8HGJ4)F>Og{RV8KW_A;+v)4HLM*KmU<|XuoxAFO2#9k)|W%f8peRj?(J3ST&7+5dS%rc<5Cvg5?FGfLI;'
    '{dfC=O*B{r*xXgAeX5%Ch{eVQd-1JNgjy)i4`Kr?tqxG>sRP_P;!M`fSL~0OyTX9^wm)8-=qaJr-'
    'm%b)gUM?GaB!y_<24Z(?p497TmGS7M-'
    'cbiW_8A7PC4Vz0TK06v$?Tk*O3MXe_F5ZE(6Mx<1MpAF|m*Dp>DkZ*PNk)f`YuP~>1iIj6by5K_5r#so48qL{@_u<Gf%o%#c`WU-'
    '~3`hs8j!C{&<tcHW69}>QOSJ#H7K-'
    'T#RP?N^C6H$IcOmKh8nG%e$`_z6xF};IxLjd47i|trvQK;q5AX^z8(Mcyz1ZENyHNQ5yOU2`<8O(~pLv_JI&A|8g6v$`uY}pmlNB'
    'Osb8EAd1U$VkjD9%+MrZU+-'
    '39?Hc)2DP;LI5(~JJl2qgk;UUNgx5WkF}^?v`m4`!F(K*dKS?jC^yv`g#<J^7YH*SsQ9r%l=6VUs&4A-'
    'J|++gT*tItR^WN4WC>!#2^rI+vH{18`zbRX%#g55`O7$l>d-'
    'nI0s|WOJ>H1I)akihJce?<b5jy69HT1+#ODsQcf8VyElUlW6>UKn2(Y}HaUw()s=tK<6S3D}0qDdC2woxpE80#C01$UvQ2N9&fDY'
    'a)B0%^?msJ@SL)Hd5Dwa)IA!bLjuZzN>Ihq=3-V6d@bj-AqaBf?h=9My|RjF4U9@A;wWQ<=B<G|gzV<uDqXvhV10$|dOh`2LDuBl'
    'OPm&ATB2Qe~FUcfLUE)}~Il_gVi#O7)}>8xNEo1UM=lN)ARMWqOS_3;t-+}E37U{+iA7E6-'
    '|1^cWIW{!h0&UzUGMJeIZn2yajVYH%neh|nVDEoDEUJDZbBhwuTPnig6`N*zRHpqL+Apq<>Ap!zcXoA@{)j+Se;!);u>VfFX_=%e'
    '-JlNa22BwQ-'
    '?&T>)gFvn|Cg>h3Vl`{`Vtnbck)4{(RNQSfQ3GwT$KD@E=EWLJxEyq9tYmpb16JS4DPUfM85wGZdRJiYp>KUw<J+iVZ;;KE;KLb&'
    '1Dp*3h0WnFvJf{rg7&8)2=yia{1kAjUL|rVZpQmg^8|t>9c<iN2=dObra@+N7FfVM93;^T<HlmTP(hgD(o5dOk1T3#Q+vE0t4oSt'
    'DIc8P-'
    'it`}WsL_1ulm6|7&TQbG}4C&LdLddOI6NtlJWp+Qtew07!)@0Ob17Nc^fMZOmS@_X!pv^d!7e@_RWl=qwiX`%3u@yNE5*riQcr%d'
    'taZ1p*{;Z%-'
    'w;tA?|M81_zj;74&NWv_I!01aBiy9Zq4SnY{;)@4jES*qrI+7x^GIs50$E{1G)?D_pbTuI45O5Te$agh0U=A=&0Vd@e<wC$%Bp`q'
    '3=1lo`3|z1CTdNM@RHER?FGcgpi&fL<iLWBH^s1Nk*7|J0x(@`g)eYgTUo!7*w<2wZ|mfMDs|ZQ&ExB_i&IDH-UgoJDpkg+OQYyj'
    'Y?RJbwVT?U>upS`f#!^AZ)055)ver!NJw;A+ndh{a-Y5XCMAxu9L@Y#*-kE(omD_cK-'
    'P0J^_D4ixR889{y?=L$(xyC?mt1t9DF=$6gzz4~YF4T7FEL#mv48|p{%deQ&OFsUt1W8n!vSJ@eiS$H37C(QuDo9}qo_#bSicNC0'
    'AMTGVos8qycJm3pEt15C`DBE6t(b;7r)8%NU?W36_Eu&PMcb|ekuqt{~&<=HdG#4nIOFB($W=b2}9PfHRG^<ImcCYv^yl|k>^WtY'
    '17BuQ@3)E|B&OWf_#>@+mIz*9!)Q^Gf_=2`;xgr&|M{~hb)_RYD$<(R(PG-O)h-'
    'U@Ba)s|vzD1)qSs!nLu^QW_O3|iJA*PWlN(r*0YPUJ4>MqMZ5bKm~&@e3h8Wxv=TqdT#u5{)hJOU3O?L%r?3oEjt``9gx_hDjHS&'
    '9+bovt-Xcp*ir*0Pf)P74PP&Vp5tVm|<Bbjt6&Yee2m8Gw3JlRlz%hn<&@qGvR?Q_u7^P#LKmnZ$zefu_E5HYqY?dm>k%ZuGn+<b'
    'ojUd~2#cOaj@PvNZDiQK`NZf{*?|j;UO!Q_B_kYxxMTvF+zri@dS>deZua%U$bBua8kr$HX;lfXJc7$@gr*?7!z|t&M!y=Lk;Zk}'
    '`$H<W3PLZ%X|W;XRy?B%%U{L&k?Nxv*I~)yprb*?X7`nB8Q3h`5qzKdy_<Z^UsauVn~{0yZCB2Lrl4FKj(Go7y7g?<W170gZ?Vru'
    'QsP_E$JOykI8<^06>taQ?$xq(scGRv}P^6SLGTG`$JBkjIBo#W5ug^s`NnhJ^l5=>?ob318x=ZpmV@a$10O7k7~e_(q|q;z)homO'
    'i6l0Xhl|)H@NJ_-TF_0Ey4o9&;CtR|KtKy5QvIP&<@dPmLTS6@K7-'
    'mQPbPdJ~Rvy*`2jz3zQDl#h#}am9$a<tjyNQC(iV0{u;FyPbWIf1+z;bBY+;z#+)<QS}m6a8RY7+6e}smhjbB#a7tOv0F~PsocQ8'
    'P4{!?QzE>ZP<^AA%X)V}?oCI-_xmxu<xe6jw3Mu3awC8>y?<kpE!4b1hCSbxC%2GDTNS;;dPaga2;*=}323%v)n)I}y%d7%92K<k'
    'sG#GOmtpRjBW6=WGA|tiL%~ncdxQGN@(H2nZCmToL-'
    'Z(7df3Ar2s9D(Xo<0zjyCpybpX%hIf1u@@G;1IUdlhF^ULlhQa|v5mPopoU>Bt>5RKbeMu!5dCGfSSkJQ1$ML~7^`iQ9plvRF~%#'
    '`2XcGd~Y^L8f0qM>&gZiY^GmLx;EHwW63NTR9#x}EF-V_P?T;6bb3k(l4%3*11d1r<=l^W&FvCfcE%rO>lOuAy`enlC+dxdtm4Mr'
    'Vfbu$}F(1Gm;lV||C)Nb#)Xp~=HUrfnoH3a5Quz%!}8-&n2?)~M9CJ<dD_;0xH~l<HEpXM7P62L*#SY-'
    '~wT7H{!FKGiU$(<9G^9*N!%pKa$@F!E%Iwro(Sj^zZidr#Iel|^L9&eB9_{0_#?kiWAzb~h$_TlToAd^XsE4J_cf;1JnefDnw=^?'
    'X=O41NXW;KT}s3vRb9HuAWq!POGepA^#*q5-'
    'D5JLl#d?`ZO?pgIjm8M6$VJ4b|KedSHC@pclS@WB`>(Da0MZ`R%jLU(k?JrCvPycuKdW^!fcEFv7d?gVnq_^f#zxOE(k<nsk^xmk'
    '>a`w6wL8sb850Vw;tl&OTofyWXybHqA2SE{gX@bh-'
    'e|3oZ5?^nqk4EGnuG;npWcF(Jkt0QGMmJq#0H8?S<T&Q7L=eTu6)sK)g*nd{^(Tg*II63db#rvoe+_O&bXhaw<yA+qthPFBnEIH('
    '+#%>=uIuurW$jjRn;CPFVhOBgAl7*Wna%~&m+N)2fCcNnJvK7z@Ll!@5oA_b95i3oVSEq;&&CIpyO#v*27-'
    'K0J6E)_O51hGa5cj6QzBiHEjE_#uB-eZ16dxiAF-DZ9fO_`xX-D!H8btMOlc4vfB-'
    '3}X0eA+4A?12;u#rItg=6gGau$q7h7g_}<Yr?g(Q%aLKqmMGJD?w<%9^LVX=psId^QI}JkmkYy-f~JVcV(*9)^?w!uAj?-'
    '6zPU!$%p8{+(C`v?y%6cjkkSzFBC5yoq@omU?P&e&j@wU^K&q`se#GqH~{fEo!ICfz7c+zvJF6p%U|(s|ns(rM9KXnHF~y#$9%}5'
    'S_e}X&d(<0LM6j*1>0c;Y%n>T%Q=svp|I{7b<|IU4+UhVm)Il%3w#~m|4Myy~fQIp*S^{PZhP=m><9-'
    'ctjhaR(+_F^4zWu!tJb*U9g<?4u$prKV^_3nUFp?iMsB~r_Qm5xT1x$uMnz4!w8h(*Ybvu7jKu(06)Suo6Zr`P)Mol=>Dx=w!YJX'
    'OZy((nA?J8+iFn)DxXG=YRf6ix+0-|NA!YiI~Zr8q($frP7@~pJ3_i&Xhm7{IV)A4%t`n`+?0tO2KeK!E-CELCnInud0y}Wf-ZoB'
    'wfk%y1r$~+7a%{GQm)TNd#ZyydyE^%{CF%jp6yTLnT}W4?E`I#Wu27YK(oXO9WjhafILCFJ_Ejl%)3KjKpjT0L3;P3c4BkToG`rk'
    'IpZr;Pevrq6g^hI9G~UK@&J~jeqjoOIHSBzUxb<Gj-'
    'hstbj71N9}?odc_YVV&n%~hf-@`gY+=RXoEvq(L`tE2=K5GT&zNZAbL(ktK-Fn83)#SPoaGa`fHd{ztE{YAZptJy2SrfPTv7*_`W'
    'RNnY;2Mb!^hDHt}{8w8paHq1DDS3H2d1Yf`hkAaBiTt=CM6_7K%(S6aF@{Lj<=?XCi~}!D-'
    'oFz~3Y+=EpB#Ti=pfkKw>TJDAQ5lxGLs`Bphx`J>Lo_?0m8`5+(YpIr$08xNJ}K3GXf5;ArscCYRi$~R+1C_2)gNp+ckF$_g?A}8'
    'W>8Bu{M4m@4GX<|FbyLH~#DGJf3ejT|bM0lhJ(W(I$A_}d(p^ojPWu9&xvOq4?;yHTNr&^DiG*^4~T$865y}O(VzoA03ji!SJJ=K'
    '%u!qjMQv5k-i3MyVbI3I;UcgeQHh|)8uK`6%oXyc|VEk=mD(h-'
    'u~y7EL4i%CG7TeAlX7fe;AaL|)Uy6tHW4A6>q_Y69zx;K*Q9@uab@p^L^NeaSM6NE!Brib<Hz!}fZ9Ck?xZkFjpAN<ZW1J*=?I0@'
    '#(S(U}0&RIUW{)tXA%f(4UKHVEXP}wMCkRZN#KbFs<N_mJzBfyw$v_TvQ7?55UHGt`I%o;w|AgtSBu>ZveI-'
    'G8LpeSFI?I<wJ&pshlQC$9;tyh)0&aE>%?kPORJcfOOwK(BZ3uXt_hUWFwx=q1?+yf<opQp8v!#|a!KxsMwSfM8tbcsE{=<vo5(O'
    'OtSC7Hk?q}NcTk8rUCJSC319c)sDL~2LFx^(B)VlRi9CNF?Q2<jtkn_#D3sn<P0p*i#Nu28{-'
    'j13^y!TaZI0K$<Jb(^fv56ZSv<LyV9zHy1jde+ZdwRe-'
    '4!84c5KNauE|MU7a>oc?hL$3KP?_UR76x9~YMj<jv7j*28h4==;FkYDa?Dc&At+*WBi(<l8InzFr3$3>@2f7U*#_;~?>Z5o$G;!L'
    '-d{@0<U;`S-'
    'B39$wbhLYG+m3G{R<3nNfR&1<Xj)&v!CyO|8hFjOE+hnA!Z?Sw&dyDYq^VT<f14BN6CWGsuXewiYD;!MCLU$mud;bf|B9xI_ZZ)&'
    'NO7^^p+KF_P^AJE<s>RhM}v(%8;Pwv6r8@b3pXH|x>yR68%1#t**-*IWQA?VRwY;a1ul*^mB$t@0h8q!rNOssPKd-'
    'Bt+YmO`4#3wpHwGm_v{<X{x9-UZ_I5D^pfYVv5li07`jO%dv$o=1fwK5ISvtG*OFjeKev--_pKlleAeWXAYjmW3+QNO4HL2*rsBM'
    '0*q;rg(!b+AXj2vwllRx+cIG{4)Y)Las1qmB$MQ1$9l3h%5dPq;G#Z10tymwQka#~0KuRxG6#z8!w$|Xp%I#|m3yduW&4k&05D>N'
    'v4T~f2!Ot=7PURga2#iE@?jKOZUgC62D2;mr<18kQ5;a7|<lhODk+iaq|Np%B07R(Hp(m#4i`nOL2E=I4l)Qt&;^4Alg=nGKmSet'
    'xe?HSb<JrgKpD{Uqd2(M^pP_v!VqDqD+#>*<5$wp5$Fg&eAZepXK5WjudUhrsBXQ2dmE$xV0!Ib7wprUs?$E%|;NTzyd1DCik3ox'
    '&LXh)rQ_%T*6`weKEF7^!f7L;?5NLb+K#&R<GI?tpQ_=;(1~MtT2eLhOK1s6WRwJgtL6VI`n?4ZHQSqn$fQ{7218QiSmorDb8^Hb'
    ';5C!*4+tO+71@@zz?(Sj_;-'
    '~lHg`QiB`4@_evVvGq)OhAg_ZIfX!QgK8;6HPuAAxhrnLX@u5fw{}|HNis=Q3js98F3TEZIYo6qpNk2z^k6t|ZA&&!*ufIeMW1sd'
    '?AS>OBNZWQ%-'
    '^z(#yj?aEMpX7IXfE>(%GSD+q%b`9CsF#BKygWTJ@or+f6OKMDm4gfwN;Fff#plM$o7tX|Non{jCzxM!Oc{(QuoFAd7Ff0c9z+NN'
    '%zr+6<sZBF4imYMeak~wSrDo*E;*OW9LU=4-xId40-Db3K<L^65Nxrsh@(pQGIFa@eS%YJiDRSl#Q2qdXCj`vR?ZC6nvC-'
    'H|6Xi)#X;6w1#1Ch7)6$P0)10>FJ6LO?+aWa-'
    'E!>dwT6Ajq93|SSIl7u1<LTJ3k8TnvJu`9{bWI3ArNMxo($l9Hf#&CJm@vTTvRED)a^x^CoT``H2$RKeKFb2tH1H@TP%}~H%K60w'
    '@~qRsFNo^WimI<1Y_jZ7t%AE^cwEqNtWWo)NcVG6P}<)-'
    'DHh<GV?Gf27JDcO6HGbME(Y#eFXPE3!MoTK!DTWgFyqFe*EON@U!Njtgba<rLR)^~{qzDoqFF{h4I@M2I4vMj*V-'
    'BX)?XE4aTBe>Wth2hM}3%6kbgBjyOR5a5CkPxscsb6hGjEQvhb{gGpZK5TrlgHVT&x|z%rVZ+?d3}zk%(gRPf2ERX#XG-mHlPyui'
    'i8OKTQmPxYm_rOx57t)y_IdECXuXR~}HY0;cbHct$*ALx|8u0E)<(1MZ#&bju-'
    '=h_Q@=GVirWjBgGgXF;(c0fZ}7g3%OV@haPaEBvtgkg9)6#Y%Qw={-'
    'amICRid<mGeSe(NY0abjl{y#GKmLmyZk~kb2n7!yB<K(=>h3B-LkCuDCl^daoa?xQ_@LAhJl4Tbrd+4u>m1+U-'
    '(3T&ty>_sZuhCbD>a>kg7`Sd4j>Rc8d}urZ-7+^kHzuu;Uv0~;&Y)?No4;}Mo*XZ_>gjt_GhPL!-'
    'xi!H%tBG$(x1LLuY?sOD8=?_;rM87L@u5rT@cKew6XH$fTeG=SZ$AY7pIO_(PZ+qFWv?ye2;3h1d|W9+ziV)5Z5oa{(4>x^gD#cl'
    'wg96c?w&^dziOf*SRY+VmFLf<?on7Y|UrrCDBL9)a;`3>Me&SyBbu2zUwJ#F^LXF3>0je<PTrNC5DAeIB^{W!zCINj6aQv${4#CU'
    'wcZKX^!HN_8gW>QAfQC9an7Dk8Q4vy6>W$L4OL&6_I@KE5Qr#AcJE?nc0ziNGuAy8tQjfN?NFqV!4*?u5FGLRm;hBDdXE)^ka|hg'
    '}o%G*?IObOF)-'
    'F(1peT==UHa58c4P&Kx#Ft93&UkVIyOIt6g?gjeLaTd2$#PIq{y+)xvhyXEODe*bSr4#_zCet&)Y%g$=Y)>|0GHHhh0tO`9gJ7=)'
    'BxIpODy=)mB9uq13ukfo8z`KkCdiKn5*&BW9+|#9zc7njH_+$g!PS+lB<s;TjLE&b`8%!)BD$z)rYKRO*fx}m35tPV*Ic0WA8H{G'
    'R=RVo;6()B)!EI8El|Z3~V!dq3xFY3z(fN-'
    'f480geM>K?)c?xI!0gm4fuo_@C!e<u>;TC`mW}?MgVa;g)SSRZv{OJ4gzB4L04Vl}?_w<$HIXoWlfIBoVCI~d|CG=<wfExp_%}#K'
    '-TV>cVFXhwp@vsh^V)7{Zu?NC5G`}OjC!)M;ki1mjY^!eM^zRY8^#MkLupMz5WTj3U+WMCMqhB>S2W?2oWl1g$qb(QdvrG{nJn@1'
    '@#_k*9`2yl%>w*QzUfBY*4&FdRODr;l?G_6ynFg#~%DyvtC;f_31IR`f-'
    'E>F7X6M*gEEV)|#Q0vWRS9fLGHCQ}vfD`V*J@VaeU#q%YDC`=WG+VRo2#sgjuz(47lOmIEzU+tY|gi}4AdE3zak?a<jyurJi(P6='
    'HO6C;S3O`ef+9T91z6MwoSH+!I-*P0O?dtLD5d{7wBuaf6WHa$X!dP*$A_zEsduG8c455S@+~P&Sts8WbsKq3mL6w{k&^-'
    '5N@;dMCf^fJ<B<;U_d!c@#Cp7Dn^%RA)1V7T2YCvo*p-p$PJ%C`PqCLD3EmaVx+t8gTa30-'
    '#LuBSLiy}ZdEJFD{BmE`1}g!_zX4&j5s`6q-3*XZ;h61)1}&fSto%z3Hb|M=X&L&8zCE5m)N8eyY-y{rk641=K#-'
    'qp3{6X()hj{?-'
    '(JTGMhzWgRv)hS&XF7raZ>7Kzh_{H8z_<2rXXBon!sSV>y(UgjZ@GindYoW@>o$e!Ns3Q{yFnG23%f{!%9E1m+B>-'
    'lbdh0>U$Rd!ZiUwghA_HqbxcMcw|lAw1+mxBczsSTNdg0~bT!<(9YxizqJ&gzuTS<|+ZilL16@mBKnA^|A+Z$qw;CA{5Rr)j=*A4'
    '}aLdf`ZJYZx@i6AulKPFVd2#^Nzt4y$tR!#cQ(K1j7TlfgCyOp1j}}>q<i1nVneNiX9EYO3%_m16@e`rI$UDqTY)K^$E8WRCA(bf'
    'qct`VgT(ZM&k$`+JRK|gfgA#xYW8{93!^Y4GutQo8>A?|3(BK%QQ$VR8{jk%TLA?+f4x;nu-mTTP)sqVh<(mOKrKecM5t~kW#>l0'
    '4o$vBuA%_>EA$YaDp4Q7>Alq5oF)8IT2w&&}eqnDH;B=%YeRMBK+?^UQX?(S4-'
    'h)dRRAzev=pjbH*I*P|f&!z^Urslw`G5bT!8@uJd6T-'
    'VM){#d(h8JS`Yqq$@A(p+ay|OUG8!S(s>OH$N^Z7*Q~*Jo6ifq4}jr!F~u-zzQ-#vce2gRi-Me)7+A>S;D(A&;Y+X@D5{HSq$u?w'
    '+I#|ubU$2+2&s-'
    'p%jvjn5whRM{fhN8nJYn;94xiq(;R?GrBrQwA#WRPkVw^TRG`jdF^>Rj3YV4{R@bMg30LS($R$6JCVV)euG`b2n+gX$mE;GQXN!`'
    'z{rw+ysu!|4nhuyUUTfUttAo{hhSOFQ8EnFc!jU|BdT`{c`$6cJG2aWU5gG^;(qiN!G-'
    '{G`wPf{lxhHGLs%?05)W!k4Hjkie~cTV`)3ndTVD$T?&2nQi9jIRS>L{y;dv<x)(H@yS&mrGY}D!ZY7L;X#ZVDxmKdIuz<!e}I~C'
    'rv=ZeY(@#$C)P)-yZU-I3pAaK!~D_L7fKu)8S1z=r@&9b&rAnI=MGhFQ=;Dq9p%o-'
    '&%rU&7nV0w*JJZ$uM0oh;%)dfooy>CM;fXs083=;3a8-_cTSpob4)SZ>cc2R?1DPzrPNI|j29Qwx4Md^ONObL3V1dH7bFs%L&u$-'
    '6tAyQJdF`~e%v=m7%5NtNiQe^8V2N5MXMGc^ehklykvFtNgY^KN#huX{<a2Sqa$0l5wRMMuv%v(IzlChHd#WaVvIO_I8))EB3qjS'
    'w1G7t|s_;X)GR3Vm!b=qm24g?oCWViGkpuAXLqamD#^eWFE?GLR8j@;X*TGu=rn?WPqEW~475B7s*Zq0;zE3CHdl<$c#3-'
    '#CypUAEX%7e@Z3)ajLK2n?pEqB11ixT#m5Q)Imk#uW$!pdH3uC@q<b2BWV1s~AZ6N@sz1n?)=q?3E&u+V259S9-'
    '{hAv$qhS4AlCCCD?Qp|h1M3#q*oRa_9MfTvAje*<VmF5;Y0|Ua*Jdngq(7l?%8jHiXl{wdp;_NaV$kvNz>)Y@**}EDaNU4+Vs@aU'
    'E&F3+b`C0RW#LpASJMB3TivG;VFs8^!aMwf@yNFWj+K6P?B+YI|kn@rq1R2(_-'
    'q0H<!!s=CV8x>v&uF$!k$Zo5GHCOjjt?Xoc7T;2$smZywY}x9$dDAMZQp|`EuYGSE2TAfwxI+z(#c)i?qN2l)s{cd3lBRSof>9jZ'
    '|LSTU<+}ZS!8>1YNZ!b96HbbyLL))tp~N>H5V)5&?GXVWVM}FbU26BOad8lfG<zCvw0k|UL~kz2Tl~r|B}}@>q@q1@!Tmm<9s9J*'
    'oQ=zV^t8#k+IOmm75&N!Lx-=+MRa?S$2jI9gMOxzD6-5bp_XnuDPlX@s1bXu|T{^E_LHGhXLNh4X+-&8T5_?p?f-'
    'oi(AqJe7IQy9K2buvA!U1i9ZRB)dSJQXZ040o9(Iol=%-*O$i+LaA2wB<Pb5O#^W(D9qooe<ye#-UnKQ2oe}1A-'
    'esyGr1I3f_1H>bkEB_`s=Nv^wGG9e`Fi(cHSt0>Zp}2foEGUA#iM+R!(w1d7+8gw2~X@6gk8O@5%J*}#?JFf^z?nUT;bcX%eYB(Q'
    'eGazsb`JB8l%sTm&2P4|D@B~zaiFMe8A20UCJQklg-*}GmKApZp$btlzn18XV{12bB42wRg#Z}h=k?!he_c%pF2HBHZ8lYfv9--'
    '!D_WP=g__6s-'
    'Q^$5#K7rx45lFNwLx*YR%LNOfa4nWIzC%u6yRzf2V%pC3a?0j*^2DC^UIFn{{QD2fV`>>OqUqmdO?|YUqmeNv<(J^dyj>?XZp=SF'
    'ub#cFGuITlCH|_$4S6=0t{jY{L%(9p`~v=<|~W<Iw??{2ezYK2+AZ*HOkA0Jr1{JyDv<pc}FPYh|uKX^Du{FFck+sSKW};cM3>wT'
    'zxvZfNaUgyqF0f{OgZG!-'
    '^suvOQ_(?|A99(t!LUmG4tYRbSe<28a>`T24zA=E2yUm}eN26imXx&sX~$3FH(@QTLqcX2DoCFn{sB8?E<R;~~-'
    'uz~Qz8gPWg8I77~-i$VvNeu?Ov^t{(zL^gaSskCo5%B=W-'
    '+CdmEjj8jhUkXn5k|6A+W<E&rwvk62$!atWDM4yX+nVqv5Ck)r{N?3$+9_6+1Z5@5Hd`315IenN;5U31Q<a*`Z5lTV%np-'
    '?Ywz)ZYnmR1S{2r7Zz1ZZ|k`^cLjG@&4yOn9jERbJ|F&@y4n6$;tzzR3@tjb6;8W<*Yua5;X*nQdYbbb>UH}pXWEn$4TY^W(`wLT'
    ';HSAm(8$aZX>Zq#+i6Z1>yX6@-'
    '1C(ByQSg1=6uL}dtNt9Ryjf)T8KSdl`=DvwQy?8wO}IkY!t{mO=AkSM3kwSDabj^)qqz8a5`7E7)@Rj+J<F5mpDUX*aDRY+^irg('
    '~F4KHFLZ>f-n*?FS}G+!*b~0B0?C(p)tBZ`-LbA*LSdTOQw!>J=|Y63RMZwf-'
    '^w)a!7>(WSPY|j`PILnk;HBdYZ(Rol5p}_Uj|q8v^6>Pelh!k;9E3^r>t;D2HQS^Q1Z0V&;d_TFrMJr_H^uYN>8Rq3u$<I#*iMxW'
    'wr6wA(?sPz*?CO{00VcQ!AfiN~6r(36S&TO=FXgp67HP!!u@;xaj^+=@c&i>fEe&+(nG`%bN4$ja>Vpp1RIcS>(sgVTe?zxU&Szk'
    'f4--%0#X0CQzaX&PaT^n|&nWEgdxp^;;Uh`uPkY9-'
    'ecPM(u;3E#MXVMPPy45Nc2Le?{w5!8aj(^4gXYs}_rLk*VZ5<Y=I>q&i8j&(<Kf!b(vlfKVE-hPOWUAFL$!a<<5X<olkCNU@^D&I'
    'z!gxzHHiX`Qk9{RUE*}x5D!wc4{as=rStj5Ii(+O9<Z!pgXq)Rup4e$Uc519$X)wvH5+cKZJngU(~VvdLunjFTi7)YRHC&Oo26ht'
    'M})qI6%+p>W#1{#pG%)|u5uBkA#FK`F0E(#lVuAvbPcGks55c{^Ie-'
    '<{O!L4XMaDU`E{aqozui;w&vBa;n`oY~BNyW7B$mpAtT~{J+lml5)GaDP7Se8OZ%7-'
    '9K{;CB4mDaldiWjY}HEyB{O2v2OiZ8=!96Y7kN)xo_^D`UNPA^Z~m!vE!9Aa}CCu`mnV8LRqbiOiS+3~4hfGFC%FpTLSw_^dOA8V'
    'SkD&~v~!T^+$$Seh_KStW$1~v5`-'
    '<N9{s!Z0HWtf1Ye!SpZ2)7L9RNds)3v%DfnKpOaT)|ENi)PesQ%Z9UAq!Z}0WA(GyY+7+c93soNhqZ=yH^m|HgDkubwUA-boTNp;'
    '$g{+B%m!-'
    'LDhg9j_gru)X}tJ@CJ$V=s>^`VP=V@53y?c+U0zB0Dt8qcpKavT3n25(q_Nj9T4BhTRjA%tcyxBTwCl$)*xaaJ$Tl{Nevon)WYL7'
    'ttf#oENt>lsVJ2R9G=hugN32MK(ivC-'
    '2TO(640o8K_la_6wvfYbLXdTBxmExH=hhkX`tIQ2Ao*3<(BNn!cSJ@Yxw{hQbxyaBFB4v8d}^lGrT`W`srv~kh4|2;BA;q{YARbz'
    '}V{+ESlkvv1(L#EyeV}twm)<v1ggMif4VBb8juP$98FKWZ&&y?N<V3ofPHIL=r=#a6<Qua!Pp1_^V%!`N$ZFrLNFNON8D6nlMAs3'
    '$C5$P_R1KQ+2Q~<Wq#&=vs|BwwNL}IJHcbM>;~TZV>qRy}lN8Pim!GxIXu%EQaD12Mb8VY6Wk?5O)Sk!nO>cgg)JZUpuGEXJDS71'
    '6%#6?$l|~F?mHDit3wbG9bs*Y7g*CpAzRg?w8YQ9okK8&(deoGQ@RL0=CzP7%}1`?3n@1ajsOYJ=3D|WP#Pwa$E=&#C3Xl+38^zu'
    'V*I`DM{(Y!RqtTW9Y<&O^<;gBQuvpwqG*<p$Cx?D*1y-DG>d!zJLfrZ>P5Rk92?x>rZRH+FmRAcrP_4tO7NgT?hmTmLU-QOpzM&0'
    'y_+mQ58BT3YxVzJ^GwUw%mpYb<apTC>lz#&+lR<&`NB04v=3+Jl+vr(Tji1{8#EAOR9S$fXBWlt@{F<j^SI{47F4FXAN)5q(g*&L'
    'T}Iy19M;fnqg&-DsDy%#B7Ldb^kGl!27TmF-IOSNHbW?G}NRUK%)2Lsp6QUf1%}UJHC`~75jP2T}o<wQ)4!J#aTXU3_cErvRa+u+'
    'X^gonGy(ILm6fOsN6nn3-'
    '+o+8+dYoqI9v5iD?TSs2pori@>H@u@S3MQ3Z*)bWfOJn_r>~^gJ#b7q@MQB;0~AxC=zlYvuRW3^9ZNXEV#V{%V-'
    'P;Kg?UHJU+@_x+eZ{vX>-'
    'Eg@>C(+!HfOC&!t{Dwj^4c<MB>op6so(f5_EHInfdzqweSiP6&fEj<Q(*t%kTodAF!*dPvXj^FzyQ6psGxg9+3MkU0^kU<dr=Hlk'
    'UCH1gWziJM(mZDbElN2f&5TgG!HNQ1tY3!hr4T^y2Mba~+B%3>cnVJ{oHA0aLDx8GRFSt<RFr}(#y%}tMYQU0qLW0TNY2D1Wqh@W'
    'p`86i<`7HG#x-K|ul=?E-'
    '9}tfKYD?(`t3%*<9u4{?<CB{0>!buyJBIRfTTN;Zj7>*m|9{gFAtVzlr#@td*q!|el`xF!P9i$>^Qe#!~VL;8GY*#DwJF^OKf@Ln'
    'dxepHvxIfyeNicZ12VloM=N0kF>5qrIKY*1mLiDXPw_3r=VLg-m_f&@BZa@Mq4JOZmWyIb;KccOIO!I;WBKe3fy^}Mw6Qv1;<#?>'
    'n+7wveoxAviddVL5=3YOI2v|9Gvm)q@*3pj7jpiYl{}@Um3H0SqGo-mT&63niwVIOa_cN@|Gr4LnMgp#-'
    'xWlo<ZA}V%k=1+reVliy)?Fe>1~FRJLAL@EI7+2G$Yye&%(oJM6-'
    'aZ_>%H!{_J&>$_Xj4vpA%2&~C}BsQ2{N($o(G%{Il8>S}0DqE|mJO7geNpRXzCS0azbQ(X2N!AKUDE(gcuW3Y(OSXeb#%NXUGqV~'
    'hC#@KfV}Cp$$B$aEpK4M(-JSevht{aT<l+_8kv84kEIpk-f6(xGn~71K-'
    '2PeGifWROE>F;Z<2$ZPvgqMX^qF4RWI0$k2Zu)#8Hvz9j%*mLHyze1WPMJkv#~qctGL1NUKRx?2v|~uub?P7T~q}v8-'
    '#pJBRV%#wwZK%D*ND0GbUjO?MMTe=TUlUXJUftib0}aFet+Xf{ik4IHc4oa0oHz75M=NqVn`^l1QU*EM+I9J8k7z=sHr0N}t;?2F'
    '^m;m&dJWs;g(GY_oHyoIMhhkBG-`wgZ{)QtwU0;f-Q&lXL<t^et_D%-'
    'G#lo9?veR**dK*Jqe^$QmZ~N%xCc1+1~PXgJ9YJj*6!CCbyO%?RTd7(AhleHemrtUZkBxd}^~Tu5{&xb001rgD{;%plFpf#k_Ss?'
    'q5sx#iz$gMKklA9ba*6%qa}Sx~}k+jT{>MYT`~7%jpsJx4{OWr;e>Sr>CpjH!Ow4{~szDW~-'
    'b0Leq0`=Aw2YbO`f3}+Js=SG@Ej<s}INP_kv7vI^RhF75ZqS|Z$*GahYY*t;=VQqJt76~a-'
    'Y+z$JSL!cSi*8rAQvL)c@K+Fn^+^PVw_CU+fzj}A3yH@4680C4btlWX2Ki8~uO|-'
    'ei?UCqg7IRKnoNKlXkjv@NS4XTVt}v<)_}Pir+@rd0a~ER{;MxhL=P|OQ{6)QlVk5qLL_1H5hY5C`k*h@=dA`s{ssdU6-'
    '2)XCYxpu8eG69;JFM_ZQ46zAjM!%OffMSAny0h79{LCX6}Z}PW$;?{U%Cu&R?zQdnc(bd-'
    '#xjm{xur|LVz8w2q(WQW86nUs+DH0c+BLFgbh}(c$u*KI;kEBf(z9xVn?E#*QvdG%{H>2m%_teuu=CPFyy-f=V^9vgtir;YlEsD-'
    'n<?YS?aP$R~@V__VDNOtv-'
    '1b(BJ_Sz)&7UQ}o_g_90Ux~Tl?fT{>_i1D128PgEDy=?txoJMH>HdDldS?Oa|Z$PfmibBbT_H#^CiKQH?)u8RLA>RdGPPp;ILO`J'
    'SR$Ky9D}f?e`5dCC!vK1pQf8Y9vhk2XJKTg-2U5PeEDb}kV*G@vi(u-_A$JWrhU*(n_KF!`#~xI0#@sZ$-'
    '9v}DZnOb$wrSl)m#(QzyI|D0vF-+Y=!VjUM>87l_3&1@r??-'
    'WFA5tx+%wMk$wQ3ZVi;`6S^ubsuUqCX(ESi)r}T0vTrO;qVCh4L`2}WR&!?zVV(RW>MFGyX=g_oLy+o>h$7qRIcrx|=lm1g<z!;p'
    '=!Z%UBO-'
    '?q!o|gnT*dzU`)p_LyZT)jH7Fm46E0YE=`9q>uBLz{8im8op0{%*eHK<Q@9CVEFG@;xw?MpdP>Dz#KNqjlO63N1h*5u$A2<@L^J&'
    'E%-JJPu4HR}VEGUM1wtN$;eb!Y7u$EiRt=jHwskHDl;08-'
    'X9b4{M_(CXPtTIMyTFGb~s)xx{86fZ=NW?~2z;|s%E6$lIYn%5N`K)q^Bup9z(#+IX%B-'
    'x@dQsQGbRIbM`EQuzT3bW*cX=IXxoCyX2QJU1*<fWptB>I7j3O-_qrEt~QZAX0-)R}Ge-dCx*;iwX8s5g4NZ<6Eq9uZVMkZp>|v`'
    '%G<H17J<vLwekiEvXyT&6G@)^8VQD02Xz9Z;t<E!yE-'
    'w<Nxs)n!U3KpiNCTQ&=meN8mxguUT_sO8m=#0qdq$n7}}E(k$iB=i1fIVz4Abr{85(-'
    'H6L)!SNfF9;?X@B6R1ZHr4^7Ln2SK^Zq8+XrQ6&yE!74p>zR-}ARt5Cm<OKVtFPGXVsYlMRNA6>fog)Ig;>k`-'
    'u^CqQ_DxaIJ{vRh{%;-*4G3dO)_0?qz7Rn}yWSHgEJL?q98hh(Q8_|B-n_?wm`v-'
    '5PH=uFpUsFk&x_oZ5b`N<QykrQ$+uD6IZhroGV7%A0EY8$}qZZ4a6f>T`p^`Q`LZLX{<Y!FV(ZZ=Yhs|%oTBqEoLT|v#IpRQBMB}'
    '?4DvqWzuS&0q3T>XlqhS2xEKQ3dtRxbaZz7xW>=BZz8zme%v9NytP`kq9UWeA0_`Tg<C=q7w?kl8$_ba_(&IxXX`>{g;#LxVO)9L'
    '%1~8a-s~zkQo?IMW9WFkDTydhD})^(e|mRrg=pTVSM3)r-^@c%PODic{k?AvPZhKoxnp9M6*=0gO`6QXIGc3{i*BcHuPQX)-'
    'srq)MdhiSa#z=FT^izr;IgroKXO&T0tbUvz^rIU<uyc5So_A8+UZS*dv0yd37x4qaR{iIVQaD*EEuolGdm(<U}L{{<8cb+fyE=P1'
    'SxqQ;Wx%ym~x-B@oE@7ZW*mYTC^OjXWK^1v6GE7r~en5<o-{I{Ftx@z4-'
    'NeFs9)!fBqh2fT@3LN5h8*ux!>}wfFmU^t#pn%_?MF8@^R;j=DHyfLbVyXF!>vmqw<tDO`2#XG4LEgajWGfJLj;{Su6Tp2<`M3OL'
    ';zM@S^m#ZFf#e~xFf(Xa0;tXmsKQ&)R~cqjZm{*lm}4S(zmC)=21lW`kX1y~w0lt8OpTGyL)m_~dsIs~l-'
    'U%eHZ{u#8s7FVP1(h{@YbHtybFDd%-w`d9+Y-'
    '(nCh!jEUdy&6Gs}Oe^AU4w(9z50R+<$Jz6GNTWZwjgv&k>P;`k>np`JnJL<)z;RwdbGpV6e?i02fV&Li0RVyoBP^pvL8*4;uh6$C'
    'G1r)}0WFm7}h81u4>OuZq%R^+K{G~9!pw<bP2Bw8vHZ(t(US95^jh=hD5Z-cSfd!-'
    'kjGUuB*bB6e1=_9(+7q>{Q+QxoKV5ABK_BRW&7RZ7aj7Y`&o3&<bqe=iN|3vT?`+h5+yu3n4W#iCk#!PSuPip5N1t2$$vU;BR9V+'
    'QYHrQ-'
    'M;3=tXFZOCQ6$4zxf}pz9Fdw{&s@E%zhL(Qemp4}h@;;95rwKYl$U??#s`y}F)AfBX(+a210*h+3j}N#o{Mv!K#DLauVfE#<wnL0'
    'h2(b{nAkRlLAY1PM_K_Av+uZ5a+Xpy5)T?8*ifDNvA)VNk_7ic4^4DnPZ);fEN|?(OR0D@{L(}k5kqe+d8Go7_qeJ|4qV6ibkXo)'
    '4Vb}%_2U`_=ytQv)W$tA=C37dkXVT%Xi~bPMw;0~#VD8R(o;)*6lcG)gr&DVa~l%&p3x)6{xbc@%QWF9M1{9`f;sXODvGz4mMc5^'
    'd}8D3r62)ROpn2(Og~c-aPVDF0yv1h7KOTLO$gKfsW&TmjpHDKpo-'
    '&Dd^5_tsf+qEpCNQ&6Pyu_>h`!M??3<IRBrUpiOi`9w2$A~Q<h|aDNS&^_P=oJ0>66Ar=9Vstb+-'
    '^$cnt<I|w>)8BZd4WHUJ%B?yg`Hj_>WxL(rFZhVye7axV``bfZdb&75O6gqSVNakpA4Vv+se9ktsD-'
    '9HazrEJti<#a=yP`+FF+a*9C*;_R4f2efG*hqE7l(d#t1@@r`jWNghy_(o8*L<oft!Y!TRZ3dWtqhf?%Y7l$_bNQA;#UnN~L%iSS'
    'd#jW=^<Cf|xr5kTVk8AXX?eZ-'
    'o=KK1n=;v7p9u1|cP@(U+D%_nR0ugF5>uJhpB}h&}rV7hKKup$~vioeK;dvtdqu20`j}dZ;Ag2_A+_Az@xsGjn<3?pG>8l5w?srX'
    'DN(OmazuFb_Tc^3b1PY0&uXZVsX*L_|K0j6$=8C9~`?T<ZG2qS9!d1BHn~7;<2EULD299|`rfM5WObdW<t;oS}<Ru*=x8BN%&5wX'
    'feuhSNWbscRjMXS^<%<G9GhFkXE$gz4_6Sw4zvQVRcJpg_#(YG#LMmuz(A{ne2%Tt};Tm*U|dFb*p?#qa<h3v*08(`>-'
    'Ztu(Zyeg4bj%2Jlk!NAkDS$P{R6Fise-N{c(c7RZz@I8p8YaA&9;h9rFID!{o-'
    'Zi(bApVt#6YEk`kyzbA!tqziM&Yz#v>C=1XlxV&aU?xhl@KY8w~Bq-zu2E5hs3%Zi0KGo9i%EL6dS|TWlaZrel*L$2;(LlhiNuu`'
    '6UMB8difQr>Wl5<~PubXn*-'
    'ME7~oO=!=pp0XE;Xw7^H`B6(^t*U0k&d;P1^YO&izI}G@sFZgVxM_?&10Hx_Fa}Gg^^RJmX$(U<}9fWE{7s$~HKuJh`a4d&iS+2D'
    '8;WG(m+vQ(k_u^66cV&ZM?(czcV#`XR1j-F$6&Qe{MEvi))@Mw(x>E`^4k~q<s*STN-rMR8J}vshq$6>rj>H)qi8k0S3qYAshzYw'
    'IB8}i=A8yZILPI#z-o7mk<99ccD(CUdmWH&OaK8f-'
    'Znm$!68eNqduZjEGSyXI(x}dj7Q5|IAU(J5qr=)2vpBD)aoA5!ioK(S)Iy>~oE5WTsXY0eojG>G+&fF?C3oMwkT5F<%J^XKnr+j&'
    'WzfE2J$IY0IYEF%sA$a<RL--Ze3-'
    'M|uY=CR>FttJ?(9z~JidwvD7Q>MCm6>VL9r@(9Mc9KLl?+w78SoOfU`7OgWAm*YWyyW882^(1s-PXO(g#^9-'
    'jAAq1xDtB!D*<1hx|M!(?P2&bAEL%_)3MtRbW-zU7Zg^v-5E9d;KAt`<#<(93}+Dal-'
    'P#4glNWsZpg@?<<uP0HuwLaVF>Ve)p<`v7;I(a~J%A0hWei~ltO(*S!4l5<GQhR%qxPnn+y1cvvFEQ);l{e^@3z|lFvO&34`-'
    'B{0P6`4N<V=|=W^aqw<KGRhSD!{iniL}IXJ<TGJl!TXLHw)xlEYHTkcqhf+$hOh>{d;6lFN?jbbN%oE9gcpqIhAGnYnGL82-'
    'XMMAd8~9I^4DElyq!4M9b12RJW)lKM_+q<#(+l5(lYnV;-'
    '+*^&QIjpdwdUpArsC$t6azuCEW_9BJ4aiAJfEs!&KvAApf+q{PZ(9w_@@pePvf^z3)kFYFf*0@bFQ5>IVuZr26Y%~}N6^mT@=jf6'
    'Mj8{QbL7)%3IS5FXFt2Ix1HhC$Fh>TgssDTRh<|d#8zb%3>=1&^*is5ZGcv5hTmWQ&Ufyk7CPVYO%)YeQ^1L8O(N{F3<d&FN?biP'
    'p<%GFCNBYwJ&Gfgth3hmLns!04zO{P$bdiKh+rEn`AWL-'
    '<0d*u126rWxkvk}3`aKaYZGp6V<f;b`e=KP`?i`7*tWwvjm1(OH2teg7Bhc|St_`EU0($I?z%IM4d+Y6!g0p-'
    'ZX`y!}d)d+iG4=apqbW+<4({@rfyQ}Hv3A|+Kl4wv9%;n1U%Y?HHL|fZiC2qIy0<}>i(%KvgYipruip4pAU#>mhnJ8xv<8$C<asb'
    '^RkkodX5<t%C23U0czQ`YSauKLo*ErS?5xLC44^MN)>D{D9+#8J_3k52RK@fFppO`yRw?3#M8*$jkB#~*SKFv=l?DVynQ&y-'
    'Me~D>vf}Vj+H?~{B<Gj_#5G&(D7(&ww0GZS_BnEFKmXUC<5}G(8m)}6?sV-HzVjjy|f%=!ENI@)3WEPW}QuSOC+Z!j*W*$;4&}u+'
    '7V3h6^r5JMTFV&5R&9j&Bnw%S_vw8Q#hBS8Qvy8xg@1jJN6FV2w=MDL5^%p-|p`nr1&Sa=}N)&*pyS(7fn0*ot9#-'
    'E_b7X)tcoU=#%-XXiFw1Yf=Z5t`sja^?f-'
    '8DK=DAL9W#sMS`Gh6w1CkFa(B9Vpi26bf%28D?HI5B}ufK|zlBgaes$}eU5(F&nZJkq`h&Qdj)1a~-'
    '^wI2$tV|YNbGKhZ!&8xJqc2KWVNvGCIC8a#8z}e+oy`wpi8J(wj+xA=UYxO0?q3KzqrqE$i(oyI#7_r{?2Z_<di(fQtuHPPveP7o'
    'W{9whH}zCMm6I1?_D0IK9Zqa6%qob`{yUK!){Eff>zXGp7q+KVN^><&K&Y!e?6h3pw%RXQI8TwI%SfdFNDJ7&#j?t0ihRs@6fAQ8'
    '5|2OlWOsr-z53>26@){rCI^e*^(M@BwL(Clo;6#h7Wte0eCA2SaarS=U>2U)3-'
    'Rq?YA$vEC7oro>k_>>`tbM1Nn)~4?X;<l5E+5#Q+iS?@sC`I|5MNrt2|$$`gW6+oP7G8+nFk$erdDTus#3!mX}55Lgbob<=)w9eJ'
    'aK1#0Q4KO^Nlm++V|U<bd4N7Sd@9F+t0vhHhD3O$E2WBpFr524<2Chk!{QGb!Q2_-'
    'n|JlR`HDT${xN_O@5=A&kd)7<ES{l|nlu4LEiK)dM_ZC}92sX>8SE!L;6av*a8$sM|q98xGXFNz)yit?MA)jBdS+cZvygnikH@FT'
    'oaYZmrPLN)FMEb?u@{!vs1_qKmw%8lm3_#pWAf&H&#_E%fq!2*fKkE(Q9p(}sC6W`-'
    'r}KQ`z^;{b_ln>MP0q$WSDCA;jLzTQLjIkcHDY85#q5~g*cb>x;rMent(9AZL@)!JsE2!{0j^pWK*3L=n8nJXx$Mp3PX<kQ87G?R'
    'Y>Z03vOBNS;bEOoo*ukI^Osd%eJP6qC6K-hZQtOiDY?3razvFGu6SRoFY6DN0mv-'
    '&95ZH=?jb^T6?_^cdg(M7$XT8A3$ndH0Cac#C!;OGwPp*>zZ{K<|!j^G^3Avj^9<guy{Sev9$_k@0|*?u)MDkW*BF^68F%fqH>iD'
    'LNyP3grKvT5-f^z^Tv?}!jgn1WuxMq9f#`h#p_nl;<_<&=fS_V1GV(-qFYDD&vaAKr4?KLed-szZ~WZi9oUZ+*I}c?B7C4&VJu;L'
    '$WDjR6)UjJ{&-'
    '7Y5)FMcN30UI0MQ7#;J8+bnyl{QuQ~u?~))HcUStK2`Iy056eVFQQPTq55bj62_VpAgtE9s@D#^qumVCOl+P5I{Ncp+r;D9rKs6-'
    '=U7M^;baf<<l-i!)eFbKoTTSaC=p<T!cBud2CahQT*&Fb>@QCrqfXRH`LtGYl)B06VBqUDtw>8BlqZl&)^LMotUL*-v@+6+E-'
    'H*N=UZ73&9W+#C8|-;feKnIHb~R79?qKEE4|UF$K?z5vivD+s%-'
    'njgCvqDp<InN;7N%(BOUD^s~%3c5mF)vkp$RKF4SWA3jRd_>DT)vEdwxyST8H#i(sM(erbfqQj(Sebp+mVU>yqMG$=NIp|>(H3&5'
    ')7Ne~7M0ZP<eDuz1Mb^lWHGq!vQw6;Y?Pr9W}!c6>5Osa4mpHxhiDO6S;wjq)G3{LD~u?T8!7uqz63%Yy;s4d{2E!Srw>lqm&H+r'
    'dgS0qE-'
    '!vV+`C7aQrzOK3ACMQ1;pLjNzU1XCjrIb30o|@3oCzN#@`dMsRFw!D;nnJ<m8~_BmW!Y%+wvvDO|LXddWl5IYxPO`TS;YJCE!W0m'
    'lx$2Jk(80tlK-x7xC0SYr)SZXm3^wR@)3am91aJS6qs3YLPLI=j~1mHp#kJz>kQ5Z+_wF#QrJ{RG(_0A0y1|Fv20ZJYnG|K-'
    'L)!<XQ=k;Sj?{5$r1S>(9;Ktz%xIDE-HFH<j(cgxOQDf5lhv2)FE=ei2ex`bUGJxKw3`|(#4Eun5%=H^N;P2tGWVbZ}v)8)CPsOM'
    'qQ7^n5tV=(h5GwxHSt%z?U0MuUew)d#r~;nID=vrP<t**xp&Ff6F1UTjj1H33Qpuy*XN}IyW*RrARotgtet0R@bdXk19A~>d}-'
    '6SG`|>9<NoGv$S3DG_->m5qjGq(RjU3p{Vds9o|-&U)*D)>YqvoZ?8a^;PxSDHrBgzssblDrGI^E2W#Rl7uf8-'
    '=|AMqFDLUdWNMTdmNoPmwfQbOlHw68IzFWTJ5iD=?L65jNlBG6PTG?^pqHwcp78F~LmZNZBEYozgLg~@SEGb7UAbBDR7U>n?)1v_'
    'sH0p&#gg*(T@DBJNiFY~R4^)-%Pq<)gXY@QZC;BqxeASFBA35x9>TZzHnnjay<Z`eO~n{wa-(wz3#>~>5<NAVlU+5@cEei-'
    'k!hJ3-'
    '|<GerZQVFN}L80!n*)^pMc8b=1eXtQp30EPO7iv7uK_xoX%C3Nm*MHgR2tG(u1;umup@0Ta0nOB5%RFgW7GlxfL59>egIUX$(577'
    'rjJwVKT)Xgae8i2v<qzOTT@}<=SPx>aVX&8uLRiZv7UF`zJmlf`qIa9Q6xuJH(u6Z-'
    'rjnkDNfS{hRi5?t6OfyIMx3kloW_P2l?WFDb$z=5*@D`jFHlO?NmL(=Hf;tl<m(BLZxHX8hImN`5Cs@$Mxp@AaezcD#s7ei;I#-'
    'zIOBLXo%6a{d+}IG^y+=s`<&Q+dA_rxp_3D%!?MyH6Z4F>9+xb4WDC?#1*PnjfDTs*|5i$bUtqedG}FRG30C?;EKax#jl3%Th&{P'
    'JG{<lkQ(hFH*FYl^c%WJxb0qy7$#z6#}FC$ct*Io8^!?msgQd)%>_5V*QKspz=*uQ~J!)1D=?_#b<~@qijdcTW`O~2y;EGKT)^6Y'
    'R$pc9BYCN-'
    '1@XfYNNiL3|Z3HiZ?QuZf7=`*iPr~Au<(7(~owET>@+mT;nZ?%8t^P@^GuV28K5x>S~5y08*tE%h`CPH?!lZD#h5ovxQiM%FCZ36'
    'HX0O&0|VW?=?0ZLJxp|FkTf>meaN>sl43VO>0N?arrVA^m|morGM6?&%<&O8q_#WO-J*KS1(FipAGnbdJ$^8K2T$-'
    '8o5w~K&$kM(qN>Ohq&Z%P+D(LzQckjXc*5|lAktJvwuS4Ozo|AXltg2T~0I=MXAnhE;oopMBE;tItxFlb0J+rDOZo+=hMI=&H5XL'
    '**W^kBPIB&j#ed`55<uI2DC5FqAtkTB@ds1t4XwJg<M{}`8XpcJ(Rlf#Lpg!_9shsoAQj$`si!s#6<NKV0|`>x@vY?oW|(Plx{(3'
    'Q{#)E?CkvfoGw*$%O8|iaUmR2E}xikXhHA(_Bt#K<S>;~wv^W^5Ah*NEXfylb{1KaO7mz~QRjcl8&c)Kvy?h>TuGzhqb7;==g2mx'
    'oJcFEQj|0#=LG_^-*1|fI%m#4t6r9E)Yyk;T$XoZswiBDem6E}yDu^%XQt&nyI!R{FV>YHc}-HXd)xXXs&I$jOLb$-'
    'PX$~fB&f;r791oY+Sh%|FCsTadj-'
    'V`k?HGyvhL{}E|6c`f9%WAC}1I}h27G4_{M7VB6h<b{m+`lNtFlu+@?aYyh3_%5q>Gdqox=s_h)&0ItMxLYhC)`i6=<OI$NlJr^t'
    'SPG_4Vt$s57tj|$lOcGe|`;^9JbS!RfTh9pY%5;YBiC|`Z@0`s%YCJWKLi%KtF`9~BS(6yW)*VZmuRC*{3j>hH7M%ys*LC!)^RRb'
    '+&<q;b~DLCannBiunoI9+-'
    'c9jbm?az7z)CjyvsO&O_?lGj%kk%tWtJRygF^d|lOAx~EPdPvzchNfeDK8k~dy*N8dcXIVP`6qiO!v;})40UfLN4I!vFu>HFsr16'
    'NW@=i%q>2xv%xUci<XsfyJ|&!;gHy;E86uWbmOu%z2l6>XOvfV{f~KR$E&`jL2OyEr7LftBSwHX&UW2d|B<&p8i(w5YF~RwcsIQS'
    'W&if>-'
    'tQy881{D9k)&O3G3<23fM7Yd+uN~{YJ_%{^#{GooJvr>h?Xdyz5=zCL{l_@Vzm}k)Yr|RRrNPS;QrAnDoGv4ZyBNJ;PglhE%dtR7'
    'i)bh(Pemiu5U?_OzwG;#G=q(b+U+?(qt^LaP9G66-'
    'eL0Fu~%Cw)*C0YO0~Y30HZm9^uU5^S+*!5$fGFKtfUz#t^_)yvkX1>KF*^ArAm3Xg5cve#w96iYSTkC{HZxnd7i4n+2)LmcSrlRC'
    'l^()O($9+}oe0s{m^r_RRM*+i2rgJj!LFr6GcGX#rLPx_m3N$rjs+)2SoEWOOLBFoEKn3Nk%wWIYg{&fhsFcIi>9kwv?&Emqw9(e'
    '9<}&lXW8c43#gegw0TW&XP@Rnt(JCGutE7PD6>hCl~sD977Aik^I&rdQa8ftj(~d~ti*-'
    '6skt^+>O!4@t@XG~q}&y$L0Wdr8RpEUrT?n&bHE0Tz`8C2p`=oe%SM?)KGB3^2E*f%k`{lu^+5DC{^NO`uIppc;A2_OM=e`&Qp_M'
    '^nw%#fVPc0DRv5Uh<s1{9N9{c(m=<B2-'
    '#4jnCf+wu5M>)1i*_d#gg8^c7`RZl5fstiJfbtf+L$WhYV2!1IW4U+whfT0~z=#L0fk+a)eaxqV$P63c{2j$*QYYs)H2J!YV#os_'
    '+)D0Z!DX^{BoA6N_P96eIqQx_m8tB(hW_*0ba3N=IPO$ejscY+S~=o9yYv;j}6Q@dcRW47lZqkVJ}X@`-'
    '=W(t{V)SQd%mWyu?uIBnEgJ)&E^`(~g^0G$U`0Qu$Dh<ML3#2NBrz_Ckt;&7}F{mEoeHKR_^RwTXo`Y8avdeR@C#as|Vu^`04Gh~'
    '@dt0RwGj5h)7Ekrg>bGls#1s9JlX!aTq4-(Lh1&`OBXs^r9L(0%aUW71Q7?k9?vj0ny1~f=A?V_;)g$n(9Nr+q7us-Ac8qF-c-vb'
    '&M#h)pV}#ijh^6BH${_%FKTxCI=37W6G9NEfRTPv`1t|22g-8L*oeQemckiv3Xxl={C<l7CD815rw&i<?b*tVo?{D9MLNwm@dZ-'
    'I<Cr>7QDMqJ9987D6@G#b?yJG`zAaeM9f3y^!#$i5Z@RPm9k@6RH?C6AifoDJ~=}Rho*8Sms)%jFXtQ%g}?2X+^ebg%RmsH7OBoy'
    ';SV5_9t>U7x*?|)rxeX9n;enqF#5;pixWJ=ETjm9oSD=qQt!0l|WPK8(={Wce24#Jgz5!ya7k$^|r&C%fwC@6UpPY7trW2X>vt6o'
    'M$PugA3yq?yo@K)-V2(6R(DX%pBXBAWLj}F6VzBE2Sw)`R1l0?+=z`h3}he5C+M)2vhQ~4+S532wBa3ZRIrfl&pQS{JvbcfluIC'
    '--2e`-qYt64w#eDY0R>Rv865mf;Xw-icO!-'
    'cgCo*6F*l+w$bc}u70+969L&X<SV70h8)5&=fEmv{mf*2|X}WJ@!^rB%5T(#n|8cgGvcu3HGfTcncQQe4r}3oZ9styF%)9h3RGg1'
    '*xN(`JBaS6FI!X4kr1hc@q`k;|~BXBx_>+(NytQh<uef5S%6YeX}VI;@kV{NXDdi#B%iIBkawK=t59M<8W=y%&W=Ox_2w_1mrw@&'
    'w(5!dQqvc&aJhGZeYWp=fXKK0|u97q=!}+w^9)UN`Ctzx^4edC<^3X}jIIn3VJz8v4zh@2EPiZi<kzs}By{o%&K9d8;D!?ANVEe#'
    '`|;HRmh<19TdFile?66wUUVqM1{DDmD`-f$HH^s0ml0#-'
    'OtH@~5(P1D5ieWE?LyL<T$uQI*YdSamG0>sf7lme2HxMtX2pemA<Rp&r&>G>Qghi?G4i309Hn02XNYFL{+tdE~NGFBUy>+IZPkGe'
    'zx=Ikhh}W{>X{CoUk7)k;S$<W^ssDhZDKC^s%;Nt<e0Mu;^(sgC&}esRliXv0T9H87d=RXQRgpsuF7LwUjIb?9BH%OgO86@NW-'
    ';;dKmj<e#I?fg-Uy*M?Z(qxKiP7;MTQcC?&#c`qOl?3&$Wj#29xazL0H%9p^lX51nr!#-'
    'cOpM{0N#fm%x6i54va|n6g{j=&Q{TvWy;uJaZ2l5OPBj(RMH28T4*OLGnDp1ZPY`Py3{FBLL9=CLI&injmqwOpOkeDHlZa__!CUf'
    '6Tjz4_&U1_eVt(-VJ?7h=08FdoPkvF?{UM+x_1ldMj+U2-x)<*s=uEf}07gkjd$R*o9qv{7#WXD%jD<elqe}oEJUwN-'
    'a%zGDtOj<K<f8{#O%#b^vd6ghS7>%o6AD!=p)OWmW`1t3WZ>gokNhjvSFekp--!ZHt|ECfrjD32lNaO)-kT-'
    'm<tM7~%&xjYrGa`@1reAUNt5coPz4%6H|k{+T&~L73c0HdUm`rZlnKZVwJ9m2X~|QldlvUCHv<_BBWFXGTfVFYEyHp8=6IjC8mUY'
    'qnOCJ&4CnqVce_Pcm-ir%C4uj*iYUc%GX0!b<qORE`Ly#bYViFG^2^3e{p054w;+vkmLdIOgHL9>eJYH$KZJ-z_qARzR{8Gr4buJ'
    'kty)5`LBQ`>-IIRm>16~P_KI!!@;;;VY!yksYQ-(TT4600=(C<Emr6eOt(jP^@Pelw(>uhwAL?!Xh-'
    '1^dj#t&RF#cimr?})pSjbm9MI&2t{Erdd9mv%YpZ!%Q1i$3BdS0UD#`boAyFc{}WvJ)8KQLm6Gi*K~bPt_&x}5|Q761H^v8mkr3H'
    '|Nz^l+nMwLc4|l&<!CqP8!}?qYxt&6}met9eqSH|>5c6Q6Qd$9gE~eiyW)rs#tA6d$S|4f<@$m&){|A~mDc$Jqiz(XxA$q>$ZSoN'
    '`8TbR1XmIQoO%_LP+<-'
    '6OvN$T*8XL{*sfAtu3xyd@LQ@<j(ed{XS?@Yd|*!PI*WoX~GezLnEes52=n&L>iBo2S)!YcH)tNK4`Um|iQ~Zry;5L#*;7F|Qh>4'
    'eQztaA4p1B6maAh`!7Se3@DnfG;Gyc@<@ii(X7C42O^BEuHP{Eq0XaV5?GyuRHl#*ZV=sasJEr;J>^&kouWDzgVQ^Wi+2ZOLY?^d'
    '6%A=9qghD_N-=f!9<Vq-XX9T^|2S?$nq>yg}S_pY(-hxEvzU{r`?^}KT{XqA3Sc!09~`P-!|t-shen`n*-'
    'nn*J4CBS|PdZl?<acnSyuMaKjx5Prb-'
    '4ZEuVSbhmyhNeQIv7uXQdG1WhbPOa{wZhG&*>fj=;pU&;kPU@ws2a9QJL%`qak6H;M#3=Ju>-'
    'Ow^&i*KJW0{R^*7?Ex(I(Ry_`Nz!H9u8!A60~imGdKFp!)k@e5@4B;M5Ctwh>LtLu1R3@bU_9<$iIdO)SY08^I2l<<_DADgID0H~'
    '~(7JVUXgT}79ya9e{&dpGY})Tms3Rp0-'
    '(XaOosua01EQUMEL2AXA;%6tk1LS41bC$1C7U@PSDLSh!F^u`#Z?h%yc{fHDGjxVX>Kn)oUD##?g_L%Dyg6#7VH=u5M75f92O!~~'
    '``3+_FJ<zu`5S^xX-JMDuT$NhMoKM=>A?r56VGC%7P%$-p7g1~P@!t0h?=3C)s#dW=Vmir;Rz-ovteL_dKf?<&k{l!_5S080N0jl'
    '-7>S{IVux$mR2e*%1uf-@-'
    'iA2Edp?@5(NiVVzch@8jN@XJw5!mwAg{U9O&edTuvxriqF*ZYGqHwnl$O#lf^J#%?rG`zyMT4f9D4MdMY%IrujLhB-bzR(S2LPl-'
    'Z2Q?^BZ9*oX03jR*XSvt!uV0d)d{m$biM<DezVwsvr3A&P-CtW6B@w<@~gbXm4Dh7|Bzc-'
    'e}ijNEIcyRbfo%I0<1bM{1Sj|MCAw6xw;pKsrN|5cH(m+!|fE^`Zia{Sv%5Wc=ce>aNau<Jyb2i1O_$pZXS?-'
    'blKuwT;ADig&hg@R)f?RXc2FU-'
    'VA}z+?5&6Y|flw6<wT!i#()*n9ayU&tlexc4uYV2Fswb6@-=h*N!kH|>Gfot|)gsqpxc$LE@lO$^3IWP%oLU-zU37llfe-'
    '+ma9Wck(d)|s@?o`4_`Y+z|Ngof*v&~Q<;XL+kVWx9L$4xK}Ka6g<hW=GK4#koNjdu~M>ljsrI*Ud05Zv~dq(;n*WB$CMfc6r^b#'
    '<{LRHH)u7^YX>a;^W@<XsNHu`P5(c^sIYb5G836K~f%%j%6?2N7>a^UGz?Aa1}@JIhjw}ihtDW`#o+W(&w53`_;5*<UB@;=6XXDi'
    'p>&dSY9u`DsN|fB>%h`WIh0-'
    '`65r*3JYORbR@FQ8uOnp=vfaQ1^&6;MxP+)=V#tWBkB6l6@|)EEyiK{(>Szr@VsuNiHp2qIHGR27*3jDO~TzmshQ0XznLsDa*-'
    'sKthMC5a_UFNJ_ZKfKgE99;0{kL^jG*_bt9U<j0#{pT$--`mNa2-'
    'Hk~T9A2vFZsI7(olqA2h*5CfBt#XK`@RTR{%5qq%RKP+VcRi++s}Xy_vg~~lBqH7K@-P-#+>->_-'
    'qnUhdr;8}XS2LH5^}n$3XyaFDnz!7ak<^uPvqS0r6UwAUzNa{E$CHmlAon6iyO*_Qz7Cd7S}gZG5_R<0^2Ens%ycq^RH?SPNdCF-'
    '=V|YmP2sgH6^32S7{^V@ixDsJoA~PZXKBla(9W3M9>)pBN5fCuc{Iw`(G$zvjof&Y;UvjP`6wWR{$28zq?V;IdDn5{MCTV#jd>KJ'
    '2Zj&Z2<(hhN@>aEYcj*b!=z?^uB16<gaAtBA9eZ$0e;`%&9Ada@5Vo!hUmvWNNo|%ODn8WjPS)kyxZNmjv-5x4H7o^AGP$qr_3(-'
    '{B|3q@M0|Rm#oj;RCnzsQrC*P@q0yAHl4<+RQa8BQ1ZC&z|L_<MAnL)V1tDHWBHTk4Tr{sOZ5X$OfA9Y#ixl!BPfFXQ}<8<88dUx'
    '~Gp;x~tpSuyn<o_99@20E{$rBo^O@kvdf872Ny5^euhUUgi|jmJ|&^K$mV8&~URxNx%G2>gF7Gw7P8M{*$7X+28&+POKmio*`_{;'
    '*7ZhNoD7cR20e4Bpln}Pk*zl$2;bTu<4md+WRl5RzADo!Ct*&A$Uglk-teKe7YV==EI>R`KH8J7egXn-'
    '%{KcZ^y@aV92Wqd6z0tb9X59M_CQN+Q%X<2CiDR5k355fT@C-'
    '{U3UN+E<Mac`vJ0Wy#bz#r<>MyFW;6q3o!A$yW<xH})T`QPs<GqE3kpie3?i*=%*G=|ysHc0GAAw&+*59Jwb`r$-'
    'cu7}QaMDvorwri9zm8@{#CUO`xQ`rFgojdYJQqiPai(KWZ<7U?VxI*vpHSotNSNmE1#?Z?0{v%(~1|43p^_iEF(wy;D_-'
    'OC`h;{qkwArdVH%p|)uk(X}aVO-zdmKudq&2=Xw64SiXc@Wm})y%UHd_~m*o<w{4DM+=>NqN#8?a_2^KK{@czSj7WY?;!DryO^wJ'
    'sIm)>=pjb^_wWm5ln|pic5ZLiY`0crtNxQucU~~=S)e@682F*`l%+Mc2UtZF5iLIB)j?%vP}|X>shwM@LTmua0I}G%iMD{?DcklD'
    'w2tId(6cC^ooT1bSmzGMCI5=gV7btnZ+7>t0|HrRMi4Wc#uNR5$yzSpV~9F=6lCA_iKKgbr{QW)$@v1^n1LOucHD#MrO`?!cyU+u'
    '84lh%`A0E=X>np5pjEYLDZsm@~*i4f#Femc{zlizaadj=96!cde5I#fuc(At}T}(Df4a11$1yd(W`z*t>~>_<u*Xa7|M^L0p<9+v'
    'l;TEB1WU7c%rGX!O%2SlHsgIlx@$`y@16b2$qGn@16UWO^jW?7~CkV6qTH^Jd~A<;-LJh>(Er4$fvsmZ*-'
    'I++W6twZG}@_;=f!`@G-RP+<44!Bz?ufX5F4#_ge<KKI}lq<gX|pw{bULDz#VI@$PKrZizBdZ-'
    '<iYB3Jogb9z1=l`A)AUtXO<>6)?lr{;bOj)tTw8TzhT8EvYAo>$C=zAB6S9V#_9q4unf?j<zaL;zkQ07r|h^gvRKx$7}yD(MlU)n'
    '^C00|nWZ8ABzg;kCD4#x1eF*}0`<tf%-Apx|}8ssuidOXGH9z2I#n0aL9$Vk{);dqwwy`Zl?q>E)1VDvsp->lIS?RnLS-'
    'ekiLdRtghHP5kif-Kn)kviNhpSYm#}xMkkhc?<M#hc(K>xKCt@yGo@O!+0>ff06JQ)ttk*)P2)QNs&asiHcAwOa7lsIam<<#h>Zr'
    'gXIPADNW(}o|%#ZQ*B(@F6mG7ThOFbgd0`%y?c-H>7_&KX4k4v|5)ljygWtO-'
    '6Jje{UW|j??=E}!KD8rDk5c;>`8@n4LLi*`=Du81FaO%X<V+78U5z+u1R=d^SY)oZpW0NAM^mi<twT<z)SZ+H9Df2q$h}vhLBO!w'
    '%JOGGg>X89Ydywpo=by@yAxTRa}tbTEXLksm>f5rPQ3~txBn=2QE(~HVGNW2x`#f`IswkzVw`wGg^JTf*=n7llNzt!x#@}M-|-'
    '@bk-=XrO4iJy27JNoW1b8hU(QV^B`Dm`T2g0bWs<UxuPp<zbT0`Af6I)aBrd(f~9>%Y!?NNz3#Evgz{RE5{Dq>CGL-eFw)aE_ezZ'
    'cXsmboVs196IcZ@~|8L22=23q69mM2iq!x7tPGF~OXbBY|&cOMs+0?W4^gnf&TXnq$M?jr()&D3b32!hz<pT2~;!!FpeN&x4d>gk'
    '>W~3EKj`1ot^4vWhBW>=isJc_lWh~W6H%WN7$fquvkX|>?>sQ5Tfd6d7>M>+h4C(3P_F8K)z@lJK0_<Lrvp$t9y{%hoSEeFg+csa'
    'hDHdrdIQNx0g}5Pbzx74>_qyTyTiZ0TWMFvBpQo)uS(BpC6hIfP(KX|8*CO|}<*ZNNy;Z&uVNAmm-'
    'SD1jD`{$(8J&cQoLQXMJ=0mX$?23X2k;P57n^Frbh!kTzB|OP-Qtg?pe0AqJ(7#RkS7of_oHTC<(bt_{e<0~M$J)_@8S0)B`@meY'
    'H3;CS-'
    '>($K@K8=$=UKG$DE<(6cKjYc0Wr3h%EM}V|uKUi{@`1I#S;Bl=fhxSU=eQ<f7;1$qjJ(znkzN317Z@nRW@^l*mf=)<0Ty*>}*u^y'
    'w&0(*Iy=nCBD3Oom~k5B4EWv1nq9QdKk;g`{KVuSbXBH@;Nw5%%Pde$=Y539%Oy2|G9^I<+&}mamjQS41-'
    '9^e#<FY)F4R?bQ||jZB_^kpY%yrqLj!d^AB;2&#bB8MUfW>uLs6f*{nSRpX+K>B<qo{MSdx29fqnS7KD<hqsdyd3$5H$zp^OL0o-'
    '$6Elkf61cL2W+tNBn_@OkFwPIb1nR_)paHn{N^83YF!2Xp9i^^YC7YLnHSB{b1wswWpB%~e>#*ZDXC!oq;O|&(n9gT6i=XMjw|Tu'
    'q|8DQu&a%mCda2>A)|Efl$D`kB0Vu9g*G|)%Px?&QEVgiWn$1%2Pn6b_yUZ{zb)3iT4Pe%z$W~Z`(c~33DFQp0KbtR$Fx@qEY^mi'
    'kn$+zmsyWhYg9%wE_EJgOXT^d@oUVaezPdW2kl<sPBGmih(vwEv954u4IX^>b3+@ptN+37Aa!KP`@z}%pz1?SSQfbL)dwE~_X0n%'
    '}3e4$cs82d-r}QULCeT^S2c4A&dViWF(27J?uJZIS>GK!r3_jxu;aj+*D_qiwy#1qkM8-8-'
    'URu|7sp*|hYdg9esrWu0ba_{YDGWQjt-n%xs&l6pbCN*>HKTv|-'
    'MFFcOLB458niK(CE47d>E;hf+lOkOy|d5T_~~48{nXuOszL}@`yeEB@)>ed^RmuLEgh#j4EJ<&RJSxDhFfalm)$m>lEpMg+9{rnx'
    'HWojbw53z=nVee9Zg~tu@vXO&skpf?cBFhI54rt9ZKzKYXnBw_L?LAbgM5XZ0%p<_t8hbHg(q;V!_yDPoCk+mFV|sb=q9x&FObli'
    'HP?1|Ni;g|N8fT{lEYIpZ_@j@!78AeAYUZPY~S}&cgJZ=R6J~Y!Uvrf2P&&Au2}3XQB!l*M!A$p4&T3>F}?-'
    'xof`sU+Y+0K(XiQ0gXeib4~yp_@f#4vuFG#%U9!=nBi6|jXzBz75-(-'
    'Nn&VOCiw(j$IDt3Ibz&;G+e?rdbhK%+%Y063?jhG05*PBhCM%DJ<kItxlGgZJh~Ia)rLRg)2rKuO+olOrGRmFzTXCa+8*;?<LqTz'
    'w%w0&vvBgm$cj9%g_}F<GFRbet*ygZL{`y}mYvzz@_wyipT==)f7*jsXUg-;bK}{4%(-'
    'O`3x`N{H@RE<R@xC#))7*%2c2=JjscsQH~Oj6<b$yrdnTma<(740w;dd}>ip}~E~e^-FmC@kQ6L{Etlo*uaRW|_Z%-'
    '#@J*%z3WCJGIqtGa9{8nomT$WBoxM{oF{(2sAx7}>!@O*kKjR-sOvl!a>7^C_8wQl{DH)7gDnx_8r*O~n`6zA4meN(SR7j9=}+C`'
    'VRi1}k<2s32i&L_^Fs;lDo<2^UoAT&|ejIn#Zx7{a0PH>jP&yv%!@j%(q<LKS3acwuU@ZOcqNwa+>W;+yUiFPPpKd5lD3!aFBD46'
    ';%AaIS)55-eDkCX#>+3{+&VZ8NZXaSD%swcCjn-'
    '7eHQm?(hjIl|Cr)qP%@Vp9t#+(`aOtSrKnGtJWVV4L8^!#kuKVzRkIJRB$;H~6=SJP+Lk_kDZscbvL)iBC12)jlLpLn)dgCh-NF#'
    'B!iIaRwmFc3DDks-'
    '1vHEHy?i>EMuD!)tHa{gX?4}tZdq3D|4W;i(T&l&QlFj6+Gs@pu72K=>Eze~CEK2*OCrE%(=&>2>F`PH2HRb(#vt-iEYYUlgt@5S'
    '%WuySYK+v7Y#@|-i^-1E-'
    'v4#y|%ZqahEG6oN){pRhkb@wkbNA@?4nj~wuyHDA3*Z1UrJ#NN5v&Jv%?28Kyj%+HFOF=!}+3(shF6OC`oo&y&=UBH+Tw8aXC;n&'
    ';s<xrW$R*}1n;Nt>xsFxoZeBR~W_JTJV}Zg5vR_`DUd8O@9RBPU7b9O0%#58`()HD;BNsn1e!0*t;UjSoc7?oNKek=|u_((|tD>{'
    'AKd+)gPd_yF3+%l;J)0AG4;!A%Y`~U@*c!*p<|QBbbn7}T?uXm284qrPJ>@fCdoTV-'
    '3^1|#WR~{)hFMB3w&#gg4$S5YT=I!!ro>Em>26i}ogc>*wx(%jrzh*CiEj)1nORvkh2yo#!kXzDS&OJWj58qo9BL0c2{UJtvUh8;'
    'gdbkAdU~>23tnMwTCg>mKi%b12>>>}Q2QBF6@8J<JG%q5%K0!0Z)a)?MT>`Rwxn|jJ%<0Rr^63>%<rJRu6y6$PQzyMyE#^Jn}@W>'
    'nPa;)u8lC2Sa|kMZl+9D9H(5)+1GM@4+*DxckLwOG1-{v-5-T5``v5@bXRFR_@s{G&>tsa=$_Ll$sAjnc#}g-'
    'D?&5R!8}1sh7|Mr+*}(?6J3Ec*sG6Gdz7>k35%(@Uh+EJy-'
    'K^eR|_}9jM}7SDsmpMHrnWaZ3XNv{XiFi!GfRM+u#?bEkzSg!(26vBl(<t)Vp5da9gX7oWdNRg>zwxe5+aZl}mNjDle}`_HS(6>@'
    'gqLn0KkVNf`EoQqe3kr*lKr6BCpUY0ASL!sK7pQt0DV6PVLuRNZ3xEpz%{LKdA{4!PtG10smw0~;Tm%e^C8C2ciXrMT1EVx=~RAT'
    '>?F{REcRg*-o1>3y^%f5WW9-'
    'U>6OUoK38S>p2>^02vh*}{3ouZ_bz=m4ETY9?>_N*03$qu!=@X*_C=xknFY`Nd*NC%M$mY_Pkrdv=3#%|E`~Vu3qfFAS3CnVyOB_'
    '=lLs763FBHe#sAm&R8uqkPP=8}N1wo85T1`VR$`ZU846JSZQA^z}6`WaX}_XcmF@W_KSIyvx%csl_euJ7`Zjm!WFj1^oMP8Ei<~t'
    'xbw3O_7w&<1q&os>!Rvy2U`T#5tP%1SrmUGtt+a9we^r*HEDD<vZ={_#m5)onLzXJQwqkjv@G|shWe?+&&-'
    'sAv>iLSvKXosiqE>{rBuC_m~3q@Hndn|F~uO6=W?BJ?viF?OEHeSka|Tm921a)@P=)>x<i?@hHva=umX~bX=eV7m-'
    'fR?q}MnG8DdIY%=;x(+_QnjM*8p#<YliD|j8n9D`;KHKOKn19<b`?Yib!LTEr0^uHP+Z7cGw?q^Lm;T`Ck2ikZ7W`0)pt4MKCSWM'
    '+0&-~2EqE+kpKXYZ^<T2W6<|rh&IXq&cTYBD9o{GiM8h1lJhMa9&D~)WO{&!B}c}D}%DK<(Kca^FgOiA-'
    'q%(pVLWYcbIUFHguQ`2BbHZ<tN8yG-rVGN&>;{(}X`<(Dq>?@p3MJw^*iOBD!HE14A+YCS7a*ub%l5XL@`MZYaoW-'
    '!W8*Q22t*GFs@Z=OwxQ4;0(Se}LMu)|P%kOW|rB5G;LpIc0N1i$7RyEfHa$THF@}W=zXWz7k+_V~=%UGITW_*hK%QkniuWuLvv=z'
    'qyMSN!FVk*j?91-'
    '*wD{ph(ZVrcr<Fke{THjjVhI<Prxy~PRmd}lLZTi#jm?3GrrMq_FI?Nh32c66gR*#D4F4H506S@_@2^*8&?e2hChplnVS!My121V'
    '5zG}OldIx~NnQ9SHOwpl&3#7#R01{Ogce6YgpMQ0g!71fsmtitBm<SI~u>&o!^I?|0S8;}KCm-'
    'E(L$^5wBlTEu<vQ}EoI~&Yqp*g#<)}cy|_)G=BtUuJfKhHl0P7#hP4QKgW5X^t4(ZA$m!<o2p!h10P#O^;~08Eu!1M#tOUme6U=f'
    'E*s+Mr5Ei{_B4D6FGBN8vY)_Kq7v!W@}f%}Ib?VKfUDN~0^;Gf|*u=@@{#Zs&XT)eDFsZ-Y>cRecwupwng!0Zg<{)v-'
    '^r3I>`HQQHwl*#)dX3HDiWiUQpe6g@`79_gF`&Tu@Y82N;!n;f;~E)6I2{drt-T6bvU7WKz&n0+d7Pt38-'
    'cjy@c#(T1xJ0`r?6ftV?fXEEwbkM~7qt(LMJ)VYaTX&pw`nW_WUflzD{0<wraJn4^ahCzkcMn8h*+Mnla|RgM-'
    '6=7nWo}v7f;4AfV6YNIm3x?!6zKy)3#ZUk7+?!_P7tC?pXNm{AjXYu$zd>aG3teAaDI)RzJixr1X{G_7N-'
    '#N6%^0N0%0d33;XHe$hhVJ_ZC<8rFhNG!fE9{c`{DGdDtVVF@G{NWp2o1>)B17oRK-'
    'Z11IJf;%90)*N3K=KTs{^NYKK@(3rllU2u<O#E_!o-'
    'he&7<+F*H?sf+g7;K;i`KM|y%N58oJTWt|?kSmJ!=Hxx?PgzVcT8UxdUHrwiyrNJV!U^3lUqQgPTC^`+355&Uzl1fOq_H(&_ZA(y'
    'm+&+C*b;#e{x2ouk#$*OQBMyljW<=-'
    'T^FQiY6O)e1sO*PM`c3z)OpsE_B910>`TPI$>W_$o&=%wsD>2^kX`*0jLZBDn2cr=z>9TCcxk;_^uAn5S7YuHaTZ<aqp+zfESl$$'
    'R*LNGvKbT3ut#fnF`DFMXZXrXLN<4az*5nFSn4zks$38QJsFDvZE$iIP`J<dZwr6k#1u2?GU%EBWG3$GFkFat54UKFOFfsZjB`X$'
    'RcBbKqkzSBk*ZhP%?`;WKr$Bj~RA|OV}?WMc!{t!&Q9UU4f0l_f;(N<i!*Kl<s4(bLFl+kt=Zr_tgzRVsK=rdU6OI)~0ye4TOhi#'
    ')&?jY$t8D=dy_2og3uN_PF7Md5PIwE%-'
    '<+t^{b^b=b@j(4Q+FnH%&}ilAIRt|Uib$yS}G?&hJ!5JsfZ0)VuX#13rE?9SGdq60#dxXbrqB$?d><Z$i+Wpr`bQTrZYBy5B`QR!'
    'j5WTs6WY<v<776H3KW{#jm)MWIw(tTA}KW!^A@vjPetBMwq-A!WTunteM1K0Q%w|=JOcGJgrS9y<>9JG?p?o*L20=-'
    'oW_NJ#$&ln-'
    '#l#*i;)2u~x410Ux0?&n@zXG?hXeEbixXA~agDE^gk!tZsRk6uVcQ7)#SgH9i;1gQDNYi@y?Qc*1!UmBqH+CHcJ?Hb-'
    '>T$=BWo*jD^P=r0wJOkLx9|-NM$9+iYf-d-!{V`A=Zy68B%sd0ZcyjG&V%`M$?+a@ZJF_{s8DL)-'
    'dJd8E3*weW`Mfq3GNgIv^<o<WXN0OR9VWhUW;nOV0yS?jzupYu`ugjrWSn*X6NJhLf_HOb7zHMDKd2?+mkO}z;0&_;1L6~Mot&wb'
    'Gb!SN_TuX)XX?2{VZf|MOwh}#k3tmlcBp2IuA2n@YK?Dd@vriEG`O6^LU&>Z_tJo?~dh)nS`1xvY6v0lPziGCtaSCZg-'
    'I~>qt{ys(Xbzo8rJojTA1hn7Bnq5=3N!$GUmk<=&7zh^ee_b~+0VqMRtrbus^81tG%$Xv->g-'
    'CrZtL^|7Ipyb(jLow`cz6{%>0YIfVr^ZcW%IT&T@O4{e>GEU5O%H--waewk*lZ=ralYtxe7oVcZ!-'
    '?7p3vjytg_X^jD8&=9Vp64LaT{Dxo;1zfj;*LImIcY<8dH`tgWE^^gs*De@Q`Su}9ReaknUK1rCB|8`1N*qjDt`&>dHlKjctjqT8'
    'f&&rbK2;=7TPwtJ}jnD|j+#KLFxb~^>iiVx#3mPy|My%?Wi$AmHfe|rv3W>yfH-'
    '+KBVCPQT&jLr2tHFZYz7+(pl>|qJ)1XH9r{I(sq3Y@2L?W*0y8o{bL%=_LsV_G~uzCG6xNoPwwM9{FoxQ~p(S?bMYP;hj{P-'
    'QZTkuLyyClO&TW(86`0;%m`3~oBDsxNXx3t<Oo_*(tImWg|Fg8958eP+o!zt8FXbnk&_*o^ax9KG-CCkbB3@)buIcUI`|0Q5EnN_'
    'AMdMy3RjM^JjmQmTOq*Z}A;*ey@6-KDf7q7gp&(#qLA8idauzi5#>=V<jg>NnZ^-'
    'T;^hf|yS&aJ1)}Uov~2F0;%mM<7>3;I)Wpid;y;2Ix)T8FbCqY)v=O%&@0;o`m|rr*NxOFW_m;lV-'
    '1{KxYq&Xy}oz^jtQT|89~0?#|GnoqFhxh_+}&w4l!+<f#bWC;{aowZ1jj%@e<g+%xB#yZ4fj=eTj5#S}_PJgHAd)EtPf5x{Yp?-'
    '(&=>q|g_HfkI^1#D4dUdz*#fk8_UmKgunGff<QEy2Jub8E?@F_J@nB4s)pvsauA%w+YPfCU#RI575iC9-'
    '#U9L?&^sHTtO9pZ+~uyD2%rQp?B`l%}Z65>g|-*ni?L-'
    'O>1uQ$D72izhE2*GV}E#|fK;1VEat8r$QOQwgs6blj&a`12wh^8lrU1nC@j@!5xPQ>LNE>a`izUw!fknf^>U|N}M8Us_Z6Y1=foO'
    '#20*F4BvPmn44jiMWZ`^z7mUH&YAhza9Ux<Av#+G=IDzs19`^|H|gw#2kq!Rk!WMLv#K{sN)Uqv=nyZ4bBk(&%7)q-'
    '~n`o8m+Tzrg&c8SG2U_|?Kg29qh$b~ZoC<v^Lg+1;jb04gcq_i)?=*_PDEu$u?SzdCf1Y-YCXyb4b|4Zi|Uf-'
    'A%iewqD};ddhozq1IoZ7mtQT$Zdsh9i7!P-Y@(v8yL2=k%~S80|XVlSl+K6x32IE3hJ<h+Ntn-AVM{rpNFtAT+Ea*ZrWK9dxp{^D'
    'aC@ReEBE%LMGw_ZWod%d|0!I$c1(Odj(OPR?xZ71^`r&vd`bgu82ug(0Vo^O7e;bQzA0%{AjNqjcZ_&Oi*O^XN+y`I@uJky$skx1'
    'X(HM;kmA#w1yHr!*yfnwyJ@;KAJ}5fa(DY6;Bleu~x`8%K7|PRX`(0PWCpOv;?k>ZH;+k_YYr!Q>;l`6hpLHGgQ8OT3Sr=Gk$K_A'
    'ZFu5Qj-ZyqMP|2CtFR+Qm66H`#SihC@T{v9Y07BpECW^0fT}jL|FwFOFuMw=V_YBTU0xa9e7c2Ps>;jOjCz9y(~C?p#cd0TIgu_0'
    'eb}J(z14Aws-sdRA11)yvJmuJH)$COS3RNF?j7clKGC5RY8N+*0OdU!=yIffiR~gkjODW8~Xyo(`_cq|BtuZTSeb=n&HroF3oWb1'
    'd1@eqH2dr&T5o<2fvudaK<-()rLL5$6-'
    'xk!hLaqRyhcU<v=r7Wbebzy)az^Ad9N_}l_IT}62)4b=~9p?7imijAjmV7X$A;N4mzEJ{ugXVSfb<8D<JnGE5EL`yOjB+@~Ob7IZ'
    'MUKb&=m118$h&{W069&T)1}#nv*G5!KzeP3)S6cgHca&C(ln*JG7V}z+ObNT{zA5=(vAkQp+oLo-&6eUe8~e9-'
    'Mwm~lIW{R;01anPDpRO_D~ajMFr%P%_F%N(?<M;mv6e9br0AGAYY!_6qZUY`h-'
    'd0KIO8A>A0q&{Q^3jK**j6gxDO+^l@^r@pW4XOwaeNuN@q1x)j|4OL}ZujsV5u4+#`5qIU>L9(-M6V<YG^&s890gNRh?E-'
    'HWL#a9bMBPaCBwCMEBzQ&=ql*FLZ>n!<W%t!CSz2*hICAiJ=r#U8S9)+RQIw6PaKtR9Q9xk}wf|M?-'
    'JFxhEYC74*&A|Svo`}D<1?8a5jsugx&J2q&e!{-==l_%a-'
    'd$KTHwO0hUgEDS(R2lLGBaWgfyjL9rng+!G5s^4|C;pFN@Yrw_&TFw(!KLt)?*Ey2frQ@Ac?HsyIKt!q@HCT1&B>UHX~AeG&g%HT'
    '#{a7y|JUu6<#q`y{$pk*Ud_Y6c0T{N|Ni%%zx|(o{MUc|<3Ik#fBeh8{qKMLC(8WJ?b|A=lKTflmc|ckH?ogWLPmG!3}T`yi1bM}'
    'Frq6>G-'
    'ef{Z}7LwOicXM(7Oo>Kp+6Hs6CE_ggZ0wuosN!&XiTQ<cwAhykk^E!r5ookoCCFuhCW9kbg?bduD6gMr`EV@=Q=1N~Y&MQ?V8@i|'
    'AKjplKnpk!_R{>qg|}E$P4=1|f}M5F{g{0SbVkrOEep%!%wTy_IZ^7ahPnaUVIN@`U68>o@W;F&a+jg|f@V%7SGBo5bU7?|IG}tf'
    'E8`8R|+3F%igwVx|*FMF$SX6(zwXpo4%mWY&v}@9`NsA2izqSz(Zi*C4i|#|%a<x%Bx=mXet~Ey)&;O|;0;8fL7)87|BmBQyD<^r'
    'Xe^s!Bj#Dsa_FYpHn8g}XI-zlb2hI!}HE=g*ufitT<>ac8BCVnsp!_J<<4=qYV^_C)s!?og2{Hu2s*-*JsyB!X_h9m5-'
    '(KMDK=iSg1EIZqS7wf~3mPZ=!LLz!$b{Vu$RViu>z!2B4jWf&%&YxWqr8+gyp;JOBnKXCpm_Bk7SPj&}ZV`WoLSbg5}8O>v83M)_'
    'w_%dZW8)V}>{~wm0NrW=-ENG|8aUKEd+Goxh3*U&0DNvUQ3!y0;$fovNC)7i|nw>K^qGhQL3#$^bl(Vs#Ec<`V(L%`aI1-'
    '>k+uST5nG{ah0{E?vvIBM5pTb(dv%z-'
    '>*MLu5nA0s7|ApaQ*%n4|Y#<;h<j59q91DvU3L@`97#v+6vCtZ_kP4RBLV)<3bLCv9aE2@ALQVKQImi_m#meBXBHxD>dq41*!Oxs'
    'AQyF+y1dNr1JhdeFY&K@j2H+u&39{>X8GOHx9p`6|>U4q#y4$np;JAGmz90X8<NwWtK=U-'
    'bx6hz;Rb?S&MFde<V{tb2p8tPrS7o`)*?=|4983d7PD0!XK?~wg`#cm2&I@TB-O9jJG?!5|WcLd5v67)uJ4+t<KHGc6qvnp-'
    '@jF1&AzEn#;7tE^V3p^*s1w&h2jpx3Uh!T{+$&~G&!z!)vg2dSSC08Q;eIoCg|so=+w(#&UP<z+1jH6UgY##gT#*;5EQ+p3Syb{2'
    'D^{OMvU`OiQ(0_W;W|~0w}u6{F@JHQc<2apVLJGKTSn1>6Yu%|8@c06G9?;Mn&jhO?bz&}^TPLNN5;J0c@)PzW{#=>vS>^do=wMl'
    '7KEM%t<i@(n|9tSMQ&1NJkJwx<9+9Q{J)K*vq5tLi&V}At~ie~aU3QQ8`5u$_xboSRx-%{$8|9yg;p-'
    '$hS+(Y_#IsM#P`OF?|i>d3TPfB0!}u=KJRgEbRN$J=#Gtt2G_Y;*P*{U=5J@QY{#0_Srgomfa!EO&gMAtOg$azjsLeX-'
    '*Il8<;flSgU;XW_^r+;vm?ss<bQRfjyu=f4tvs>_dc8B<T2qf5Lyiuj1NRlg9yR^)HfLS4vc}peBLO?OR(hWZ1&HnW)se);k|w47'
    '>o0s>4A3c;b<&uJ~!@XFeihC#y^7*3*>yx`^lU-erwpX8=&2eVe@q0Sr6Wyf%NtuiP_O(zF*+yF|}-'
    'E<)lnb0}5F2IupWHas{SkJel~Omhxkj$25$P<8#7B%}P3DMaW1l?Sz=gtV|C!pfO6t!xqiP1Y1L_O<Qy}@FSVpZUwQg*%^o?cs(j'
    'B^0r`0rh2;bOQht$P?%UPb8jkSNXY+@;so-CGeD4*1N2FhXtbLz%*%tBvjHwZQf02mY4a>AWwZ>!58hLw0eRR=&;{#=NSEN-co{5'
    'pA6QWbiQS)7c`enVaON{BOTCD#;Fyi>HT2&XZS_J^i^Y`ZLY3TN7`;#jv>G)d5Hzhuha5iDl5Pk5;YF&!0;0U=U!i5n&!WvgoAa5'
    'jLLgw_EKxq2f%hbuf-==r6m#YQAdszk31D-'
    '%3pndgrOCk`2H>Abc*fQ~sWM==Y4)l;)B=touOHz6g>|sk>~A7IVz}T{7{s!_i8=%mc&v%e1%~gudT|i?;bti#oX2H?!9~PQgy3v'
    '%fW2x*5o--o&0dLUL&d6a^+u}P+GcsNUE`i=dAs45b$!&tPwnjh52N3iveun{+o}Vx-_z9=yQ0FRU^a*Fypjl8F*U900OJc_F@>Z'
    '1Y+=$1sfP_pOJ}cZg@PGsg`j-HIbn}=31^kUCr$S6?475Bz0Or@*b+9iwjugIuPiMnww#Ex-'
    '%#Rj3mku8+c8O4z_!|+R>EORq0ObU-'
    'B<uw8h}NSOJmhuw>{F^Qh@+aQn@6k4DH+w;lQ|}nitC*3tFO7!ll@5+fy!k;1{Sm(JW!ys;|YkyXRFXEXrxlE3dDjImcO{_3OX|>'
    '{aVFew}j#TO6|sp+0dc_H+&1u7T8ah=|$W*3ce~(&Zx{FoD`Fzc;{M!>jh}Z~jTRC3UFObDDF(*@11hJzo2?>!ZE}w$1i<wLq02W'
    'zH))o&@%~{x<A;LwggyM_&wE`-'
    'RObnsAq@{EBK0L|yuYWmhDBqMAcdxyoZ}pQQg)&8cW40Gw?Fe}(JU*`A}ESGzA~3m2jj1|4?RqtWX)TP(dzHB07<jmGDWjw}7gS|'
    'oFC*&cDIvz^#89|GHjg`N697;v^Xw8y_K?iF(Z%&uZvZLet$zuqh+8R+Qn15ri(`Z3)<=;z7rZ8#Wh(^KhK64B{YzQnOvkoF^>21'
    'X;q_~~09uz+Sdi``BLfW662o}2vxNk;A=Ovfva+4#4`-^22O*RDPMI$1SmYrozbKM4zlUS)&{&7C<$UwfN-'
    '`sBtZhs|kh@wdTi?_=Oxgy_)$8w5eIUD)gV8yw~HT8R}_{Q6*(JcLUV_LQ;$oH6j=Z4c~~F;uLECM4K1YdWCZefGF2!VAMu$VfQb'
    'YEBuwwK46?mOhD*d|s)9HI^My>Mmo$$s6RJpDkL54*JK~;(OE~z}CJu^gWCT&r<=&#2y7Z&h|Xb$;cSnYI{w4a2Qi3B@I^boVv};'
    'v)$NZjoL7G?7UVUrQuwr7x973G-'
    'lT^vA&<ztUcV%+3DxbZpSpOdfUU_hOIf+ZtQV<o~`|@HSM9r>li!=*7u=`O59`|Q>+O<=4X%9ZrE~<-'
    '$S@JM2i?GPsSd{nYoffKI7RQ2XBb2HIRn;8A`_CF`n;<8H4S#J=R|!Ksm3f#=x3-'
    'T<{`)9CvUrnR~Wy)fZzrY;hFL_l4Z<I48@$0iHjvyi2QLzAthL$Fg8ykso}960UtQ1{}-A;L_}X2+G-'
    '8VmRBVTdX7>%kb=_v9v7jxhUual_6l=PGf?jk5~T9!kI05wgIwY$Pyz*mfK#rhiga2dG6ubH+{>q#no<dKF&Z+CQwwgS1kui`?6='
    'VWNZ_Hk6FB@WBeF{d#MW3p7bV~H6UeZcs;*o^JtSOjP15P9>q8YN#rqT`r00m`vm4sB0&NZ2lza3rJ1ZoE@gevNV@DUh&}K!5;?f'
    'oBIL@7vrOb56@3z6BVM(KpeK=;iLL#OmEoyaPC=QZv&AtPW1cNiX{qWw!QMWv`Wv1KUiCM-'
    'qDBZmHOoH`nLS$o`KeLc3f)z0k7o^N$X>Z;=h<|gfN1f`Pa<eNuT^_s2`74vXKTO5R4=x;OP%#ZXIpKLxsU-eax(o*>>>4@h{m2R'
    'm>p^A4o;AK5}8S73sPKShUsjPnogvpoNc#ki~e@3vF$0rfh2Aw`C*>vj$oJ?9Wz@vj6N;4SC9h9V*6(cgELprv&AoDW$3bnvzf8m'
    'v?RoNWs=b1mC4TRXU}-FnO^$YB7vS+AA>FTy7q9Ct+U<;w$Cen-NNL|Lf#2=${%~RXx*q6g)73@;;c(ngl{9cunOaB@uZ8!Zn3e8'
    'SMDL#StvO%kCDUQNW(f?jGjUnmb1mUE~?cl1<#>W)?O5>#rDcQ0~<6V(V6A!4Sf%h&3Vn*%eJRqu-v9la5Rj^-'
    'V#n>ccN_0hhj6l5brr#Waz4Dd#jc8;FWuwpaxT2CU{dR$A_)`jibC8n@U*ni4bwtwedMr_)GSxJ$%oJ{YK!b6w8|ij@qm3A>KTej'
    '`s!F8PaOZE82=Tl^h2Dr&)CsH6vI!N|GSR5nvS2pEz5iYIJ4Qaj(zoCOHle#a0etBmaOsGS@AN+=Ybrvt5>w6IBu(%N90&wr5P>m'
    '110gTXt1}%ekKwdsI8D@Sb*PV@>>eWdyw|Ai1-'
    'U@I1|fw^#0M(*&UGfB~;~o!Ya%u|NU%>Q7Fox?U3IBQl3e5^;kG#UZX$>QV(H$tWk&D#m8|zG+PM?Vkrx^;n6e(zUfst7wVxnzkV'
    'ZX*(AIHFw--zwz%ZBu8==ctfr#9;m}Glq!-'
    '+uB2pEr54W{`~B*24Hhq9LoqgM=KW(jzD4`!@PM#iG}bdCxUbKH=fjQfSMK}gQM(HDWB)sPGQg&Yq7FZk8{Y@kk&<}5=S{bLX|JS'
    'YCF&`Oz?IVE=a=uBrfEvjv^j5pWh9ne*_+D!;rq+>RRNy>%yHxU{Co5LX2&Oiz~0#3CGvqc_I-'
    'z75;2(6C1H5p6x&ys6q<AVut6R7hmpV%(;fE4e$RoLH28bde{b^lrk~&B`Az>l76~RN!JS|fpD%uVH0HB?SJk!gcQ#@`R>XhUsB9'
    'H&8*P7!2wXB{403aDiY0VXGBUgYOPyHb79pF4g-'
    'T<qskOJd9l%N?!6lVYu5Bz=6{z7&<Noyh&0bF~2E*fTJ>SOb=>)9sX0c~+7YtU6E!+v!0D;8(!DjltiAyp(wKtRdd|n8bf4K4e%K'
    'hs5-S%~VfXj1Vzt<rHYa8F^_ltkO^7z${UwJ%1bqKh*KYf35-;dvT{N~4N{Jtcd5QcN#uc!0#-'
    'Os1<e7b+X^ZVU@zw`II|9<E1`~B>L$CDuPdDFP>_haz;)31-Z7k&EqEMEWe-'
    '(URw<$vG&`}Xf!A%M~x_QuYC`29Y?z3po}o;Osct~&nU{T;miYUNA$EUbi0r(sBWO7Q}3FrGj-'
    '5{;v<dywcD=MDb8b7L{od4uEaLOv&aGT86kS%!Z6FGOS3*n$*}j5r5Y<2=8=|NQOW{@=g;&kvH$d3wC_P+8_2#|@%&f~nwr8P=77'
    'aI&okqGj>yNR2>xr7-j=y&46AlnKy9`22CZKuaJMVS{q8mC8qIo>hFiTCt*~Zox0E)NQL7EH=_bD!jnTv?iLr{jlQ{Qp`v<GmYD-'
    '(KhWf5ZGWJo4;&@t3&Bp9EV~|6_I(RmSclo(wJ&*M5URU1XZ9hjdfP&^wX&}HY_^l8ZodUjLg6p#~C~h`WyWWL0A?u4}x&hu!=1R'
    'zz_h1g@_dtY{7Gv<5$4qBD-)JL5ivT1+-'
    '~VjckE7vm|YU$ZXN^Mc>n()g>GXZm?<cgXMI*9m;TT#9&*j^Mt}Q@So@{APZx!vi03eeRoT4hFo^Pnb~$L!AT|}9cXB1g~%DUY{D'
    '+BFUxe$mX1l|yJefq6)0qjH)AI@RSp5}H_*Q-'
    'S&bBcoM@n%*i*6_5)9azRiHRv;wBKcCj#I68$bcPZbG^^N@Ge7z19H<GI4f{jlqQ^(ZRytZ{#+gEkK>lvPhVX3V={xUNyk3R_zS}'
    'gxW*x>!?bxc0i@0JJ44Q_7lJ~3{o1Z@C1B5R7W@<RE#}AZD8KuReQsrD8jZY6+%&4P!zx*I?)65tu`zLK4DNx3JCyHHn>RuQ@8vt'
    'VMa8l1i(}UDK&$QbWGfNyz=+D9fipr07f_$sw|GR4f>))0Ja7jRRKlWpdo=XfE@t;2n_1TKeL1fP){|$N5GANNE8Ad+d#R!RGSKp'
    'EWjDgJPTA46-XwU#bd7u^swNb^<n@33>09PoJ^-Tk-{{yP1MYpdct!Hlli2l@dVE+Ln$-x2-'
    'OurBmB(Nw3)>uF^aSz3S7DKDyv{XLs<$34NRyMXhCuud!!njInpgxt`HOE@VpXIGVa9*{Dzp3=l9NQVVH$*_q;Of0ci{28Oz-'
    'e%p%On3cB*f9=~V5xAlA57Qpi=*1OeC+C~M0<`16*O^MpgOzoCGauzs?U-p>pzb^Anmr7iX7-gI2-B8%JK_#%+3{w-'
    '06ON*)yd|rWjl|!j)_knYpuab@z9*C48QJ}-T(m!Z-~0iO2SzdfauoM}3jLp=c*LwZHuS$M8vOiVbI$kg|NZ{~)7r&A'
)

_table_bytes = zlib.decompress(base64.b85decode(_TABLES_B85))
if len(_table_bytes) != _TABLES_SIZE or hashlib.sha256(_table_bytes).hexdigest() != _TABLES_SHA256:
    raise RuntimeError('Integrità delle tabelle Unicode Minerva non valida.')
_tables = json.loads(_table_bytes)
_CCC = {int(key): value for key, value in _tables['ccc'].items()}
_DECOMPOSE = {int(key): tuple(value) for key, value in _tables['decomp'].items()}
_COMPOSE = {(left, right): value for left, right, value in _tables['compose']}
_LETTER_RANGES = tuple(tuple(value) for value in _tables['letter_ranges'])
_NUMBER_RANGES = tuple(tuple(value) for value in _tables['number_ranges'])
_LETTER_STARTS = tuple(left for left, _ in _LETTER_RANGES)
_NUMBER_STARTS = tuple(left for left, _ in _NUMBER_RANGES)
del _tables, _table_bytes
