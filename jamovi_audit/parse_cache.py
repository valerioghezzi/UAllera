"""Bounded parsing reuse within one audit, without files or external services.

Create a new ParsingCache for each audit. Only successful reader dictionaries
are retained. Cached values are private copies; callers receive independent
copies so student-specific ZIP metadata cannot contaminate another submission.
The byte budget uses a conservative retained-object estimate plus JSON and entry
headroom. It is an allocation estimate, not an operating-system RSS guarantee.
Bump CACHE_SCHEMA_VERSION when the shape or meaning of reader results changes.
"""
from collections import OrderedDict
import copy
import hashlib
import itertools
import json
import threading
import sys

CACHE_SCHEMA_VERSION = 'uallera-parsing-1'
BYTES_ACCOUNTING = 'unique_deepcopy_graph+json+entry_overhead'
_ENTRY_OVERHEAD = (sys.getsizeof(OrderedDict([(b'', None)])) -
                   sys.getsizeof(OrderedDict()) + sys.getsizeof((None, 0, 0)) +
                   2 * sys.getsizeof(0))
_NATIVE_TYPES = {dict, list, tuple, str, int, float, bool, type(None)}


def _key(kind, data):
    digest = hashlib.sha256()
    # Length framing prevents a kind containing separators from colliding with
    # a different kind/input pair, without concatenating large document bytes.
    for component in (CACHE_SCHEMA_VERSION.encode('utf-8'), kind.encode('utf-8')):
        digest.update(len(component).to_bytes(8, 'big'))
        digest.update(component)
    digest.update(data)
    return digest.digest()


def _graph_size(parsed, limit):
    """Bounded, cycle-aware traversal of plain reader data, counting unique IDs."""
    size = 0
    seen = set()
    active = set()
    stack = [(iter((parsed,)), None)]
    while stack:
        iterator, parent = stack[-1]
        try:
            value = next(iterator)
        except StopIteration:
            stack.pop()
            active.discard(parent)
            continue
        identity = id(value)
        if identity in active:
            return None
        if identity in seen:
            continue
        if type(value) not in _NATIVE_TYPES:
            return None
        seen.add(identity)
        size += sys.getsizeof(value)
        if size > limit:
            return None
        if type(value) in (dict, list, tuple):
            # Reject pathological nesting without recursion or large work lists.
            if len(stack) >= 256:
                return None
            children = itertools.chain.from_iterable(value.items()) if type(value) is dict else iter(value)
            active.add(identity)
            stack.append((iter(children), identity))
    return size


def _encoded_size(parsed, limit):
    """Count the real JSON encoding incrementally; retain no serialized payload."""
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    try:
        for chunk in encoder.iterencode(parsed):
            # Every Unicode code point needs at least one UTF-8 byte. Avoid an
            # additional huge bytes allocation for an obviously oversized string.
            if len(chunk) > limit - size:
                return None
            size += len(chunk.encode('utf-8'))
            if size > limit:
                return None
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        return None
    return size


class ParsingCache:
    """Thread-safe LRU; zero budgets disable retention, and errors are retried.

    Readers run outside the cache lock. Concurrent misses may parse the same
    input twice; publication and accounting remain atomic. There is no global
    cache, persistent content, or cross-audit reuse.
    """

    def __init__(self, max_entries=32, max_bytes=32 * 1024 * 1024):
        for value in (max_entries, max_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError('Cache budgets must be nonnegative integers.')
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._encoded_bytes = 0
        self._counts = dict(hits=0, misses=0, bypassed=0, evictions=0)
        self._lock = threading.Lock()

    def _bypass(self):
        with self._lock:
            self._counts['bypassed'] += 1

    def read(self, kind, data, reader):
        """Read bytes with reader(data), reusing successful results by content.

        Error dictionaries, non-dictionaries, non-JSON values and oversized
        results are returned uncached. Exceptions from the reader propagate;
        they are never retained. Input bytes and document text never enter stats.
        """
        if not isinstance(kind, str) or not kind:
            raise ValueError('A nonempty reader kind is required.')
        if not isinstance(data, bytes):
            raise TypeError('ParsingCache requires immutable input bytes.')
        if not callable(reader):
            raise TypeError('A callable reader is required.')
        key = _key(kind, data)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._counts['hits'] += 1
            else:
                self._counts['misses'] += 1
        if cached is not None:
            return copy.deepcopy(cached[0])
        try:
            parsed = reader(data)
        except Exception:
            self._bypass()
            raise
        if (not isinstance(parsed, dict) or parsed.get('errors') or
                self._max_entries == 0 or self._max_bytes == 0):
            self._bypass()
            return parsed
        overhead = _ENTRY_OVERHEAD + sys.getsizeof(key)
        graph_size = _graph_size(parsed, self._max_bytes - overhead)
        if graph_size is None:
            self._bypass()
            return parsed
        encoded_size = _encoded_size(parsed, self._max_bytes - graph_size - overhead)
        if encoded_size is None:
            self._bypass()
            return parsed
        try:
            owned = copy.deepcopy(parsed)
            # A copied list can have different allocation capacity. Charge the
            # actual private graph that will be retained, not the source graph.
            retained_size = _graph_size(owned, self._max_bytes - encoded_size - overhead)
            if retained_size is None:
                self._bypass()
                return parsed
            size = retained_size + encoded_size + overhead
            result = copy.deepcopy(owned)
        except Exception:
            self._bypass()
            return parsed
        with self._lock:
            # A simultaneous miss can have published this key while parsing.
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
                self._encoded_bytes -= previous[2]
            while self._entries and (len(self._entries) >= self._max_entries or
                                     self._bytes + size > self._max_bytes):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed[1]
                self._encoded_bytes -= removed[2]
                self._counts['evictions'] += 1
            self._entries[key] = (owned, size, encoded_size)
            self._bytes += size
            self._encoded_bytes += encoded_size
        return result

    def stats(self):
        """Return a content-free snapshot; modifying it cannot change counters."""
        with self._lock:
            return dict(self._counts, entries=len(self._entries), bytes=self._bytes,
                        encoded_bytes=self._encoded_bytes, bytes_accounting=BYTES_ACCOUNTING,
                        max_entries=self._max_entries, max_bytes=self._max_bytes)
