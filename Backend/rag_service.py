"""Read-only manual retrieval with scoped recall, evidence, cache and timings."""
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

ALIASES = {
    '洗衣机': ('washing machine', 'washing_machine', 'washer', 'cwe'),
    '干衣机': ('dryer', 'tumble dryer', '烘干机', 'cde'),
    '冰箱': ('refrigerator', 'fridge', 'fge'),
    '热泵': ('heat pump', 'heatpump', '空调', 'hpe'),
    '电视': ('television', 'tv', 'tve'),
    '电烤箱': ('oven', '烤箱', 'woe'),
    '热水壶': ('kettle', '电热水壶', '电水壶'),
}


def tokens(value):
    value = value.casefold()
    result = re.findall(r'[a-z0-9]+', value)
    for run in re.findall(r'[\u4e00-\u9fff]+', value):
        result.extend(run)
        result.extend(run[i:i+2] for i in range(len(run)-1))
    return result or ['_empty_']


def contains(text, term):
    if re.search(r'[\u4e00-\u9fff]', term):
        return term in text
    return re.search(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', text.casefold()) is not None


def expand_query(query):
    additions = []
    for fragment, translated in [
        ('clean', '清洁 清洗'), ('maintenan', '维护 保养'), ('filter', '滤网 过滤器'),
        ('precaution', '注意事项 安全'), ('detergent', '洗涤剂'), ('drain', '排水'),
        ('error code', '故障代码'), ('door', '机门 门封'),
    ]:
        if fragment in query.casefold():
            additions.append(translated)
    return query + ('\n检索关键词：' + ' '.join(additions) if additions else '')


def clean_text(value):
    value = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', value, flags=re.S | re.I)
    value = re.sub(r'<think\b[^>]*>.*$', '', value, flags=re.S | re.I).strip()
    if value.startswith('```') and value.endswith('```'):
        value = re.sub(r'^```[^\n]*\n', '', value).removesuffix('```').strip()
    return value


def key(doc):
    return (doc.metadata.get('source'), doc.metadata.get('page'), clean_text(doc.page_content))


class ManualRetriever:
    def __init__(self, store, reranker=None, cache_size=32):
        self.store = store
        self.reranker = reranker
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def query(self, app_name, problem_desc):
        # Serialize local retrieval to protect the model and cache; no LLM calls here.
        with self.lock:
            return self._query(app_name.strip(), problem_desc.strip())

    def _query(self, app_name, problem_desc):
        begin = time.perf_counter()
        times = {}
        if not problem_desc:
            return json.dumps({'status': 'invalid_query', 'evidence': []})
        start = time.perf_counter()
        data = self.store._collection.get(include=['documents', 'metadatas'])
        docs = [Document(page_content=clean_text(t), metadata=m or {})
                for t, m in zip(data.get('documents') or [], data.get('metadatas') or []) if t and clean_text(t)]
        times['corpus'] = round(time.perf_counter()-start, 4)
        available = sorted({d.metadata.get('appliance_name') for d in docs if d.metadata.get('appliance_name')})
        name_text = app_name or problem_desc
        targets = set()
        for name in available:
            if any(contains(name_text, alias) for alias in (name, *ALIASES.get(name, ()))):
                targets.add(name)
        # Also recognize a concrete manual filename / model identifier.
        for d in docs:
            src = str(d.metadata.get('source', ''))
            model = re.sub(r'^manual_', '', src, flags=re.I).removesuffix('.pdf')
            if model and re.search(r'\d', model) and contains(name_text, model):
                targets.add(d.metadata.get('appliance_name'))
        if app_name and not targets:
            return json.dumps({'status': 'unknown_appliance', 'available_appliances': available,
                               'evidence': [], 'message': '没有匹配的设备；不会退回其他设备的说明书。'}, ensure_ascii=False)
        scoped = [d for d in docs if not targets or d.metadata.get('appliance_name') in targets]
        if not scoped:
            return json.dumps({'status': 'no_documents', 'evidence': []}, ensure_ascii=False)
        fingerprint = hashlib.sha256(json.dumps([(d.metadata, d.page_content) for d in scoped],
                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_key = (fingerprint, tuple(sorted(targets)), ' '.join(problem_desc.casefold().split()))
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            result = json.loads(self.cache[cache_key])
            result['cache_hit'] = True
            result['timings_seconds'] = {**times, 'total': round(time.perf_counter()-begin, 4)}
            logger.info('[RAG耗时] cache_hit total=%.3fs', time.perf_counter()-begin)
            return json.dumps(result, ensure_ascii=False)
        where = None
        if len(targets) == 1:
            where = {'appliance_name': next(iter(targets))}
        elif targets:
            where = {'appliance_name': {'$in': sorted(targets)}}
        query = expand_query(problem_desc)
        start = time.perf_counter()
        vector_hits = self.store.similarity_search_with_score(query, k=min(8, len(scoped)), filter=where)
        # Enforce the same scope on both channels, even if a backend ignores a filter.
        allowed = {key(d) for d in scoped}
        vector_docs = [Document(page_content=clean_text(d.page_content), metadata=d.metadata)
                       for d, score in vector_hits if key(d) in allowed]
        times['vector'] = round(time.perf_counter()-start, 4)
        start = time.perf_counter()
        lexical = BM25Okapi([tokens(d.page_content) for d in scoped])
        scores = lexical.get_scores(tokens(query))
        lexical_docs = [scoped[i] for i in sorted(range(len(scoped)), key=lambda i: scores[i], reverse=True)
                        if set(tokens(query)).intersection(tokens(scoped[i].page_content))][:8]
        # Reciprocal rank fusion retains original documents and their metadata.
        fused, by_key = {}, {}
        for ranking in (vector_docs, lexical_docs):
            for rank, d in enumerate(ranking, 1):
                k = key(d)
                by_key[k] = d
                fused[k] = fused.get(k, 0) + 1/(60+rank)
        candidates = [by_key[k] for k in sorted(fused, key=fused.get, reverse=True)[:10]]
        times['bm25_fusion'] = round(time.perf_counter()-start, 4)
        used_rerank = False
        start = time.perf_counter()
        if self.reranker is not None and len(candidates) > 1:
            try:
                scores = self.reranker.compute_score([[query, d.page_content] for d in candidates], normalize=True)
                candidates = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)]
                used_rerank = True
            except Exception:
                logger.exception('RAG 重排失败，保留融合排序')
        times['rerank'] = round(time.perf_counter()-start, 4)
        evidence = []
        for i, d in enumerate(candidates[:3], 1):
            page = d.metadata.get('page')
            if isinstance(page, bool) or not isinstance(page, (int, str)):
                page = None
            evidence.append({'citation_id': f'S{i}', 'filename': d.metadata.get('source'),
                             'page': page, 'appliance': d.metadata.get('appliance_name'),
                             'text': d.page_content})
        times['total'] = round(time.perf_counter()-begin, 4)
        result = {'status': 'retrieved' if evidence else 'no_results', 'device_filter': sorted(targets),
                  'corpus_chunks': len(scoped), 'cache_hit': False, 'rerank_used': used_rerank,
                  'timings_seconds': times, 'evidence': evidence,
                  'instruction': '只根据确实支持问题的片段作答；引用[S编号]、filename和已存储page。page可能是PDF页序号，未经原PDF核验不得声称为印刷页码。相关性不足请明确说明，不能编造引用。'}
        serialized = json.dumps(result, ensure_ascii=False)
        self.cache[cache_key] = serialized
        self.cache.move_to_end(cache_key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        logger.info('[RAG耗时] corpus=%.3fs vector=%.3fs bm25_fusion=%.3fs rerank=%.3fs total=%.3fs',
                    times['corpus'], times['vector'], times['bm25_fusion'], times['rerank'], times['total'])
        return serialized
