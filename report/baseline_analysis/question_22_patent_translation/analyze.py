"""Compare lexical overlap and paired ranks for the reported patent language directions."""
from pathlib import Path
from functools import cache
import pandas as pd
import pyarrow.parquet as pq
from clir_benchmark.lexical import lexical_tokens

HERE = Path(__file__).resolve().parent
ART = HERE.parents[2] / "artifacts"
SHARED = ART / "narval-20261119-shared-indexes/artifacts"
TRANS = ART / "narval-20261120-transformed-retrieval/artifacts"
data = pd.read_csv(HERE.parent / "question_21_retrieval_tradeoffs/by_setting.csv")
data = data.query("corpus=='chemistry' and regime=='cross_language' and (method=='original' or (method=='translate' and language==target_languages)) and retriever!='hybrid'")
wide = data.pivot(index="setting_id",columns=["method","model"],values="recall10")
wide.to_csv(HERE / "direction_scores.csv")
print(wide.mean())
qrels = pd.read_parquet(next((SHARED/"relevance/chemistry").glob("*/qrels.parquet")))
qrels = qrels[qrels.setting_id.isin(wide.index) & qrels.relevance.gt(0)]
ids = set(qrels.chunk_id)
chunks = {}
for batch in pq.ParquetFile(next((SHARED/"corpus/chemistry").glob("*/chunks.parquet"))).iter_batches(columns=["id","text"]):
    chunks.update((k,t) for k,t in zip(*batch.to_pydict().values()) if k in ids)
tokens = {k:set(lexical_tokens(t)) for k,t in chunks.items()}

@cache
def rankings(run):
    path = TRANS/"runs/chemistry"/run/"run.parquet"
    frame = pd.read_parquet(path, columns=["setting_id","query_id","chunk_id","rank"],filters=[("rank","<=",10)])
    return frame.merge(qrels[["setting_id","query_id","chunk_id"]].drop_duplicates(), on=["setting_id","query_id","chunk_id"]).groupby(["setting_id","query_id"])["rank"].min().to_dict()

@cache
def translations(lang):
    return pd.read_parquet(next((ART/"verified-20261118-query-generation/translations/chemistry"/lang).glob("*/queries.parquet"))).set_index("id")

runs = data.query("method=='translate'").set_index(["setting_id","model"]).run_id
records = []
for (setting, query), group in qrels.groupby(["setting_id","query_id"]):
    tr = translations(setting.split("__")[-1]).loc[query]
    bm, dense = [rankings(runs.loc[setting,m]).get((setting,query)) for m in ["BM25","bge-m3"]]
    if not isinstance(tr.text,str) or tr.status=="provider_refused":
        continue
    src, dest = [set(lexical_tokens(t)) for t in [tr.original_text,tr.text]]
    def coverage(words):
        return max(len(words & tokens[k])/len(words) for k in group.chunk_id)
    best = max(group.chunk_id, key=lambda k:len(dest & tokens[k]))
    records.append(dict(setting=setting,query_id=query,original_overlap=coverage(src),translated_overlap=coverage(dest),bm25_first_relevant=bm,bge_first_relevant=dense,translation=tr.text,source_question=tr.original_text,example_relevant_chunk=chunks[best],chunk_id=best))
out = pd.DataFrame(records)
out.to_csv(HERE/"paired_examples.csv",index=False)
print(len(out),out[['original_overlap','translated_overlap']].mean())
print(out.groupby(out.bm25_first_relevant.notna()).translated_overlap.agg(['mean','size']))
examples=out[out.bm25_first_relevant.eq(1)&out.bge_first_relevant.isna()&out.setting.str.endswith('__en')]
for r in examples.head(10).itertuples(): print(r.setting,r.translation[:220],r.example_relevant_chunk[:150])
