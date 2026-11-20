from dataclasses import replace

import pytest
import yaml
from test_pipeline import write_rows

from clir_benchmark.adapters import DatasetReader
from clir_benchmark.config import SourceSpec, load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import BM25Index, prepare_bm25
from clir_benchmark.pipeline import prepare_corpus
from clir_benchmark.util import read_json
from clir_benchmark.verified_corpora import iter_verified_units, prepare_verified_span_map


def release_reader(directory):
    return DatasetReader(SourceSpec("fixture/verified", "f" * 40, str(directory)))


def test_verified_eurlex_includes_annexes_and_preserves_ids_and_text(tmp_path):
    parent = "32020L0001"
    for table, kind, part in [("corpus", "article", "art_1"), ("annexes", "annex", "anx_pos1")]:
        eli = f"https://data.europa.eu/eli/dir/2020/1/{part}/oj"
        old_eli = eli.replace("https://", "http://")
        write_rows(
            tmp_path,
            table,
            [
                {
                    "_id": eli,
                    "parent_document_id": parent,
                    "unit_type": kind,
                    "title": {"en": f"Heading of the {kind}", "fr": None, "de": None},
                    "text": {
                        "en": f"  Text of the {kind}.\n",
                        "fr": f"{kind} français",
                        "de": None,
                    },
                    "source_unit_ids": {
                        "en": f"eurlex:{old_eli}:en",
                        "fr": f"eurlex:{old_eli}:fr",
                        "de": None,
                    },
                    "available_languages": ["en", "fr"],
                }
            ],
        )
    units = list(iter_verified_units(release_reader(tmp_path), "eurlex"))
    assert len(units) == 4
    assert {(u.unit_type, u.language) for u in units} == {
        ("article", "en"),
        ("article", "fr"),
        ("annex", "en"),
        ("annex", "fr"),
    }
    for unit in units:
        assert unit.corpus == "eurlex"
        assert unit.parent_id == parent
        part = "art_1" if unit.unit_type == "article" else "anx_pos1"
        assert unit.source_id == f"https://data.europa.eu/eli/dir/2020/1/{part}/oj"
        assert unit.id == f"eurlex:http://data.europa.eu/eli/dir/2020/1/{part}/oj:{unit.language}"
        if unit.language == "en":
            assert unit.text == f"  Text of the {unit.unit_type}.\n"


def test_verified_patent_indexes_text_of_every_available_edition(tmp_path):
    publication = "AP-1066-A"
    write_rows(
        tmp_path,
        "corpus",
        [
            {
                "_id": publication,
                "publication_number": publication,
                "publication_date": "2005-05-31",
                "title": {"en": "English title", "fr": "Titre français", "zh": None},
                "abstract": {"en": "Abstract body.", "fr": None, "zh": None},
                "text": {
                    "en": "English title\n\nAbstract body.",
                    "fr": "Titre français",
                    "zh": None,
                },
                "source_unit_ids": {"en": "AP-1066-A_en", "fr": "AP-1066-A_fr", "zh": None},
                "abstract_languages": ["en"],
                "available_languages": ["en", "fr"],
            }
        ],
    )
    units = list(iter_verified_units(release_reader(tmp_path), "chemistry"))
    assert {u.id: u.text for u in units} == {
        "AP-1066-A_en": "English title\n\nAbstract body.",
        "AP-1066-A_fr": "Titre français",
    }
    assert all(u.parent_id == u.source_id == publication for u in units)


def test_verified_un_rejects_available_edition_without_published_id(tmp_path):
    write_rows(
        tmp_path,
        "corpus",
        [
            {
                "_id": "2009/a/res/64/3",
                "parent_document_id": "2009/a/res/64/3",
                "unit_type": "document",
                "source_unit_ids": {"ar": None, "en": "un:2009/a/res/64/3:en"},
                "text": {"ar": "نص", "en": "Body of the resolution."},
                "available_languages": ["ar", "en"],
            }
        ],
    )
    with pytest.raises(ValidationError):
        list(iter_verified_units(release_reader(tmp_path), "un"))


def test_bm25_indexes_verified_corpus_without_reading_qas(tmp_path, chunker):
    release = tmp_path / "release"
    write_rows(
        release,
        "corpus",
        [
            {
                "_id": "x",
                "parent_document_id": "x",
                "unit_type": "document",
                "available_languages": ["fr", "en"],
                "text": {"en": "apple", "fr": "pomme"},
                "source_unit_ids": {"en": "release:x:en", "fr": "release:x:fr"},
            }
        ],
    )
    source = {"repo_id": "fixture/release", "revision": "1" * 40, "local_dir": str(release)}
    spec = {
        "qa": {**source, "local_dir": "absent"},
        "corpus": source,
        "corpus_format": "verified-multilingual-v1",
    }
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "datasets": {"un": spec},
                "baselines": ["bm25"],
                "output_dir": "artifacts",
                "chunking": {"max_tokens": 12, "overlap_tokens": 3},
            }
        )
    )
    config = load_config(path)
    legacy = replace(
        config, datasets={"un": replace(config.datasets["un"], corpus_format="legacy")}
    )
    assert legacy.corpus_dir("un") != config.corpus_dir("un")
    prepared = prepare_corpus(config, "un", chunker=chunker)
    manifest = read_json(prepared / "manifest.json")
    assert manifest["corpus_format"] == "verified-multilingual-v1"
    assert manifest["source_count"] == manifest["chunk_count"] == 2
    index = prepare_bm25(config, "un")
    with BM25Index(index) as db:
        assert db.manifest["corpus_fingerprint"] == manifest["corpus_fingerprint"]
        assert len(next(db.search(["apple"], ("en",)))) == 1
        assert not next(db.search(["apple"], ("fr",)))
    revised_qa = replace(config.datasets["un"].qa, revision="2" * 40)
    updated = replace(config, datasets={"un": replace(config.datasets["un"], qa=revised_qa)})
    assert prepare_bm25(updated, "un") == index


def test_release_offset_map_requires_whole_span_for_a_positive(tmp_path, chunker):
    corpus_dir, qa_dir = tmp_path / "documents", tmp_path / "qa"
    text = "0123456789abcdefghijklmnop"
    base = {"repo_id": "fixture/verified", "revision": "a" * 40}
    write_rows(
        corpus_dir,
        "corpus",
        [
            {
                "_id": "d",
                "parent_document_id": "d",
                "unit_type": "document",
                "available_languages": ["en"],
                "source_unit_ids": {"en": "u"},
                "text": {"en": text},
            }
        ],
    )
    span = {
        "query_id": "q",
        "corpus_id": "d",
        "source_unit_id": "u",
        "language": "en",
        "match_status": "exact",
    }
    write_rows(
        qa_dir,
        "evidence",
        [
            {**span, "span_index": 0, "start": 0, "end": 3, "text": text[:3]},
            {**span, "span_index": 1, "start": 0, "end": 20, "text": text[:20]},
        ],
    )
    path = tmp_path / "configs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "output_dir": "artifacts",
                "chunking": {"max_tokens": 12, "overlap_tokens": 3},
                "datasets": {
                    "un": {
                        "corpus_format": "verified-multilingual-v1",
                        "qa": {**base, "local_dir": str(qa_dir)},
                        "corpus": {**base, "local_dir": str(corpus_dir)},
                    }
                },
            }
        )
    )
    config = load_config(path)
    prepare_corpus(config, "un", chunker=chunker)
    dest = prepare_verified_span_map(config, "un")
    assert prepare_verified_span_map(config, "un") == dest
    manifest = read_json(dest / "manifest.json")
    assert manifest["outcomes"] == {"contained": 1, "split_across_chunks": 1}
    assert manifest["spans"] == 2 and manifest["query_chunk_positives"] == 1
    assert len((dest / "known-chunk-qrels.tsv").read_text().splitlines()) == 2
