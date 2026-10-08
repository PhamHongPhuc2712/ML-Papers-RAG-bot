"""Bounded evidence retrieval over a real release: caps, budgets, exclusions and provenance."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import Engine, text

from copilot.corpus.export import export_snapshot
from copilot.corpus.releases import activate_release, capture_release
from copilot.db.session import assert_safe_test_database, cleanup_test_vectors
from copilot.evidence.retrieve import (
    EvidenceConfig,
    EvidenceRetriever,
    build_context,
    expand_to_section,
    load_evidence_config,
)
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.index import (
    CHUNKS,
    PAPERS,
    CollectionSettings,
    IndexConfig,
    build_index,
    index_validator,
)
from copilot.search.rerank import FixtureReranker

pytestmark = pytest.mark.integration

RELEASE = "evidence-a"
TABLES = (
    "chunks",
    "paper_versions",
    "paper_identifiers",
    "paper_authors",
    "authors",
    "papers",
    "venues",
)
# (title, abstract, [(section, kind, text), ...]); the misleading abstract names the
# method the body says the paper did not use (plan P4.1's two-paper case).
PAPERS_SEEDED = (
    (
        "Contrastive sentence encoders",
        "We use a contrastive loss to train sentence encoders.",
        [
            (
                "Method",
                "body",
                "The encoder is trained with a contrastive loss over sentence pairs.",
            ),
            ("Method", "body", "Pairs are mined from paraphrase corpora with hard negatives."),
            ("Method", "body", "Training runs for ten epochs on contrastive sentence pairs."),
            ("Results", "body", "Contrastive sentence encoders gain 3.2 points on STS."),
            (
                "Results",
                "body",
                "Ablations show hard negatives matter for contrastive sentence encoders.",
            ),
            ("References", "references", "Contrastive sentence encoders, Gao et al. 2021."),
        ],
    ),
    (
        "Misleading abstract about contrastive encoders",
        "A contrastive loss trains our sentence encoders well.",
        [
            (
                "Method",
                "body",
                "We do not use a contrastive loss; a masked objective trains the model.",
            ),
            ("Discussion", "body", "The abstract overstates the role of contrastive training."),
        ],
    ),
    (
        "Diffusion models",
        "Denoising generates images.",
        [("Method", "body", "A noise schedule is learned for image synthesis.")],
    ),
    (
        "Unparsed survey of contrastive encoders",
        "A survey of contrastive sentence encoders and their losses.",
        [],
    ),
)


@dataclass
class EvidenceCorpus:
    engine: Engine
    client: QdrantClient
    ids: dict[str, str]
    chunks: dict[str, list[tuple[str, str, str]]]  # paper id -> (chunk id, section, text)
    release: str = RELEASE


def _seed(engine: Engine) -> tuple[dict[str, str], dict[str, list[tuple[str, str, str]]]]:
    ids: dict[str, str] = {}
    chunks: dict[str, list[tuple[str, str, str]]] = {}
    with engine.begin() as connection:
        venue_id = connection.execute(
            text(
                "insert into venues (id, name, track) values (gen_random_uuid(), 'ICLR',"
                " 'main') returning id"
            )
        ).scalar_one()
        for title, abstract, sections in PAPERS_SEEDED:
            paper_id = str(uuid4())
            ids[title] = paper_id
            chunks[paper_id] = []
            connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, acceptance_type, metadata_status) values (:id, :title,"
                    " :abstract, 2024, :venue, now(), 'poster', 'active')"
                ),
                {"id": paper_id, "title": title, "abstract": abstract, "venue": venue_id},
            )
            if not sections:
                continue
            version = connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper, 'papercli', 'u', 'rev',"
                    " :sha, 'unknown', 'pypdf-text-v2', 'parsed') returning id"
                ),
                {"paper": paper_id, "sha": paper_id.replace("-", "")[:32] * 2},
            ).scalar_one()
            section_ordinals: dict[str, int] = {}
            for ordinal, (section, kind, body) in enumerate(sections):
                section_ordinals.setdefault(section, len(section_ordinals))
                chunk_id = connection.execute(
                    text(
                        "insert into chunks (id, paper_version_id, section_path,"
                        " section_ordinal, kind, ordinal, text, token_count, parser_version,"
                        " chunker_version, evidence_default, page_start, page_end) values"
                        " (gen_random_uuid(), :version, :section, :section_ordinal, :kind,"
                        " :ordinal, :text, 12, 'pypdf-text-v2', 'paragraph-pack-v1',"
                        " :evidence, :page, :page) returning id"
                    ),
                    {
                        "version": version,
                        "section": section,
                        "section_ordinal": section_ordinals[section],
                        "kind": kind,
                        "ordinal": ordinal,
                        "text": body,
                        "evidence": kind != "references",
                        "page": ordinal + 1,
                    },
                ).scalar_one()
                chunks[paper_id].append((str(chunk_id), section, body))
    return ids, chunks


@pytest.fixture(scope="module")
def corpus(migrated_database, test_settings, tmp_path_factory) -> Iterator[EvidenceCorpus]:
    assert_safe_test_database(test_settings)
    client = QdrantClient(url=test_settings.qdrant_url, timeout=60)
    with migrated_database.begin() as connection:
        connection.execute(text("delete from active_release"))
        connection.execute(text("delete from corpus_releases"))
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    cleanup_test_vectors(test_settings, client)
    ids, chunks = _seed(migrated_database)
    root = tmp_path_factory.mktemp("evidence")
    snapshot = root / "exports" / RELEASE
    export_snapshot(RELEASE, snapshot, engine=migrated_database)
    model = FixtureEmbedding()
    build_index(
        Path(snapshot / "manifest.json"),
        model,
        engine=migrated_database,
        client=client,
        prefix=test_settings.qdrant_collection_prefix,
        data_dir=root / "data",
        config=IndexConfig(
            collections={
                PAPERS: CollectionSettings(max_tokens=64),
                CHUNKS: CollectionSettings(max_tokens=64),
            },
            batch_points=4,
            canary_queries=2,
            canary_limit=3,
            canary_terms=3,
        ),
        release_id=RELEASE,
    )
    activate_release(
        migrated_database,
        RELEASE,
        validator=index_validator(client, model_identity=model.identity, dimensions=2),
    )
    try:
        yield EvidenceCorpus(migrated_database, client, ids, chunks)
    finally:
        # Leave no release active: the health suite that follows expects none.
        with migrated_database.begin() as connection:
            connection.execute(text("delete from active_release"))
            connection.execute(text("delete from corpus_releases"))
        cleanup_test_vectors(test_settings, client)
        client.close()


def _config(**overrides) -> EvidenceConfig:
    base = load_evidence_config("configs/evidence.yaml")
    values = {"tokenizer_kind": "whitespace", **overrides}
    return base.replace(**values)


def _retriever(corpus, **overrides) -> EvidenceRetriever:
    return EvidenceRetriever(
        engine=corpus.engine,
        client=corpus.client,
        embedder=FixtureEmbedding(),
        reranker=FixtureReranker(),
        config=_config(**overrides),
        data_dir=Path("/nonexistent"),
    )


QUERY = "contrastive loss sentence encoders"


def test_evidence_comes_from_the_named_papers_only_with_caps_and_provenance(corpus):
    retriever = _retriever(corpus, per_paper=3, evidence=12)
    papers = [corpus.ids["Contrastive sentence encoders"], corpus.ids["Diffusion models"]]
    evidence = retriever.retrieve(QUERY, papers, corpus.release)
    assert evidence, "the named papers have matching chunks"
    assert {str(item.source_id) for item in evidence} <= set(papers)
    # At most three per paper, ids are the sequential labels claims will cite.
    per_paper = {}
    for item in evidence:
        per_paper[str(item.source_id)] = per_paper.get(str(item.source_id), 0) + 1
    assert max(per_paper.values()) <= 3
    assert [item.id for item in evidence] == [f"E{n}" for n in range(1, len(evidence) + 1)]
    # Each piece names its chunk, section and page span from the stored row.
    known = {
        chunk_id: (section, body)
        for cid in papers
        for chunk_id, section, body in corpus.chunks[cid]
    }
    for item in evidence:
        section, body = known[str(item.chunk_id)]
        assert item.section == section and item.text == body
        assert item.page_start is not None and item.page_end == item.page_start
        assert item.source_kind == "paper" and item.version_id


def test_references_are_excluded_by_default_and_included_on_request(corpus):
    paper = corpus.ids["Contrastive sentence encoders"]
    default = _retriever(corpus).retrieve("Gao et al. 2021 contrastive", [paper], corpus.release)
    assert all(item.section != "References" for item in default)
    included = _retriever(corpus, include_references=True).retrieve(
        "Gao et al. 2021 contrastive", [paper], corpus.release
    )
    assert any(item.section == "References" for item in included)


def test_one_paper_cannot_fill_the_context(corpus):
    paper = corpus.ids["Contrastive sentence encoders"]
    evidence = _retriever(corpus, per_paper=2, evidence=12).retrieve(QUERY, [paper], corpus.release)
    assert len(evidence) == 2


def test_the_token_budget_counts_model_tokens_and_drops_what_does_not_fit(corpus):
    paper = corpus.ids["Contrastive sentence encoders"]
    # The whitespace tokenizer stands in for the pinned one: 20 tokens fit two short chunks.
    small = _retriever(corpus, per_paper=5, evidence=12, token_budget=20)
    evidence = small.retrieve(QUERY, [paper], corpus.release)
    assert 1 <= len(evidence) <= 2
    assert sum(len(item.text.split()) for item in evidence) <= 20
    assert small.last_stats["budget_dropped"] >= 1
    with pytest.raises(ValueError, match="token_budget"):
        _config(token_budget=0)


def test_no_evidence_is_an_empty_list_not_an_error(corpus):
    retriever = _retriever(corpus)
    assert retriever.retrieve(QUERY, [], corpus.release) == []
    assert retriever.retrieve(QUERY, [str(uuid4())], corpus.release) == []
    # Without a floor the dense branch always answers inside a paper, so an unrelated
    # query still yields that paper's chunk; a floor on the reranker logit removes it.
    diffusion = [corpus.ids["Diffusion models"]]
    assert retriever.retrieve("zzzz qqqq", diffusion, corpus.release)
    floored = _retriever(corpus, min_rerank_score=0.0)
    assert floored.retrieve("zzzz qqqq", diffusion, corpus.release) == []
    assert floored.last_stats["floor_dropped"] == 1


def test_an_unparsed_paper_contributes_its_abstract_labelled_as_such(corpus):
    paper = corpus.ids["Unparsed survey of contrastive encoders"]
    evidence = _retriever(corpus).retrieve(QUERY, [paper], corpus.release)
    assert len(evidence) == 1
    (item,) = evidence
    assert item.abstract_only is True and item.chunk_id is None and item.section == "abstract"
    assert item.text == "A survey of contrastive sentence encoders and their losses."
    assert item.page_start is None and item.page_end is None


def test_a_misleading_abstract_does_not_hide_what_the_body_says(corpus):
    """Evidence is read from the body: the paper whose abstract claims a contrastive loss
    supplies the chunk saying it did not use one, never the abstract."""

    paper = corpus.ids["Misleading abstract about contrastive encoders"]
    evidence = _retriever(corpus).retrieve(QUERY, [paper], corpus.release)
    texts = [item.text for item in evidence]
    assert any("do not use a contrastive loss" in t for t in texts)
    assert all(not item.abstract_only for item in evidence)


def test_a_cross_release_or_unknown_chunk_id_is_rejected_as_a_citation(corpus):
    from copilot.contracts import Claim
    from copilot.evidence.citations import provenance_valid

    paper = corpus.ids["Contrastive sentence encoders"]
    evidence = _retriever(corpus).retrieve(QUERY, [paper], corpus.release)
    allowed = {item.id: item for item in evidence}
    foreign = Claim(text="from another release", evidence_ids=[str(uuid4())])
    assert provenance_valid([foreign], allowed) is False
    assert provenance_valid([Claim(text="ok", evidence_ids=[evidence[0].id])], allowed) is True


def test_paper_level_miss_is_measured_against_a_global_chunk_baseline(corpus):
    """Restricting to the wrong papers misses what a global chunk search finds; the retriever
    exposes the global search so an experiment can compare the two."""

    retriever = _retriever(corpus, min_rerank_score=0.0)
    restricted = retriever.retrieve(QUERY, [corpus.ids["Diffusion models"]], corpus.release)
    assert restricted == []
    everywhere = retriever.retrieve(QUERY, None, corpus.release)
    assert everywhere and corpus.ids["Contrastive sentence encoders"] in {
        str(item.source_id) for item in everywhere
    }


def test_expansion_returns_the_section_when_it_fits_and_grows_outward_when_not(corpus):
    paper = corpus.ids["Contrastive sentence encoders"]
    retriever = _retriever(corpus, per_paper=1, evidence=1)
    (hit,) = retriever.retrieve("hard negatives paraphrase corpora", [paper], corpus.release)
    assert "hard negatives" in hit.text and hit.section == "Method"
    method = [body for _, section, body in corpus.chunks[paper] if section == "Method"]
    # The whole section fits and is at least a fifth of the budget: the section is returned.
    section = retriever.section_chunks(hit)
    whole = expand_to_section(hit, chunks=section, max_chars=400, min_share=0.2)
    assert whole.expanded is True and whole.text == "\n\n".join(method)
    assert whole.id == hit.id and whole.chunk_id == hit.chunk_id
    # Too large for the budget: grow outward from the hit, never clipping the hit itself.
    grown = expand_to_section(hit, chunks=section, max_chars=len(hit.text) + 70, min_share=0.2)
    assert grown.expanded is True and hit.text in grown.text
    assert len(hit.text) < len(grown.text) < len(whole.text)
    # A section smaller than the minimum share is left as the chunk.
    tiny = expand_to_section(hit, chunks=section, max_chars=100_000, min_share=0.2)
    assert tiny.expanded is False and tiny.text == hit.text
    # Under `expansion: section` the retriever applies it and keeps the budget.
    expanded = _retriever(
        corpus, per_paper=1, evidence=1, expansion="section", expansion_max_chars=400
    )
    (item,) = expanded.retrieve("hard negatives paraphrase corpora", [paper], corpus.release)
    assert item.expanded is True and item.text == whole.text


def test_build_context_keeps_rank_order_within_the_budget():
    from copilot.contracts import Evidence

    def ev(label: str, words: int) -> Evidence:
        return Evidence(
            id=label,
            source_kind="paper",
            source_id=uuid4(),
            version_id="v",
            chunk_id=uuid4(),
            text=" ".join(["w"] * words),
            section="s",
        )

    items = [ev("E1", 5), ev("E2", 10), ev("E3", 3)]
    kept, dropped = build_context(items, token_budget=9, count_tokens=lambda t: len(t.split()))
    assert [i.id for i in kept] == ["E1", "E3"] and dropped == 1
    kept, dropped = build_context(items, token_budget=100, count_tokens=lambda t: len(t.split()))
    assert len(kept) == 3 and dropped == 0


def test_the_release_is_the_one_named_not_the_active_pointer(corpus):
    assert capture_release(corpus.engine).id == corpus.release
    retriever = _retriever(corpus)
    with pytest.raises(Exception, match="release_unknown"):
        retriever.retrieve(QUERY, [corpus.ids["Diffusion models"]], "no-such-release")
