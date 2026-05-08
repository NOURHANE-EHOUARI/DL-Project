"""
tests/test_student_b.py — Unit Tests for Student B components
Owner:   Student B (Hiba)
Phase:   All phases

Tests:
  - src/data/preprocessing.py
  - src/data/collators.py
  - src/models/pos_head.py
  - src/models/coref_head.py
  - src/evaluation/pos_eval.py
  - src/evaluation/coref_eval.py
  - src/utils/arabic_utils.py
  - api/schemas.py

Run:
    pytest tests/test_student_b.py -v
    pytest tests/test_student_b.py -v --tb=short
"""

import pytest
import torch
import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# preprocessing.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_remove_diacritics(self):
        from src.data.preprocessing import remove_diacritics
        assert remove_diacritics("مَرْحَبَا") == "مرحبا"
        assert remove_diacritics("الرَّئِيسُ") == "الرئيس"
        assert remove_diacritics("hello") == "hello"

    def test_remove_tatweel(self):
        from src.data.preprocessing import remove_tatweel
        assert remove_tatweel("ـــالكتابـــ") == "الكتاب"
        assert remove_tatweel("مرحبا") == "مرحبا"

    def test_normalize_hamza(self):
        from src.data.preprocessing import normalize_hamza
        assert normalize_hamza("أحمد") == "احمد"
        assert normalize_hamza("إبراهيم") == "ابراهيم"
        assert normalize_hamza("آسيا") == "اسيا"

    def test_normalize_arabic_pipeline(self):
        from src.data.preprocessing import normalize_arabic
        text = "الرَّئِيسُ الأَمْرِيكِيُّ"
        result = normalize_arabic(text)
        assert "َ" not in result   # no diacritics
        assert "أ" not in result   # hamza normalized
        assert len(result) > 0

    def test_normalize_token_empty(self):
        from src.data.preprocessing import normalize_token
        assert normalize_token("") == ""
        assert normalize_token("   ") == "   "

    def test_normalize_token_preserves_non_arabic(self):
        from src.data.preprocessing import normalize_token
        result = normalize_token("2024")
        assert result == "2024"

    def test_conll_ner_to_jsonl(self, tmp_path):
        from src.data.preprocessing import conll_ner_to_jsonl
        # Write sample CoNLL file
        conll = tmp_path / "test.conll"
        conll.write_text(
            "محمد\tB-PER\nعلي\tI-PER\nيعمل\tO\nفي\tO\nالقاهرة\tB-LOC\n\n"
            "شركة\tB-ORG\nأرامكو\tI-ORG\nأعلنت\tO\n\n",
            encoding="utf-8"
        )
        out = tmp_path / "test.jsonl"
        n = conll_ner_to_jsonl(conll, out, normalize=False)
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert "tokens" in rec and "ner_tags" in rec
        assert rec["tokens"] == ["محمد", "علي", "يعمل", "في", "القاهرة"]
        assert rec["ner_tags"] == ["B-PER", "I-PER", "O", "O", "B-LOC"]

    def test_conll_pos_to_jsonl(self, tmp_path):
        from src.data.preprocessing import conll_pos_to_jsonl
        conll = tmp_path / "pos.conll"
        conll.write_text(
            "الطالب\tNOUN\tالطالب\nيدرس\tV\tيدرس\nفي\tPREP\tفي\n\n",
            encoding="utf-8"
        )
        out = tmp_path / "pos.jsonl"
        n = conll_pos_to_jsonl(conll, out, token_col=0, pos_col=1, seg_col=2)
        assert n == 1
        rec = json.loads(out.read_text(encoding="utf-8").strip())
        assert rec["tokens"]   == ["الطالب", "يدرس", "في"]
        assert rec["pos_tags"] == ["NOUN", "V", "PREP"]
        assert rec["segments"] == ["الطالب", "يدرس", "في"]


# ══════════════════════════════════════════════════════════════════════════════
# collators.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCollators:

    def test_mtl_batch_sampler_proportional(self):
        from src.data.collators import MTLBatchSampler
        sizes = {"ner": 1000, "pos": 500, "coref": 100}
        sampler = MTLBatchSampler(sizes, batch_size=32, strategy="proportional", seed=42)
        tasks = list(sampler)
        assert len(tasks) == sampler.steps_per_epoch
        # NER should dominate proportionally
        from collections import Counter
        counts = Counter(tasks)
        assert counts["ner"] > counts["pos"] > counts["coref"]

    def test_mtl_batch_sampler_round_robin(self):
        from src.data.collators import MTLBatchSampler
        sizes = {"ner": 320, "pos": 320, "coref": 320}
        sampler = MTLBatchSampler(sizes, batch_size=32, strategy="round_robin", seed=42)
        tasks = list(sampler)
        # Should cycle: ner, pos, coref, ner, pos, coref...
        assert tasks[0] == "ner"
        assert tasks[1] == "pos"
        assert tasks[2] == "coref"

    def test_mtl_batch_sampler_uniform(self):
        from src.data.collators import MTLBatchSampler
        sizes = {"ner": 1000, "pos": 100, "coref": 50}
        sampler = MTLBatchSampler(sizes, batch_size=32, strategy="uniform", seed=42)
        tasks = list(sampler)
        from collections import Counter
        counts = Counter(tasks)
        # All tasks should be roughly equal
        ratios = {k: v / len(tasks) for k, v in counts.items()}
        for r in ratios.values():
            assert 0.05 < r < 0.65

    def test_mtl_collator_detects_ner(self):
        from src.data.collators import MTLCollator
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        collator = MTLCollator(tokenizer)
        batch = [{"input_ids": torch.tensor([1,2,3]),
                  "attention_mask": torch.tensor([1,1,1]),
                  "ner_labels": torch.tensor([0,1,0])}]
        result = collator(batch)
        assert result["task"] == "ner"
        assert "ner_labels" in result

    def test_mtl_collator_detects_pos(self):
        from src.data.collators import MTLCollator
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        collator = MTLCollator(tokenizer)
        batch = [{"input_ids": torch.tensor([1,2,3]),
                  "attention_mask": torch.tensor([1,1,1]),
                  "pos_labels": torch.tensor([0,1,0])}]
        result = collator(batch)
        assert result["task"] == "pos"

    def test_mtl_collator_unknown_task_raises(self):
        from src.data.collators import MTLCollator
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        collator = MTLCollator(tokenizer)
        batch = [{"input_ids": torch.tensor([1,2,3]),
                  "attention_mask": torch.tensor([1,1,1])}]
        with pytest.raises(ValueError):
            collator(batch)


# ══════════════════════════════════════════════════════════════════════════════
# pos_head.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPOSHead:

    def setup_method(self):
        from src.models.pos_head import POSHead, NUM_LABELS
        self.B, self.S, self.H = 2, 15, 128
        self.head = POSHead(hidden_size=self.H, intermediate_size=64,
                            num_labels=NUM_LABELS, dropout=0.0)
        self.head.eval()

    def test_forward_returns_logits(self):
        from src.models.pos_head import NUM_LABELS
        seq = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        out = self.head(seq, mask)
        assert "logits" in out
        assert out["logits"].shape == (self.B, self.S, NUM_LABELS)

    def test_forward_with_labels_returns_loss(self):
        from src.models.pos_head import NUM_LABELS, IGNORE_INDEX
        seq    = torch.randn(self.B, self.S, self.H)
        mask   = torch.ones(self.B, self.S, dtype=torch.long)
        labels = torch.randint(0, NUM_LABELS, (self.B, self.S))
        labels[1, 10:] = IGNORE_INDEX
        out = self.head(seq, mask, labels=labels)
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_no_labels_no_loss(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        out  = self.head(seq, mask)
        assert "loss" not in out

    def test_predict_returns_strings(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        tags = self.head.predict(seq, mask)
        assert len(tags) == self.B
        assert all(isinstance(t, str) for seq in tags for t in seq)

    def test_morphology_injection(self):
        from src.models.pos_head import POSHead, NUM_LABELS
        head = POSHead(hidden_size=self.H, use_morphology=True,
                       morph_dim=32, num_labels=NUM_LABELS, dropout=0.0)
        head.eval()
        seq   = torch.randn(self.B, self.S, self.H)
        mask  = torch.ones(self.B, self.S, dtype=torch.long)
        morph = torch.randn(self.B, self.S, 32)
        out   = head(seq, mask, morph_features=morph)
        assert "logits" in out

    def test_compute_accuracy(self):
        from src.models.pos_head import NUM_LABELS, IGNORE_INDEX
        seq    = torch.randn(self.B, self.S, self.H)
        mask   = torch.ones(self.B, self.S, dtype=torch.long)
        labels = torch.zeros(self.B, self.S, dtype=torch.long)
        out    = self.head(seq, mask, labels=labels)
        acc    = self.head.compute_accuracy(out["logits"], labels, mask)
        assert "accuracy" in acc
        assert 0.0 <= acc["accuracy"] <= 1.0

    def test_get_label2id_with_extra(self):
        from src.models.pos_head import get_label2id, NUM_LABELS
        mapping = get_label2id(extra_tags=["DIALECT_X", "UNKNOWN"])
        assert "DIALECT_X" in mapping
        assert len(mapping) == NUM_LABELS + 2


# ══════════════════════════════════════════════════════════════════════════════
# coref_head.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCorefHead:

    def setup_method(self):
        from src.models.coref_head import CoreferenceHead
        self.B, self.S, self.H = 2, 20, 128
        self.head = CoreferenceHead(
            hidden_size=self.H, max_span_width=4,
            top_lambda=0.4, antecedent_k=5,
            mention_hidden=64, antecedent_hidden=64, dropout=0.0,
        )
        self.head.eval()

    def test_inference_no_clusters(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        with torch.no_grad():
            out = self.head(seq, mask)
        assert "mention_scores" in out
        assert "predictions" in out
        assert "span_starts" in out
        assert "span_ends" in out
        assert len(out["predictions"]) == self.B

    def test_training_with_clusters_returns_loss(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        clusters = [
            [[[0, 0], [5, 5]]],
            [[[0, 1], [8, 9]]],
        ]
        with torch.no_grad():
            out = self.head(seq, mask, clusters=clusters)
        assert "loss" in out
        assert out["loss"].item() >= 0

    def test_span_starts_ends_valid(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        with torch.no_grad():
            out = self.head(seq, mask)
        starts = out["span_starts"]
        ends   = out["span_ends"]
        assert (ends >= starts).all(), "All span ends must be >= starts"
        assert (starts >= 0).all()
        assert (ends < self.S).all()

    def test_predictions_format(self):
        seq  = torch.randn(self.B, self.S, self.H)
        mask = torch.ones(self.B, self.S, dtype=torch.long)
        with torch.no_grad():
            out = self.head(seq, mask)
        for doc_clusters in out["predictions"]:
            assert isinstance(doc_clusters, list)
            for cluster in doc_clusters:
                assert isinstance(cluster, list)
                for mention in cluster:
                    assert len(mention) == 2


# ══════════════════════════════════════════════════════════════════════════════
# pos_eval.py
# ══════════════════════════════════════════════════════════════════════════════

class TestPOSEval:

    def setup_method(self):
        self.gold = [
            ["NOUN", "V",    "PREP", "NOUN",  "PUNC"],
            ["PART", "V",    "NOUN", "PREP",  "PRON"],
            ["ADJ",  "NOUN", "V",    "NOUN+PRON"],
        ]
        self.pred = [
            ["NOUN", "V",    "PREP", "ADJ",   "PUNC"],  # 1 error
            ["PART", "NOUN", "NOUN", "PREP",  "PRON"],  # 1 error
            ["ADJ",  "NOUN", "V",    "NOUN"],            # 1 error
        ]
        self.tokens = [
            ["الكتاب", "يقرأ",  "في",  "المكتبة", "."],
            ["لما",    "تحب",   "حد",  "من",      "قلبك"],
            ["جديد",   "الطالب","يذهب","بيته"],
        ]

    def test_accuracy_partial(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=0)
        assert 0.0 < res.accuracy < 1.0

    def test_per_tag_populated(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=0)
        assert len(res.per_tag) > 0
        for tag, m in res.per_tag.items():
            assert "accuracy" in m
            assert "f1" in m
            assert "support" in m

    def test_confusions_detected(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=0)
        assert len(res.top_confusions) > 0

    def test_dialect_breakdown(self):
        from src.evaluation.pos_eval import POSEvaluator
        dialects = ["egy", "glf", "lev"]
        ev = POSEvaluator(self.gold, self.pred, self.tokens, dialects=dialects)
        res = ev.evaluate(bootstrap_n=0)
        assert len(res.dialect_accuracy) == 3
        for acc in res.dialect_accuracy.values():
            assert 0.0 <= acc <= 1.0

    def test_morph_breakdown(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=0)
        assert len(res.morph_accuracy) > 0

    def test_bootstrap_ci(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=100, seed=42)
        assert res.acc_ci_low <= res.accuracy <= res.acc_ci_high

    def test_errors_collected(self):
        from src.evaluation.pos_eval import POSEvaluator
        ev = POSEvaluator(self.gold, self.pred, self.tokens)
        res = ev.evaluate(bootstrap_n=0)
        assert len(res.errors) > 0
        for e in res.errors:
            assert "token" in e
            assert "gold" in e
            assert "pred" in e


# ══════════════════════════════════════════════════════════════════════════════
# coref_eval.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCorefEval:

    def setup_method(self):
        self.gold = [
            [[[0,0],[5,5],[10,11]], [[2,3],[7,8]]],
            [[[0,1],[4,4]], [[2,2],[6,7],[9,9]]],
        ]
        self.pred = [
            [[[0,0],[5,5]], [[2,3],[7,8],[10,11]]],
            [[[0,1],[4,4],[2,2]], [[6,7],[9,9]]],
        ]

    def test_muc_score(self):
        from src.evaluation.coref_eval import to_clusters, muc_score
        gold = [c for doc in self.gold for c in [to_clusters(doc)]][0]
        pred = [c for doc in self.pred for c in [to_clusters(doc)]][0]
        score = muc_score(gold, pred)
        assert 0.0 <= score.f1 <= 1.0
        assert 0.0 <= score.precision <= 1.0
        assert 0.0 <= score.recall <= 1.0

    def test_b3_score(self):
        from src.evaluation.coref_eval import to_clusters, b3_score
        gold = to_clusters(self.gold[0])
        pred = to_clusters(self.pred[0])
        score = b3_score(gold, pred)
        assert 0.0 <= score.f1 <= 1.0

    def test_ceafe_score(self):
        from src.evaluation.coref_eval import to_clusters, ceafe_score
        gold = to_clusters(self.gold[0])
        pred = to_clusters(self.pred[0])
        score = ceafe_score(gold, pred)
        assert 0.0 <= score.f1 <= 1.0

    def test_conll_avg(self):
        from src.evaluation.coref_eval import CorefEvaluator
        ev  = CorefEvaluator(self.gold, self.pred)
        res = ev.evaluate(bootstrap_n=0)
        assert 0.0 <= res.conll_f1 <= 1.0
        expected = (res.muc.f1 + res.b3.f1 + res.ceafe.f1) / 3
        assert abs(res.conll_f1 - expected) < 1e-6

    def test_mention_score(self):
        from src.evaluation.coref_eval import CorefEvaluator
        ev  = CorefEvaluator(self.gold, self.pred)
        res = ev.evaluate(bootstrap_n=0)
        assert 0.0 <= res.mention_f1 <= 1.0

    def test_cluster_counts(self):
        from src.evaluation.coref_eval import CorefEvaluator
        ev  = CorefEvaluator(self.gold, self.pred)
        res = ev.evaluate(bootstrap_n=0)
        assert res.n_gold_clusters == 4
        assert res.n_gold_mentions == 10

    def test_bootstrap_ci(self):
        from src.evaluation.coref_eval import CorefEvaluator
        ev  = CorefEvaluator(self.gold, self.pred)
        res = ev.evaluate(bootstrap_n=100, seed=42)
        assert res.conll_ci_low <= res.conll_f1 <= res.conll_ci_high

    def test_to_clusters(self):
        from src.evaluation.coref_eval import to_clusters
        raw = [[[0,1],[4,5]], [[2,2],[7,8]]]
        clusters = to_clusters(raw)
        assert len(clusters) == 2
        assert frozenset([(0,1),(4,5)]) in clusters

    def test_perfect_prediction(self):
        from src.evaluation.coref_eval import CorefEvaluator
        ev  = CorefEvaluator(self.gold, self.gold)  # pred == gold
        res = ev.evaluate(bootstrap_n=0)
        assert res.muc.f1   > 0.95
        assert res.b3.f1    > 0.95
        assert res.conll_f1 > 0.95


# ══════════════════════════════════════════════════════════════════════════════
# arabic_utils.py
# ══════════════════════════════════════════════════════════════════════════════

class TestArabicUtils:

    def test_is_arabic(self):
        from src.utils.arabic_utils import is_arabic
        assert is_arabic("مرحبا") is True
        assert is_arabic("Hello") is False
        assert is_arabic("مرحبا Hello") is True

    def test_is_arabic_word(self):
        from src.utils.arabic_utils import is_arabic_word
        assert is_arabic_word("مرحبا") is True
        assert is_arabic_word("Hello") is False
        assert is_arabic_word("") is False

    def test_arabic_ratio(self):
        from src.utils.arabic_utils import arabic_ratio
        assert arabic_ratio("مرحبا") == 1.0
        assert arabic_ratio("Hello") == 0.0
        ratio = arabic_ratio("مرحبا Hello")
        assert 0.3 < ratio < 0.7

    def test_arabic_ratio_empty(self):
        from src.utils.arabic_utils import arabic_ratio
        assert arabic_ratio("") == 0.0

    def test_has_diacritics(self):
        from src.utils.arabic_utils import has_diacritics
        assert has_diacritics("مَرْحَبَا") is True
        assert has_diacritics("مرحبا")    is False

    def test_is_stopword(self):
        from src.utils.arabic_utils import is_stopword
        assert is_stopword("في")    is True
        assert is_stopword("مرحبا") is False

    def test_numeral_conversion(self):
        from src.utils.arabic_utils import arabic_indic_to_western, western_to_arabic_indic
        assert arabic_indic_to_western("٢٠٢٤") == "2024"
        assert western_to_arabic_indic("2024") == "٢٠٢٤"

    def test_text_direction(self):
        from src.utils.arabic_utils import get_text_direction
        assert get_text_direction("مرحبا")  == "rtl"
        assert get_text_direction("Hello")  == "ltr"

    def test_iob2_to_spans(self):
        from src.utils.arabic_utils import iob2_to_spans
        tokens = ["محمد", "صلاح", "يلعب", "في", "ليفربول"]
        tags   = ["B-PER", "I-PER", "O", "O", "B-ORG"]
        spans  = iob2_to_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0] == {"type":"PER","start":0,"end":1,"text":"محمد صلاح"}
        assert spans[1] == {"type":"ORG","start":4,"end":4,"text":"ليفربول"}

    def test_spans_to_iob2_roundtrip(self):
        from src.utils.arabic_utils import iob2_to_spans, spans_to_iob2
        tokens = ["محمد", "صلاح", "يلعب", "في", "ليفربول"]
        tags   = ["B-PER", "I-PER", "O", "O", "B-ORG"]
        spans  = iob2_to_spans(tokens, tags)
        recovered = spans_to_iob2(len(tokens), spans)
        assert recovered == tags

    def test_validate_iob2_fixes_broken(self):
        from src.utils.arabic_utils import validate_iob2
        broken = ["O", "I-PER", "I-PER", "O"]
        valid, fixed = validate_iob2(broken)
        assert not valid
        assert fixed == ["O", "B-PER", "I-PER", "O"]

    def test_validate_iob2_valid(self):
        from src.utils.arabic_utils import validate_iob2
        tags = ["B-PER", "I-PER", "O", "B-LOC"]
        valid, fixed = validate_iob2(tags)
        assert valid
        assert fixed == tags

    def test_sentence_stats(self):
        from src.utils.arabic_utils import sentence_stats
        sents = [["محمد","يعمل","في","الرياض"],
                 ["شركة","أرامكو","أعلنت"]]
        stats = sentence_stats(sents)
        assert stats["total_sentences"] == 2
        assert stats["total_tokens"]    == 7
        assert stats["vocab_size"]      == 7

    def test_ner_corpus_stats(self):
        from src.utils.arabic_utils import ner_corpus_stats
        sents = [["محمد","يعمل","في","الرياض"]]
        labs  = [["B-PER","O","O","B-LOC"]]
        stats = ner_corpus_stats(sents, labs)
        assert stats["total_entities"] == 2
        assert stats["entity_type_counts"]["PER"] == 1
        assert stats["entity_type_counts"]["LOC"] == 1

    def test_annotate_text_html(self):
        from src.utils.arabic_utils import annotate_text_html
        html = annotate_text_html(
            "محمد يلعب في ليفربول",
            [{"type":"PER","start":0,"end":0,"text":"محمد"},
             {"type":"ORG","start":3,"end":3,"text":"ليفربول"}],
        )
        assert '<mark data-type="PER"' in html
        assert '<mark data-type="ORG"' in html

    def test_entity_colors(self):
        from src.utils.arabic_utils import get_entity_color
        assert get_entity_color("PER")     == "#4A90D9"
        assert get_entity_color("ORG")     == "#27AE60"
        assert get_entity_color("UNKNOWN") == "#95A5A6"


# ══════════════════════════════════════════════════════════════════════════════
# api/schemas.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemas:

    def test_ner_request_strips_whitespace(self):
        from api.schemas import NERRequest
        req = NERRequest(text="  مرحبا  ")
        assert req.text == "مرحبا"

    def test_ner_request_blank_rejected(self):
        from api.schemas import NERRequest
        with pytest.raises(Exception):
            NERRequest(text="   ")

    def test_pos_request_morphology_flag(self):
        from api.schemas import POSRequest
        req = POSRequest(text="الطالب يدرس", include_morphology=True)
        assert req.include_morphology is True

    def test_analyze_request_deduplicates_tasks(self):
        from api.schemas import AnalyzeRequest
        req = AnalyzeRequest(text="نص", tasks=["ner","ner","pos"])
        assert len(req.tasks) == 2

    def test_analyze_request_invalid_task(self):
        from api.schemas import AnalyzeRequest
        with pytest.raises(Exception):
            AnalyzeRequest(text="نص", tasks=["ner","bad_task"])

    def test_coref_request_threshold(self):
        from api.schemas import CorefRequest
        req = CorefRequest(text="نص عربي", threshold=0.7)
        assert req.threshold == 0.7

    def test_coref_request_threshold_out_of_range(self):
        from api.schemas import CorefRequest
        with pytest.raises(Exception):
            CorefRequest(text="نص", threshold=1.5)

    def test_ner_response_construction(self):
        from api.schemas import NERResponse, NERSpan
        resp = NERResponse(
            text="محمد في القاهرة",
            spans=[NERSpan(text="محمد", type="PER", start=0, end=0, score=0.9)],
            n_entities=1, entity_types={"PER":1}, latency_ms=30.0,
        )
        assert resp.n_entities == 1
        assert resp.latency_ms == 30.0

    def test_batch_ner_filters_blank(self):
        from api.schemas import BatchNERRequest
        req = BatchNERRequest(texts=["نص أول", "  ", "نص ثاني", ""])
        assert len(req.texts) == 2

    def test_health_response(self):
        from api.schemas import HealthResponse
        h = HealthResponse(status="ok", model_loaded=True, version="1.0.0")
        assert h.status == "ok"
        assert h.model_loaded is True
