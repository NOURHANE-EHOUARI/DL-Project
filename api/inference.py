"""
api/inference.py — PyTorch Inference Engine for Arabic MTL NLP
Uses the trained model directly without ONNX export.
"""
from __future__ import annotations
import time, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ── Label maps (must match training) ─────────────────────────────────────────
NER_ID2LABEL: dict[int, str] = {
    0:"O",1:"B-PER",2:"I-PER",3:"B-ORG",4:"I-ORG",
    5:"B-LOC",6:"I-LOC",7:"B-GPE",8:"I-GPE",9:"B-DATE",
    10:"I-DATE",11:"B-TIME",12:"I-TIME",13:"B-MONEY",14:"I-MONEY",
    15:"B-PERCENT",16:"I-PERCENT",17:"B-PRODUCT",18:"I-PRODUCT",
    19:"B-EVENT",20:"I-EVENT",21:"B-NORP",22:"I-NORP",
    23:"B-FAC",24:"I-FAC",25:"B-LANGUAGE",26:"I-LANGUAGE",
}
POS_ID2LABEL: dict[int, str] = {}  # filled at load time

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class NEREntity:
    text: str; label: str; start_tok: int; end_tok: int; score: float = 1.0
    def to_dict(self): return {"text":self.text,"label":self.label,
        "start_tok":self.start_tok,"end_tok":self.end_tok,"score":round(self.score,4)}

@dataclass
class POSToken:
    token: str; tag: str; score: float = 1.0
    def to_dict(self): return {"token":self.token,"tag":self.tag,"score":round(self.score,4)}

@dataclass
class CorefCluster:
    cluster_id: int; mentions: list[dict[str,Any]]
    def to_dict(self): return {"cluster_id":self.cluster_id,"mentions":self.mentions}

@dataclass
class InferenceResult:
    tokens: list[str]; ner: list[NEREntity]
    pos: list[POSToken]; coref: list[CorefCluster]; latency_ms: float = 0.0
    def to_dict(self):
        return {"tokens":self.tokens,"ner":[e.to_dict() for e in self.ner],
                "pos":[t.to_dict() for t in self.pos],
                "coref":[c.to_dict() for c in self.coref],
                "latency_ms":round(self.latency_ms,2)}

# ── Inference Engine ──────────────────────────────────────────────────────────
class InferenceEngine:
    """PyTorch inference engine — no ONNX needed."""

    def __init__(self, model, tokenizer, device="cpu", model_path=""):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.model_path = model_path
        self.model.to(device)
        self.model.eval()
        logger.info("InferenceEngine ready on %s", device)

    @classmethod
    def load(cls, checkpoint_path: str, backbone: str = "aubmindlab/bert-base-arabertv02") -> "InferenceEngine":
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.models.mtl_model import ArabicMTLModel
        from src.models.ner_head import NUM_LABELS as NER_NUM_LABELS, ID2LABEL as NER_ID2LABEL_SRC
        from src.models.pos_head import NUM_LABELS as POS_NUM_LABELS, ID2LABEL as POS_ID2LABEL_SRC

        # Update global label maps
        global NER_ID2LABEL, POS_ID2LABEL
        NER_ID2LABEL.update(NER_ID2LABEL_SRC)
        POS_ID2LABEL.update(POS_ID2LABEL_SRC)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model from %s on %s", checkpoint_path, device)

        tokenizer = AutoTokenizer.from_pretrained(backbone)
        ckpt      = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = ArabicMTLModel(
            backbone_name=backbone,
            ner_num_labels=NER_NUM_LABELS,
            pos_num_labels=POS_NUM_LABELS,
            use_bilstm_bridge=False, lstm_hidden=128, dropout=0.0,
            loss_weighting="uncertainty", use_morphology=False,
            max_span_width=5, top_lambda=0.3, antecedent_k=20,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        logger.info("Model loaded: NER=%.4f POS=%.4f Coref=%.4f",
                    ckpt.get("ner_f1",0), ckpt.get("pos_accuracy",0),
                    ckpt.get("coref_f1",0))
        return cls(model=model, tokenizer=tokenizer, device=device,
                   model_path=checkpoint_path)

    def predict_all(self, texts: list[str]) -> list[InferenceResult]:
        t0  = time.perf_counter()
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=512, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            # Single backbone pass, then route to each head
            backbone_out = self.model.backbone(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            hidden = backbone_out.last_hidden_state

            ner_out   = self.model.ner_head(hidden, enc["attention_mask"])
            pos_out   = self.model.pos_head(hidden, enc["attention_mask"])
            coref_out = self.model.coref_head(hidden, enc["attention_mask"])

        ner_logits = ner_out.get("logits", ner_out.get("emissions", 
                     torch.zeros(enc["input_ids"].shape[0], 
                                 enc["input_ids"].shape[1], 27))).cpu()
        pos_logits = pos_out.get("logits", 
                     torch.zeros(enc["input_ids"].shape[0],
                                 enc["input_ids"].shape[1], 31)).cpu()
        coref_pred = coref_out.get("predictions", [])
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)

        results = []
        for i, text in enumerate(texts):
            words    = text.split()
            word_ids = enc["input_ids"][i]

            # NER decode
            gold_ids  = enc["input_ids"][i].cpu().tolist()
            attn      = enc["attention_mask"][i].cpu().tolist()
            pred_ids  = ner_logits[i].argmax(-1).tolist()
            pos_ids   = pos_logits[i].argmax(-1).tolist()

            # Align subwords → words using attention mask
            encoding  = self.tokenizer(words, is_split_into_words=True,
                                        return_tensors="pt", truncation=True,
                                        max_length=512)
            w_ids     = encoding.word_ids()

            ner_tags  = self._align_labels(words, w_ids, pred_ids, NER_ID2LABEL)
            pos_tags  = self._align_labels(words, w_ids, pos_ids,  POS_ID2LABEL)

            entities  = self._iob2_to_spans(words, ner_tags)
            pos_toks  = [POSToken(token=w, tag=t) for w, t in zip(words, pos_tags)]

            # Coref clusters
            clusters  = []
            if i < len(coref_pred):
                for cid, cluster in enumerate(coref_pred[i]):
                    mentions = [{"text": " ".join(words[m[0]:m[1]+1]),
                                 "start": m[0], "end": m[1]}
                                for m in cluster if m[1] < len(words)]
                    if len(mentions) >= 2:
                        clusters.append(CorefCluster(cluster_id=cid, mentions=mentions))

            results.append(InferenceResult(
                tokens=words, ner=entities, pos=pos_toks,
                coref=clusters, latency_ms=latency_ms))
        return results

    def predict_ner(self, texts: list[str]) -> list[list[NEREntity]]:
        return [r.ner for r in self.predict_all(texts)]

    def predict_pos(self, texts: list[str]) -> list[list[POSToken]]:
        return [r.pos for r in self.predict_all(texts)]

    def predict_coref(self, texts: list[str]) -> list[list[CorefCluster]]:
        return [r.coref for r in self.predict_all(texts)]

    @staticmethod
    def _align_labels(words, word_ids, pred_ids, id2label):
        tags = ["O"] * len(words)
        seen = set()
        for j, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(words): continue
            seen.add(wid)
            if j < len(pred_ids):
                tags[wid] = id2label.get(pred_ids[j], "O")
        return tags

    @staticmethod
    def _iob2_to_spans(words, tags):
        entities, i = [], 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                etype, start = tags[i][2:], i
                i += 1
                while i < len(tags) and tags[i] == f"I-{etype}": i += 1
                entities.append(NEREntity(
                    text=" ".join(words[start:i]),
                    label=etype, start_tok=start, end_tok=i-1))
            else:
                i += 1
        return entities

def get_engine_info(engine: InferenceEngine) -> dict:
    return {"backend": "pytorch", "model_path": engine.model_path,
            "device": engine.device, "ner_labels": list(NER_ID2LABEL.values()),
            "pos_labels": list(POS_ID2LABEL.values())}
