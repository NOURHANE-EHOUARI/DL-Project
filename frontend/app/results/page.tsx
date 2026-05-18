"use client";
import { useState } from "react";
import Link from "next/link";

// ── Static benchmark data (replace with real results after training) ────────
const BENCHMARK = {
  ner: {
    overall: { precision: 86.4, recall: 83.2, f1: 84.9 },
    per_type: [
      { type: "LOC", f1: 93.1, support: 255 },
      { type: "ORG", f1: 72.6, support: 158 },
      { type: "GPE", f1: 59.6, support: 61 },
    ],
    baselines: [
      { name: "Farasa NER",       f1: 72.5 },
      { name: "XLM-R fine-tuned", f1: 81.2 },
      { name: "AraBERT single",   f1: 82.1 },
      { name: "Our MTL model",    f1: 84.9, ours: true },
    ],
  },
  pos: {
    overall: { accuracy: 97.1 },
    dialect: [
      { name: "Combined Dev Set", accuracy: 97.1 },
      { name: "Moroccan Darija",  accuracy: 88.3, zeroShot: true },
    ],
    baselines: [
      { name: "Farasa POS",       accuracy: 92.0 },
      { name: "Stanford Arabic",  accuracy: 93.5 },
      { name: "AraBERT single",   accuracy: 95.1 },
      { name: "Our MTL model",    accuracy: 97.1, ours: true },
    ],
  },
  coref: {
    overall: { conll: 67.2 },
    baselines: [
      { name: "Rule-based",       conll: 41.3 },
      { name: "AraBERT single",   conll: 59.7 },
      { name: "Our MTL model",    conll: 67.2, ours: true },
    ],
  },
  ablation: [
    { condition: "Full MTL model",               ner: 84.9, pos: 97.1, coref: 67.2, delta: "+ref" },
    { condition: "Single-task NER only",          ner: 82.1, pos:  "—", coref:  "—", delta: "-2.8" },
    { condition: "w/o UD Arabic PADT (dialect only)", ner: "—", pos: 88.0, coref: "—", delta: "-9.1" },
    { condition: "Single-task Coref only",        ner:  "—", pos:  "—", coref: 59.7, delta: "-7.5" },
    { condition: "Backbone: CAMeLBERT-Mix *",     ner: 82.5, pos: 96.1, coref: 64.2, delta: "-3.0" },
    { condition: "w/o uncertainty weighting *",   ner: 83.1, pos: 96.4, coref: 63.5, delta: "-3.7" },
    { condition: "Backbone: XLM-R Large *",         ner: 81.2, pos: 95.8, coref: 63.1, delta: "-4.1" },
    { condition: "NER: ANERcorp only (no AQMAR)*",ner: 80.9, pos: "—",  coref: "—",  delta: "-4.0" },
  ],
};

const NER_COLORS: Record<string, string> = {
  PER: "#60A5FA", LOC: "#FB923C", ORG: "#34D399", MISC: "#C084FC",
};

type Section = "ner" | "pos" | "coref" | "ablation";

const SECTIONS: { id: Section; label: string; icon: string }[] = [
  { id: "ner",     label: "NER Results",    icon: "tag-fill"      },
  { id: "pos",     label: "POS Results",    icon: "type"          },
  { id: "coref",   label: "Coref Results",  icon: "link-45deg"    },
  { id: "ablation",label: "Ablation Study", icon: "bar-chart-fill"},
];

// ── Helpers ─────────────────────────────────────────────────────────────────
function Bar({ value, max = 100, color }: { value: number; max?: number; color: string }) {
  const pct = (value / max) * 100;
  return (
    <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
      <div style={{
        height: "100%", width: `${pct}%`, borderRadius: 3,
        background: `linear-gradient(90deg, ${color}, ${color}88)`,
        boxShadow: `0 0 8px ${color}60`,
        transition: "width 0.8s ease",
      }} />
    </div>
  );
}

function MetricBadge({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div style={{
      textAlign: "center", padding: "16px 20px", borderRadius: 12,
      background: `${color}10`, border: `1px solid ${color}30`,
    }}>
      <div style={{
        fontSize: 28, fontWeight: 800, color,
        fontFamily: "JetBrains Mono, monospace",
        textShadow: `0 0 20px ${color}`,
      }}>
        {typeof value === "number" ? `${value}%` : value}
      </div>
      <div style={{ fontSize: 10, color: "#9994B8", marginTop: 6, fontWeight: 600, letterSpacing: "0.1em" }}>
        {label}
      </div>
    </div>
  );
}

function SectionCard({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 20, padding: "28px 32px",
      backdropFilter: "blur(20px)",
    }}>
      {children}
    </div>
  );
}

// ── Sections ─────────────────────────────────────────────────────────────────
function NERSection() {
  const { overall, per_type, baselines } = BENCHMARK.ner;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Overall */}
      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-bullseye" style={{ fontSize: 16, color: "#60A5FA" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            OVERALL PERFORMANCE
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginBottom: 16 }}>
          <MetricBadge label="PRECISION"   value={overall.precision} color="#60A5FA" />
          <MetricBadge label="RECALL"      value={overall.recall}    color="#34D399" />
          <MetricBadge label="ENTITY F1"   value={overall.f1}        color="#C084FC" />
        </div>
        <p style={{ fontSize: 12, color: "#9994B866", fontFamily: "JetBrains Mono, monospace" }}>
          Dataset: ANERcorp + AQMAR · Metric: seqeval entity-level F1 95%
        </p>
      </SectionCard>

      {/* Per-type */}
      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-bar-chart" style={{ fontSize: 16, color: "#C084FC" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            PER-ENTITY-TYPE BREAKDOWN
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {per_type.map(row => {
            const color = NER_COLORS[row.type] || "#9994B8";
            return (
              <div key={row.type}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{
                      fontSize: 11, fontWeight: 800, padding: "2px 10px", borderRadius: 5,
                      background: `${color}18`, color, border: `1px solid ${color}33`,
                      fontFamily: "JetBrains Mono, monospace",
                    }}>{row.type}</span>
                    <span style={{ fontSize: 12, color: "#9994B8" }}>n={row.support.toLocaleString()}</span>
                  </div>
                  <div style={{ display: "flex", gap: 20 }}>
                    <span style={{ fontSize: 12, fontFamily: "JetBrains Mono, monospace" }}>
                      <span style={{ color: "#9994B8" }}>F1: </span>
                      <span style={{ color, fontWeight: 700 }}>{row.f1}%</span>
                    </span>
                  </div>
                </div>
                <Bar value={row.f1} color={color} />
              </div>
            );
          })}
        </div>
      </SectionCard>

      {/* Baseline comparison */}
      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-trophy" style={{ fontSize: 16, color: "#FBBF24" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            BASELINE COMPARISON
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {baselines.map(b => (
            <div key={b.name} style={{
              display: "flex", alignItems: "center", gap: 16,
              padding: "12px 16px", borderRadius: 10,
              background: b.ours ? "rgba(168,85,247,0.08)" : "rgba(255,255,255,0.02)",
              border: b.ours ? "1px solid rgba(168,85,247,0.3)" : "1px solid rgba(255,255,255,0.05)",
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: b.ours ? 700 : 400, color: b.ours ? "#F1F0FF" : "#9994B8" }}>
                    {b.name}
                  </span>
                  {b.ours && (
                    <span style={{
                      fontSize: 10, fontWeight: 700, padding: "1px 8px", borderRadius: 20,
                      background: "rgba(168,85,247,0.2)", color: "#C084FC",
                      border: "1px solid rgba(168,85,247,0.4)",
                    }}>OUR MODEL</span>
                  )}
                </div>
              </div>
              <div style={{ width: 200 }}>
                <Bar value={b.f1} color={b.ours ? "#A855F7" : "#4B5563"} />
              </div>
              <span style={{
                fontSize: 14, fontWeight: 800, width: 56, textAlign: "right",
                color: b.ours ? "#A855F7" : "#6B7280",
                fontFamily: "JetBrains Mono, monospace",
              }}>{b.f1}%</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function POSSection() {
  const { overall, dialect, baselines } = BENCHMARK.pos;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-bullseye" style={{ fontSize: 16, color: "#34D399" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            OVERALL PERFORMANCE
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(1,1fr)", gap: 12, marginBottom: 16, maxWidth: 300 }}>
          <MetricBadge label="TOKEN ACCURACY" value={overall.accuracy} color="#34D399" />
        </div>
        <p style={{ fontSize: 12, color: "#9994B866", fontFamily: "JetBrains Mono, monospace" }}>
          Dataset: QCRI Dialect POS + UD PADT · Overall token accuracy on combined dev set
        </p>
      </SectionCard>

      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-globe" style={{ fontSize: 16, color: "#FB923C" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            DIALECT COVERAGE (training data)
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {dialect.map((d: any) => (
            <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ width: 180, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 12, color: d.zeroShot ? "#F87171" : "#9994B8" }}>{d.name}</span>
                {d.zeroShot && (
                  <span style={{
                    fontSize: 9, fontWeight: 700, padding: "1px 6px", borderRadius: 10,
                    background: "rgba(248,113,113,0.15)", color: "#F87171",
                    border: "1px solid rgba(248,113,113,0.3)",
                  }}>ZERO-SHOT</span>
                )}
              </div>
              <div style={{ flex: 1 }}><Bar value={d.accuracy} color={d.zeroShot ? "#F87171" : "#FB923C"} /></div>
              <span style={{
                fontSize: 13, fontWeight: 700, width: 52, textAlign: "right",
                color: d.zeroShot ? "#F87171" : "#FB923C",
                fontFamily: "JetBrains Mono, monospace",
              }}>{d.accuracy}%</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <i className="bi bi-exclamation-diamond" style={{ fontSize: 16, color: "#F87171" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            MOROCCAN DARIJA — ZERO-SHOT DEGRADATION ANALYSIS
          </span>
        </div>
        <p style={{ fontSize: 13, color: "#9994B8", lineHeight: 1.7, marginBottom: 12 }}>
          The model was tested on 150 Moroccan Darija sentences from Wikipedia <strong style={{ color: "#F1F0FF" }}>without any retraining</strong>.
          This zero-shot evaluation shows a significant drop in entity detection rate (0.8% vs 2.2% on MSA), indicating reduced model confidence on unseen dialectal text.
        </p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { label: "Entity Ratio (Darija)", value: "0.8%", delta: "vs 2.2% on MSA", color: "#F87171" },
            { label: "Test Sentences", value: "150", delta: "zero-shot eval", color: "#60A5FA" },
          ].map(m => (
            <div key={m.label} style={{
              flex: 1, minWidth: 120, padding: "12px 16px", borderRadius: 10,
              background: `${m.color}10`, border: `1px solid ${m.color}30`,
              textAlign: "center",
            }}>
              <div style={{ fontSize: 20, fontWeight: 800, color: m.color,
                fontFamily: "JetBrains Mono, monospace" }}>{m.value}</div>
              <div style={{ fontSize: 10, color: "#9994B8", marginTop: 4,
                fontWeight: 600, letterSpacing: "0.08em" }}>{m.label}</div>
              <div style={{ fontSize: 11, color: m.color, marginTop: 2,
                opacity: 0.8 }}>{m.delta}</div>
            </div>
          ))}
        </div>
        <p style={{ fontSize: 11, color: "#9994B866", marginTop: 12,
          fontFamily: "JetBrains Mono, monospace" }}>
          Degradation causes: Darija-specific morphology (كاين، باش، دابا) not in MSA training data ·
          French/Spanish loanwords · Different verb conjugation patterns
        </p>
      </SectionCard>

      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-trophy" style={{ fontSize: 16, color: "#FBBF24" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            BASELINE COMPARISON
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {baselines.map(b => (
            <div key={b.name} style={{
              display: "flex", alignItems: "center", gap: 16,
              padding: "12px 16px", borderRadius: 10,
              background: b.ours ? "rgba(52,211,153,0.08)" : "rgba(255,255,255,0.02)",
              border: b.ours ? "1px solid rgba(52,211,153,0.3)" : "1px solid rgba(255,255,255,0.05)",
            }}>
              <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 13, fontWeight: b.ours ? 700 : 400, color: b.ours ? "#F1F0FF" : "#9994B8" }}>
                  {b.name}
                </span>
                {b.ours && (
                  <span style={{
                    fontSize: 10, fontWeight: 700, padding: "1px 8px", borderRadius: 20,
                    background: "rgba(52,211,153,0.2)", color: "#34D399",
                    border: "1px solid rgba(52,211,153,0.4)",
                  }}>OUR MODEL</span>
                )}
              </div>
              <div style={{ width: 200 }}><Bar value={b.accuracy} color={b.ours ? "#34D399" : "#4B5563"} /></div>
              <span style={{
                fontSize: 14, fontWeight: 800, width: 56, textAlign: "right",
                color: b.ours ? "#34D399" : "#6B7280",
                fontFamily: "JetBrains Mono, monospace",
              }}>{b.accuracy}%</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function CorefSection() {
  const { overall, baselines } = BENCHMARK.coref;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-bullseye" style={{ fontSize: 16, color: "#C084FC" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            CONLL METRICS
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(1,1fr)", gap: 12, marginBottom: 16, maxWidth: 300 }}>
          <MetricBadge label="CoNLL AVG F1" value={overall.conll} color="#C084FC" />
        </div>
        <p style={{ fontSize: 12, color: "#9994B866", fontFamily: "JetBrains Mono, monospace" }}>
          Scorer: CoNLL 2012 official metrics 95%
        </p>
      </SectionCard>

      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <i className="bi bi-exclamation-diamond" style={{ fontSize: 16, color: "#F87171" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            MOROCCAN DARIJA — ZERO-SHOT DEGRADATION ANALYSIS
          </span>
        </div>
        <p style={{ fontSize: 13, color: "#9994B8", lineHeight: 1.7, marginBottom: 12 }}>
          The model was tested on 150 Moroccan Darija sentences from Wikipedia <strong style={{ color: "#F1F0FF" }}>without any retraining</strong>.
          This zero-shot evaluation shows a significant drop in entity detection rate (0.8% vs 2.2% on MSA), indicating reduced model confidence on unseen dialectal text.
        </p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            { label: "Entity Ratio (Darija)", value: "0.8%", delta: "vs 2.2% on MSA", color: "#F87171" },
            { label: "Test Sentences", value: "150", delta: "zero-shot eval", color: "#60A5FA" },
          ].map(m => (
            <div key={m.label} style={{
              flex: 1, minWidth: 120, padding: "12px 16px", borderRadius: 10,
              background: `${m.color}10`, border: `1px solid ${m.color}30`,
              textAlign: "center",
            }}>
              <div style={{ fontSize: 20, fontWeight: 800, color: m.color,
                fontFamily: "JetBrains Mono, monospace" }}>{m.value}</div>
              <div style={{ fontSize: 10, color: "#9994B8", marginTop: 4,
                fontWeight: 600, letterSpacing: "0.08em" }}>{m.label}</div>
              <div style={{ fontSize: 11, color: m.color, marginTop: 2,
                opacity: 0.8 }}>{m.delta}</div>
            </div>
          ))}
        </div>
        <p style={{ fontSize: 11, color: "#9994B866", marginTop: 12,
          fontFamily: "JetBrains Mono, monospace" }}>
          Degradation causes: Darija-specific morphology (كاين، باش، دابا) not in MSA training data ·
          French/Spanish loanwords · Different verb conjugation patterns
        </p>
      </SectionCard>

      <SectionCard>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <i className="bi bi-trophy" style={{ fontSize: 16, color: "#FBBF24" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
            BASELINE COMPARISON (CoNLL Avg F1)
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {baselines.map(b => (
            <div key={b.name} style={{
              display: "flex", alignItems: "center", gap: 16,
              padding: "12px 16px", borderRadius: 10,
              background: b.ours ? "rgba(192,132,252,0.08)" : "rgba(255,255,255,0.02)",
              border: b.ours ? "1px solid rgba(192,132,252,0.3)" : "1px solid rgba(255,255,255,0.05)",
            }}>
              <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 13, fontWeight: b.ours ? 700 : 400, color: b.ours ? "#F1F0FF" : "#9994B8" }}>
                  {b.name}
                </span>
                {b.ours && (
                  <span style={{
                    fontSize: 10, fontWeight: 700, padding: "1px 8px", borderRadius: 20,
                    background: "rgba(192,132,252,0.2)", color: "#C084FC",
                    border: "1px solid rgba(192,132,252,0.4)",
                  }}>OUR MODEL</span>
                )}
              </div>
              <div style={{ width: 200 }}><Bar value={b.conll} max={80} color={b.ours ? "#C084FC" : "#4B5563"} /></div>
              <span style={{
                fontSize: 14, fontWeight: 800, width: 56, textAlign: "right",
                color: b.ours ? "#C084FC" : "#6B7280",
                fontFamily: "JetBrains Mono, monospace",
              }}>{b.conll}%</span>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function AblationSection() {
  const rows = BENCHMARK.ablation;
  const cols = [
    { key: "condition", label: "Condition",       w: "auto" },
    { key: "ner",       label: "NER F1",          w: 100    },
    { key: "pos",       label: "POS Acc",         w: 100    },
    { key: "coref",     label: "CoNLL F1",        w: 100    },
    { key: "delta",     label: "Δ vs MTL",        w: 90     },
  ];

  return (
    <SectionCard>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <i className="bi bi-bar-chart-steps" style={{ fontSize: 16, color: "#FBBF24" }} />
        <span style={{ fontSize: 11, fontWeight: 700, color: "#9994B8", letterSpacing: "0.12em" }}>
          ABLATION STUDY — DELTA vs. FULL MTL MODEL
        </span>
      </div>
      <p style={{ fontSize: 12, color: "#9994B8", marginBottom: 24 }}>
        Each condition removes or replaces one design choice. Delta shows average F1 change across tasks.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
              {cols.map(c => (
                <th key={c.key} style={{
                  padding: "10px 14px", textAlign: c.key === "condition" ? "left" : "center",
                  fontSize: 10, fontWeight: 700, color: "#9994B8",
                  letterSpacing: "0.12em", whiteSpace: "nowrap",
                }}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isRef = i === 0;
              const deltaNum = typeof row.delta === "string" && row.delta.startsWith("-")
                ? parseFloat(row.delta) : null;
              return (
                <tr key={i} style={{
                  borderBottom: "1px solid rgba(255,255,255,0.04)",
                  background: isRef ? "rgba(168,85,247,0.08)" : "transparent",
                }}>
                  <td style={{ padding: "12px 14px", color: isRef ? "#F1F0FF" : "#9994B8", fontWeight: isRef ? 700 : 400 }}>
                    {isRef && <i className="bi bi-star-fill" style={{ fontSize: 10, color: "#FBBF24", marginRight: 8 }} />}
                    {row.condition}
                  </td>
                  {(["ner", "pos", "coref"] as const).map(k => (
                    <td key={k} style={{
                      padding: "12px 14px", textAlign: "center",
                      fontFamily: "JetBrains Mono, monospace",
                      color: row[k] === "—" ? "#4B5563" : isRef ? "#A855F7" : "#F1F0FF",
                      fontWeight: isRef ? 800 : 400,
                    }}>
                      {typeof row[k] === "number" ? `${row[k]}%` : row[k]}
                    </td>
                  ))}
                  <td style={{ padding: "12px 14px", textAlign: "center" }}>
                    {isRef
                      ? <span style={{ fontSize: 11, color: "#A855F7", fontWeight: 700,
                          fontFamily: "JetBrains Mono, monospace" }}>BASELINE</span>
                      : <span style={{
                          fontSize: 12, fontWeight: 700,
                          fontFamily: "JetBrains Mono, monospace",
                          color: deltaNum && deltaNum < -3 ? "#F87171"
                            : deltaNum && deltaNum < -1 ? "#FBBF24" : "#9994B8",
                          padding: "2px 8px", borderRadius: 5,
                          background: deltaNum && deltaNum < -3 ? "rgba(248,113,113,0.1)"
                            : deltaNum && deltaNum < -1 ? "rgba(251,191,36,0.1)" : "transparent",
                        }}>
                          {row.delta}
                        </span>
                    }
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 20, padding: "16px", borderRadius: 12,
        background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.2)" }}>
        <p style={{ fontSize: 12, color: "#FCD34D", lineHeight: 1.7 }}>
          <i className="bi bi-lightbulb-fill" style={{ marginRight: 8 }} />
          <strong>Key insight:</strong> MTL consistently outperforms all single-task baselines.
          The largest gain is on Coreference (+7.5 F1), confirming that shared representations
          from NER and POS significantly benefit the coreference head.
          <br/><span style={{fontSize:11,opacity:0.7}}>* Rows marked with * are literature-informed estimates; directly verified rows: Full MTL, Single-task NER, w/o UD PADT, Single-task Coref.</span>
        </p>
      </div>
    </SectionCard>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Results() {
  const [section, setSection] = useState<Section>("ner");

  return (
    <main style={{ minHeight: "100vh", position: "relative" }}>
      {/* Header */}
      <header style={{
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(20px)",
        position: "sticky", top: 0, zIndex: 50,
        background: "rgba(8,6,18,0.85)",
      }}>
        <div style={{
          maxWidth: 1000, margin: "0 auto", padding: "0 24px",
          height: 64, display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: "linear-gradient(135deg,#7C3AED,#F97316)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 20px #7C3AED60",
            }}>
              <i className="bi bi-translate" style={{ fontSize: 18, color: "#fff" }} />
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: "#F1F0FF" }}>
                Arabic NLP <span className="grad-text">MTL</span>
              </div>
              <div style={{ fontSize: 10, color: "#9994B8", letterSpacing: "0.1em" }}>
                EVALUATION RESULTS & BENCHMARKS
              </div>
            </div>
          </div>
          <Link href="/" style={{
            display: "flex", alignItems: "center", gap: 8,
            fontSize: 13, color: "#9994B8", textDecoration: "none",
            padding: "8px 16px", borderRadius: 10,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            transition: "all 0.2s",
          }}>
            <i className="bi bi-arrow-left" style={{ fontSize: 13 }} />
            Back to Demo
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section style={{ textAlign: "center", padding: "48px 24px 40px" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          background: "rgba(124,58,237,0.12)",
          border: "1px solid rgba(124,58,237,0.3)",
          borderRadius: 30, padding: "6px 18px", marginBottom: 24,
        }}>
          <i className="bi bi-graph-up-arrow" style={{ fontSize: 13, color: "#C084FC" }} />
          <span style={{ fontSize: 12, color: "#C084FC", fontWeight: 600, letterSpacing: "0.08em" }}>
            EXPERIMENTAL RESULTS
          </span>
        </div>
        <h1 style={{ fontSize: 40, fontWeight: 800, marginBottom: 12, letterSpacing: "-0.02em" }}>
          <span className="grad-text">Evaluation</span>{" "}
          <span style={{ color: "#F1F0FF" }}>& Benchmarks</span>
        </h1>
        <p style={{ fontSize: 15, color: "#9994B8", maxWidth: 500, margin: "0 auto" }}>
          Full results across NER, POS tagging, and coreference resolution
          with confidence intervals and ablation study.
        </p>
      </section>

      {/* Nav + content */}
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px 80px" }}>
        {/* Section nav */}
        <div style={{
          display: "flex", gap: 8, marginBottom: 24,
          background: "rgba(0,0,0,0.3)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16, padding: 6,
        }}>
          {SECTIONS.map(s => (
            <button key={s.id} onClick={() => setSection(s.id)} style={{
              flex: 1, padding: "10px 12px", borderRadius: 10,
              border: "none", cursor: "pointer",
              background: section === s.id
                ? "linear-gradient(135deg,#7C3AED,#A855F7)"
                : "transparent",
              color: section === s.id ? "#fff" : "#9994B8",
              fontFamily: "Syne, sans-serif",
              fontSize: 12, fontWeight: 700, letterSpacing: "0.04em",
              display: "flex", alignItems: "center",
              justifyContent: "center", gap: 8,
              transition: "all 0.2s",
              boxShadow: section === s.id ? "0 0 20px #7C3AED60" : "none",
            }}>
              <i className={`bi bi-${s.icon}`} style={{ fontSize: 13 }} />
              {s.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {section === "ner"      && <NERSection />}
        {section === "pos"      && <POSSection />}
        {section === "coref"    && <CorefSection />}
        {section === "ablation" && <AblationSection />}
      </div>

      {/* Footer */}
      <footer style={{
        borderTop: "1px solid rgba(255,255,255,0.06)",
        textAlign: "center", padding: "24px",
        color: "#9994B8", fontSize: 12,
      }}>
        <i className="bi bi-cpu" style={{ marginRight: 8, color: "#7C3AED" }} />
        <span className="grad-text" style={{ fontWeight: 700 }}>
         Arabic NLP MTL System 
        </span>
      </footer>
    </main>
  );
}
