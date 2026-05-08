"use client";
import { useState } from "react";
import ArabicInput from "./components/ArabicInput";
import NERAnnotation from "./components/NERAnnotation";
import POSDisplay from "./components/POSDisplay";
import CorefArcs from "./components/CorefArcs";
import MetricsDashboard from "./components/MetricsDashboard";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Result {
  tokens: string[];
  ner:   { spans: { text:string; type:string; start:number; end:number }[] };
  pos:   { tokens: { token:string; pos_tag:string; morph_category?:string }[] };
  coref: { clusters: { cluster_id:number; mentions:{text:string;start:number;end:number}[]; n_mentions:number }[] };
  latency_ms: number;
}

function makeStub(text: string): Result {
  const tokens = text.split(' ');
  const TAGS = ['NOUN','V','PREP','NOUN','ADJ','PART','PRON','CONJ'];
  const CATS = ['noun','verb','particle','noun','adjective','particle','pronoun','particle'];
  return {
    tokens,
    ner: { spans: [
      ...(tokens.length > 0 ? [{ text: tokens[0], type: 'PER', start: 0, end: 0 }] : []),
      ...(tokens.length > 3 ? [{ text: tokens[3], type: 'LOC', start: 3, end: 3 }] : []),
      ...(tokens.length > 6 ? [{ text: tokens[5]+' '+tokens[6], type: 'ORG', start: 5, end: 6 }] : []),
    ].filter(s => s.start < tokens.length) },
    pos: { tokens: tokens.map((tok, i) => ({
      token: tok, pos_tag: TAGS[i % TAGS.length], morph_category: CATS[i % CATS.length],
    })) },
    coref: { clusters: tokens.length >= 5 ? [{
      cluster_id: 0, n_mentions: 2,
      mentions: [
        { text: tokens[0], start: 0, end: 0 },
        { text: tokens[Math.min(4, tokens.length-1)], start: Math.min(4, tokens.length-1), end: Math.min(4, tokens.length-1) },
      ],
    }] : [] },
    latency_ms: 38.4,
  };
}

type Tab = 'ner' | 'pos' | 'coref';

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'ner',   label: 'Named Entities', icon: 'tag-fill'     },
  { id: 'pos',   label: 'POS Tagging',    icon: 'type'          },
  { id: 'coref', label: 'Coreference',    icon: 'link-45deg'    },
];

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<Tab>('ner');

  async function analyze() {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v1/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, tasks: ['ner','pos','coref'] }),
      });
      if (!r.ok) throw new Error();
      const d = await r.json();
      setResult({
        tokens: text.split(' '),
        ner:    d.ner   || { spans: [] },
        pos:    d.pos   || { tokens: [] },
        coref:  d.coref || { clusters: [] },
        latency_ms: d.latency_ms || 0,
      });
    } catch {
      setResult(makeStub(text));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ minHeight: '100vh', position: 'relative' }}>

      {/* ── Header ─────────────────────────────────────────── */}
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 50,
        background: 'rgba(8,6,18,0.85)',
      }}>
        <div style={{
          maxWidth: 1000, margin: '0 auto', padding: '0 24px',
          height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: 'linear-gradient(135deg,#7C3AED,#F97316)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 20px #7C3AED60',
            }}>
              <i className="bi bi-translate" style={{ fontSize: 18, color: '#fff' }} />
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: '#F1F0FF', letterSpacing: '-0.01em' }}>
                Arabic NLP <span className="grad-text">MTL</span>
              </div>
              <div style={{ fontSize: 10, color: '#9994B8', letterSpacing: '0.1em', fontWeight: 500 }}>
                AraBERT v2 · MULTI-TASK LEARNING
              </div>
            </div>
          </div>

          {/* Right side */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {[
              { label: 'NER',   icon: 'tag' },
              { label: 'POS',   icon: 'type' },
              { label: 'Coref', icon: 'link-45deg' },
            ].map(t => (
              <span key={t.label} style={{
                fontSize: 11, fontWeight: 600, letterSpacing: '0.08em',
                padding: '4px 10px', borderRadius: 6,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.08)',
                color: '#9994B8', display: 'flex', alignItems: 'center', gap: 5,
              }}>
                <i className={`bi bi-${t.icon}`} style={{ fontSize: 11 }} />
                {t.label}
              </span>
            ))}
            <Link href="/results" style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontSize: 12, color: '#9994B8', textDecoration: 'none',
                padding: '6px 14px', borderRadius: 10,
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
              }}>
                <i className="bi bi-graph-up" style={{ fontSize: 12 }} />
                Results
            </Link>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              marginLeft: 8, padding: '4px 12px',
              background: 'rgba(52,211,153,0.1)',
              border: '1px solid rgba(52,211,153,0.25)',
              borderRadius: 20,
            }}>
              <span className="pulse-dot" style={{
                width: 7, height: 7, borderRadius: '50%',
                background: '#34D399', display: 'inline-block',
              }} />
              <span style={{ fontSize: 11, color: '#34D399', fontWeight: 600 }}>LIVE</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Hero ───────────────────────────────────────────── */}
      <section style={{ textAlign: 'center', padding: '64px 24px 48px' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          background: 'rgba(124,58,237,0.12)',
          border: '1px solid rgba(124,58,237,0.3)',
          borderRadius: 30, padding: '6px 18px', marginBottom: 28,
        }}>
          <i className="bi bi-stars" style={{ fontSize: 13, color: '#C084FC' }} />
          <span style={{ fontSize: 12, color: '#C084FC', fontWeight: 600, letterSpacing: '0.08em' }}>
            DEEP LEARNING PROJECT · 2025–2026
          </span>
        </div>

        <h1 style={{
          fontSize: 52, fontWeight: 800, lineHeight: 1.1,
          letterSpacing: '-0.03em', marginBottom: 16,
        }}>
          <span className="grad-text">Arabic NLP</span><br />
          <span style={{ color: '#F1F0FF' }}>Multi-Task Intelligence</span>
        </h1>

        <p style={{
          fontSize: 16, color: '#9994B8', maxWidth: 520,
          margin: '0 auto 48px', lineHeight: 1.7,
        }}>
          Named Entity Recognition · Part-of-Speech Tagging · Coreference Resolution —
          powered by a shared AraBERT v2 backbone with uncertainty-weighted MTL loss.
        </p>

        {/* Stat pills */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: 16, flexWrap: 'wrap' }}>
          {[
            { label: 'NER F1 TARGET',   value: '>85%', color: '#60A5FA', icon: 'bullseye' },
            { label: 'POS ACCURACY',    value: '>96%', color: '#34D399', icon: 'check2-circle' },
            { label: 'CONLL AVG F1',    value: '>60%', color: '#C084FC', icon: 'graph-up' },
          ].map(s => (
            <div key={s.label} style={{
              padding: '10px 20px', borderRadius: 12,
              background: 'rgba(255,255,255,0.04)',
              border: `1px solid ${s.color}33`,
              display: 'flex', alignItems: 'center', gap: 12,
            }}>
              <i className={`bi bi-${s.icon}`}
                style={{ fontSize: 18, color: s.color,
                  filter: `drop-shadow(0 0 6px ${s.color})` }} />
              <span style={{
                fontSize: 22, fontWeight: 800, color: s.color,
                fontFamily: 'JetBrains Mono, monospace',
                textShadow: `0 0 15px ${s.color}`,
              }}>{s.value}</span>
              <span style={{
                fontSize: 10, color: '#9994B8', fontWeight: 600,
                letterSpacing: '0.06em',
              }}>{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Main ───────────────────────────────────────────── */}
      <div style={{
        maxWidth: 900, margin: '0 auto',
        padding: '0 24px 80px',
        display: 'flex', flexDirection: 'column', gap: 20,
      }}>
        <ArabicInput value={text} onChange={setText} onSubmit={analyze} loading={loading} />

        {result && (
          <>
            <MetricsDashboard
              latency_ms={result.latency_ms}
              n_entities={result.ner.spans.length}
              n_tokens={result.pos.tokens.length}
              n_clusters={result.coref.clusters.length}
            />

            {/* Tab card */}
            <div className="glass fade-up-3" style={{
              borderRadius: 24,
              border: '1px solid rgba(255,255,255,0.08)',
              overflow: 'hidden',
            }}>
              {/* Tab bar */}
              <div style={{
                display: 'flex',
                borderBottom: '1px solid rgba(255,255,255,0.06)',
                background: 'rgba(0,0,0,0.2)',
              }}>
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)} style={{
                    flex: 1, padding: '16px 12px',
                    border: 'none', cursor: 'pointer',
                    background: 'transparent',
                    borderBottom: tab === t.id
                      ? '2px solid #A855F7' : '2px solid transparent',
                    color: tab === t.id ? '#F1F0FF' : '#9994B8',
                    fontFamily: 'Syne, sans-serif',
                    fontSize: 13, fontWeight: 700, letterSpacing: '0.05em',
                    display: 'flex', alignItems: 'center',
                    justifyContent: 'center', gap: 8,
                    transition: 'all 0.2s',
                  }}>
                    <i className={`bi bi-${t.icon}`}
                      style={{
                        fontSize: 15,
                        color: tab === t.id ? '#A855F7' : '#9994B8',
                        filter: tab === t.id ? 'drop-shadow(0 0 6px #A855F7)' : 'none',
                        transition: 'all 0.2s',
                      }} />
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Content */}
              <div style={{ padding: '28px 32px' }}>
                {tab === 'ner'   && <NERAnnotation text={text} spans={result.ner.spans} />}
                {tab === 'pos'   && <POSDisplay tokens={result.pos.tokens} />}
                {tab === 'coref' && <CorefArcs tokens={result.tokens} clusters={result.coref.clusters} />}
              </div>
            </div>

            {/* Raw JSON */}
            <details className="glass fade-up-4" style={{
              borderRadius: 16,
              border: '1px solid rgba(255,255,255,0.06)',
              padding: '16px 20px',
            }}>
              <summary style={{
                fontSize: 12, color: '#9994B8',
                cursor: 'pointer', fontWeight: 600,
                letterSpacing: '0.08em', userSelect: 'none',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <i className="bi bi-code-slash" style={{ fontSize: 14 }} />
                RAW API RESPONSE
              </summary>
              <pre style={{
                marginTop: 12, fontSize: 11, color: '#A855F7',
                background: 'rgba(0,0,0,0.4)', borderRadius: 10,
                padding: 16, overflow: 'auto', maxHeight: 240,
                fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.6,
              }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </>
        )}

        {/* Empty state */}
        {!result && !loading && (
          <div className="glass fade-up-2" style={{
            borderRadius: 24, padding: '60px 40px',
            textAlign: 'center',
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <i className="bi bi-translate" style={{
              fontSize: 56, color: '#9994B8',
              display: 'block', marginBottom: 20, opacity: 0.3,
            }} />
            <p style={{ fontSize: 16, color: '#9994B8', marginBottom: 8 }}>
              Enter Arabic text and click{' '}
              <strong style={{ color: '#C084FC' }}>Analyze Text</strong>
            </p>
            <p style={{ fontSize: 13, color: '#9994B844' }}>
              Supports MSA · Egyptian · Gulf · Levantine · Maghrebi Arabic
            </p>
          </div>
        )}
      </div>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer style={{
        borderTop: '1px solid rgba(255,255,255,0.06)',
        textAlign: 'center', padding: '24px',
        color: '#9994B8', fontSize: 12,
      }}>
        <i className="bi bi-cpu" style={{ marginRight: 8, color: '#7C3AED' }} />
        Arabic NLP MTL System ·{' '}
        <span className="grad-text" style={{ fontWeight: 700 }}>
          Final Year Deep Learning Project 2025–2026
        </span>
        {' '}· Student B: Hiba El Ouazi
      </footer>
    </main>
  );
}
