"use client";
import { useState } from "react";

interface POSToken { token: string; pos_tag: string; morph_category?: string; }
interface Props { tokens: POSToken[]; }

const TAG_COLORS: Record<string, [string, string]> = {
  NOUN: ['#60A5FA', '#1E3A5F'], V: ['#F87171', '#3B1515'],
  ADJ:  ['#34D399', '#0F2D1E'], PREP: ['#C084FC', '#2D1B4E'],
  PRON: ['#FBBF24', '#3B2D00'], CONJ: ['#38BDF8', '#0C2233'],
  PART: ['#FB923C', '#3B1F0C'], ADV:  ['#A3E635', '#1E2D0C'],
  NUM:  ['#94A3B8', '#1E2430'], PUNC: ['#64748B', '#1A1F28'],
  DET:  ['#E879F9', '#2D0E33'], OTHER:['#71717A','#1C1C1E'],
};

function tagColor(tag: string): [string, string] {
  const base = tag.split('+')[0];
  return TAG_COLORS[base] || TAG_COLORS.OTHER;
}

export default function POSDisplay({ tokens }: Props) {
  const [view, setView] = useState<'tokens'|'stats'>('tokens');
  const counts: Record<string, number> = {};
  tokens.forEach(t => { counts[t.pos_tag] = (counts[t.pos_tag] || 0) + 1; });

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <span style={{ fontSize: 11, color: '#9994B8', fontWeight: 600, letterSpacing: '0.12em' }}>
          PART-OF-SPEECH TAGGING
        </span>
        <div style={{
          display: 'flex', background: 'rgba(0,0,0,0.4)',
          borderRadius: 10, padding: 3, border: '1px solid rgba(255,255,255,0.08)',
        }}>
          {(['tokens','stats'] as const).map(v => (
            <button key={v} onClick={() => setView(v)} style={{
              padding: '5px 16px', borderRadius: 8, border: 'none',
              cursor: 'pointer', fontSize: 12, fontWeight: 600,
              fontFamily: 'Syne, sans-serif', letterSpacing: '0.05em',
              background: view === v ? 'linear-gradient(135deg,#7C3AED,#A855F7)' : 'transparent',
              color: view === v ? '#fff' : '#9994B8',
              transition: 'all 0.2s',
            }}>
              {v === 'tokens' ? 'TOKENS' : 'STATS'}
            </button>
          ))}
        </div>
      </div>

      {view === 'tokens' ? (
        <div dir="rtl" style={{
          display: 'flex', flexWrap: 'wrap', gap: 16,
          padding: '20px 24px', background: 'rgba(0,0,0,0.3)',
          borderRadius: 12, border: '1px solid rgba(255,255,255,0.06)',
          minHeight: 80,
        }}>
          {tokens.length === 0
            ? <span style={{ color: '#9994B8' }}>No tokens.</span>
            : tokens.map((tok, i) => {
                const [fg, bg] = tagColor(tok.pos_tag);
                return (
                  <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}
                    title={tok.morph_category || tok.pos_tag}>
                    <span style={{
                      fontSize: 10, fontWeight: 800, letterSpacing: '0.08em',
                      background: bg, color: fg,
                      border: `1px solid ${fg}44`,
                      padding: '2px 8px', borderRadius: 5,
                      boxShadow: `0 0 8px ${fg}30`,
                      fontFamily: 'JetBrains Mono, monospace',
                    }}>
                      {tok.pos_tag.length > 12 ? tok.pos_tag.slice(0,11)+'…' : tok.pos_tag}
                    </span>
                    <span style={{
                      fontFamily: 'Noto Naskh Arabic, serif',
                      fontSize: 17, color: '#E2E0F0',
                    }}>
                      {tok.token}
                    </span>
                  </div>
                );
              })}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([tag, count]) => {
            const [fg] = tagColor(tag);
            const pct = Math.round(count / tokens.length * 100);
            return (
              <div key={tag} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{
                  fontSize: 11, fontWeight: 700, width: 110, flexShrink: 0,
                  fontFamily: 'JetBrains Mono, monospace', color: fg,
                }}>
                  {tag}
                </span>
                <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3 }}>
                  <div style={{
                    height: '100%', borderRadius: 3, width: `${pct}%`,
                    background: `linear-gradient(90deg, ${fg}, ${fg}88)`,
                    boxShadow: `0 0 8px ${fg}60`,
                    transition: 'width 0.5s ease',
                  }} />
                </div>
                <span style={{ fontSize: 11, color: '#9994B8', width: 50, textAlign: 'right',
                  fontFamily: 'JetBrains Mono, monospace' }}>
                  {count} ({pct}%)
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
