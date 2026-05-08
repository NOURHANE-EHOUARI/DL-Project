"use client";
import { useState } from "react";

interface Span { text: string; type: string; start: number; end: number; }
interface Props { text: string; spans: Span[]; }

const COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  PER:  { bg: '#60A5FA18', border: '#60A5FA', text: '#93C5FD', glow: '#60A5FA40' },
  ORG:  { bg: '#34D39918', border: '#34D399', text: '#6EE7B7', glow: '#34D39940' },
  LOC:  { bg: '#FB923C18', border: '#FB923C', text: '#FCA76A', glow: '#FB923C40' },
  GPE:  { bg: '#FB923C18', border: '#FB923C', text: '#FCA76A', glow: '#FB923C40' },
  MISC: { bg: '#C084FC18', border: '#C084FC', text: '#D8B4FE', glow: '#C084FC40' },
  DATE: { bg: '#FBBF2418', border: '#FBBF24', text: '#FCD34D', glow: '#FBBF2440' },
};
const LABELS: Record<string, string> = {
  PER:'PERSON', ORG:'ORGANIZATION', LOC:'LOCATION',
  GPE:'LOCATION', MISC:'MISC', DATE:'DATE',
};

export default function NERAnnotation({ text, spans }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  const tokens = text.split(' ');
  const spanMap: Record<number, Span> = {};
  spans.forEach(s => { for (let i = s.start; i <= s.end; i++) spanMap[i] = s; });

  const typeCounts: Record<string, number> = {};
  spans.forEach(s => { typeCounts[s.type] = (typeCounts[s.type] || 0) + 1; });

  const rendered: React.ReactNode[] = [];
  let i = 0;
  while (i < tokens.length) {
    if (spanMap[i]) {
      const span = spanMap[i];
      const c = COLORS[span.type] || COLORS.MISC;
      const idx = spans.indexOf(span);
      rendered.push(
        <span key={i} style={{ position: 'relative', display: 'inline' }}>
          <mark
            className="ner-mark"
            style={{
              background: c.bg,
              borderBottom: `2px solid ${c.border}`,
              color: c.text,
              boxShadow: hovered === idx ? `0 0 12px ${c.glow}` : 'none',
              transition: 'all 0.2s',
            }}
            onMouseEnter={() => setHovered(idx)}
            onMouseLeave={() => setHovered(null)}
          >
            {tokens.slice(span.start, span.end + 1).join(' ')}
          </mark>
          {hovered === idx && (
            <span style={{
              position: 'absolute', bottom: '110%', left: '50%',
              transform: 'translateX(-50%)',
              background: c.border, color: '#000',
              fontSize: 10, fontWeight: 800,
              padding: '3px 8px', borderRadius: 6,
              letterSpacing: '0.1em', whiteSpace: 'nowrap',
              zIndex: 10, pointerEvents: 'none',
            }}>
              {LABELS[span.type] || span.type}
            </span>
          )}
        </span>
      );
      i = span.end + 1;
    } else {
      rendered.push(
        <span key={i} style={{ color: '#E2E0F0', margin: '0 1px',
          fontFamily: 'Noto Naskh Arabic, serif' }}>
          {tokens[i]}
        </span>
      );
      i++;
    }
  }

  return (
    <div>
      {/* Badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#9994B8', fontWeight: 600, letterSpacing: '0.12em' }}>
            NAMED ENTITY RECOGNITION
          </span>
        </div>
        <span style={{
          fontSize: 12, fontWeight: 700,
          background: 'rgba(124,58,237,0.15)',
          border: '1px solid rgba(124,58,237,0.3)',
          color: '#C084FC', padding: '3px 12px', borderRadius: 20,
        }}>
          {spans.length} entities
        </span>
      </div>

      {/* Annotated text */}
      <div dir="rtl" style={{
        lineHeight: 2.4, fontSize: 20,
        fontFamily: 'Noto Naskh Arabic, serif',
        padding: '20px 24px',
        background: 'rgba(0,0,0,0.3)',
        borderRadius: 12,
        border: '1px solid rgba(255,255,255,0.06)',
        marginBottom: 20,
        minHeight: 80,
      }}>
        {spans.length === 0
          ? <span style={{ color: '#9994B8', fontSize: 16 }}>No entities detected.</span>
          : rendered}
      </div>

      {/* Legend */}
      {spans.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {Object.entries(typeCounts).map(([type, count]) => {
            const c = COLORS[type] || COLORS.MISC;
            return (
              <div key={type} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: c.bg, border: `1px solid ${c.border}33`,
                borderRadius: 8, padding: '6px 14px',
              }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%',
                  background: c.border, boxShadow: `0 0 6px ${c.glow}`, display: 'inline-block' }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: c.text, letterSpacing: '0.08em' }}>
                  {LABELS[type] || type}
                </span>
                <span style={{ fontSize: 11, color: c.text, opacity: 0.6 }}>×{count}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
