"use client";
import { useState } from "react";

interface Mention { text: string; start: number; end: number; }
interface Cluster { cluster_id: number; mentions: Mention[]; n_mentions: number; }
interface Props { tokens: string[]; clusters: Cluster[]; }

const PALETTE = ['#A855F7','#F97316','#60A5FA','#34D399','#FBBF24','#F87171','#38BDF8','#E879F9'];

export default function CorefArcs({ tokens, clusters }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  const W = 700, H = 200, MARGIN = { top: 90, left: 16, right: 16 };
  const tokW = Math.max(36, (W - MARGIN.left - MARGIN.right) / Math.max(tokens.length, 1));
  const tokX = (i: number) => MARGIN.left + i * tokW + tokW / 2;

  const mentionColor: Record<number, string> = {};
  clusters.forEach((c, ci) => {
    c.mentions.forEach(m => {
      for (let i = m.start; i <= m.end; i++) mentionColor[i] = PALETTE[ci % PALETTE.length];
    });
  });

  if (!clusters.length) return (
    <div style={{ textAlign: 'center', padding: '40px 0', color: '#9994B8' }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>∅</div>
      <p style={{ fontSize: 14 }}>No coreference chains detected.</p>
    </div>
  );

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <span style={{ fontSize: 11, color: '#9994B8', fontWeight: 600, letterSpacing: '0.12em' }}>
          COREFERENCE RESOLUTION
        </span>
        <span style={{
          fontSize: 12, fontWeight: 700,
          background: 'rgba(249,115,22,0.15)',
          border: '1px solid rgba(249,115,22,0.3)',
          color: '#FB923C', padding: '3px 12px', borderRadius: 20,
        }}>
          {clusters.length} chain{clusters.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div style={{
        background: 'rgba(0,0,0,0.3)', borderRadius: 12,
        border: '1px solid rgba(255,255,255,0.06)',
        padding: '0 0 16px', overflowX: 'auto',
      }}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', minWidth: 400 }}>

          {/* Arcs */}
          {clusters.map((cluster, ci) => {
            const color = PALETTE[ci % PALETTE.length];
            const isHov = hovered === ci;
            return cluster.mentions.slice(0, -1).map((m1, mi) => {
              const m2 = cluster.mentions[mi + 1];
              const x1 = tokX(m1.start), x2 = tokX(m2.start);
              const cx = (x1 + x2) / 2;
              const ry = Math.abs(x2 - x1) * 0.45 + 25;
              const d = `M ${x1} ${MARGIN.top} Q ${cx} ${MARGIN.top - ry} ${x2} ${MARGIN.top}`;
              return (
                <path key={`${ci}-${mi}`} d={d} fill="none"
                  stroke={color} strokeWidth={isHov ? 3 : 1.5}
                  opacity={hovered === null || isHov ? 0.9 : 0.15}
                  style={{ filter: isHov ? `drop-shadow(0 0 6px ${color})` : 'none', transition: 'all 0.2s', cursor: 'pointer' }}
                  onMouseEnter={() => setHovered(ci)}
                  onMouseLeave={() => setHovered(null)}
                />
              );
            });
          })}

          {/* Tokens */}
          {tokens.slice(0, Math.floor(W / tokW)).map((tok, i) => {
            const x = tokX(i);
            const color = mentionColor[i];
            return (
              <g key={i}>
                {color && <circle cx={x} cy={MARGIN.top} r={7} fill={color}
                  style={{ filter: `drop-shadow(0 0 6px ${color})` }} />}
                <line x1={x} y1={MARGIN.top + (color ? 7 : 0)} x2={x} y2={MARGIN.top + 22}
                  stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
                <text x={x} y={MARGIN.top + 36} textAnchor="middle"
                  fontSize={11} fontFamily="Noto Naskh Arabic, serif"
                  fill={color || '#6B7280'} fontWeight={color ? '700' : '400'}>
                  {tok.length > 5 ? tok.slice(0,4)+'…' : tok}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Cluster pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 16 }}>
        {clusters.map((c, ci) => {
          const color = PALETTE[ci % PALETTE.length];
          return (
            <button key={ci}
              onMouseEnter={() => setHovered(ci)}
              onMouseLeave={() => setHovered(null)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: hovered === ci ? `${color}22` : 'rgba(255,255,255,0.04)',
                border: `1px solid ${hovered === ci ? color : 'rgba(255,255,255,0.1)'}`,
                borderRadius: 8, padding: '6px 14px', cursor: 'pointer',
                transition: 'all 0.2s',
                boxShadow: hovered === ci ? `0 0 12px ${color}40` : 'none',
              }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%',
                background: color, boxShadow: `0 0 6px ${color}` }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: hovered === ci ? color : '#9994B8',
                fontFamily: 'JetBrains Mono, monospace' }}>
                CHAIN {ci + 1}
              </span>
              <span style={{ fontSize: 11, color: color, opacity: 0.7 }}>
                {c.n_mentions} mentions
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
