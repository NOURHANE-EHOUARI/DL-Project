"use client";
import { useState, useRef, useEffect } from "react";

interface Mention { text: string; start: number; end: number; }
interface Cluster { cluster_id: number; mentions: Mention[]; n_mentions: number; }
interface Props { tokens: string[]; clusters: Cluster[]; }

const PALETTE = ['#A855F7','#F97316','#60A5FA','#34D399','#FBBF24','#F87171','#38BDF8','#E879F9'];

export default function CorefArcs({ tokens, clusters }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const tokenRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [arcs, setArcs] = useState<{ d: string; color: string; ci: number }[]>([]);
  const [svgHeight, setSvgHeight] = useState(80);

  // Map token index → cluster color
  const mentionColor: Record<number, string> = {};
  clusters.forEach((c, ci) => {
    c.mentions.forEach(m => {
      for (let k = m.start; k <= m.end; k++) mentionColor[k] = PALETTE[ci % PALETTE.length];
    });
  });

  // Recompute arcs whenever tokens render or window resizes
  useEffect(() => {
    function compute() {
      if (!containerRef.current) return;
      const containerRect = containerRef.current.getBoundingClientRect();
      const newArcs: { d: string; color: string; ci: number }[] = [];
      let maxArcHeight = 40;

      clusters.forEach((cluster, ci) => {
        const color = PALETTE[ci % PALETTE.length];
        cluster.mentions.slice(0, -1).forEach((m1, mi) => {
          const m2 = cluster.mentions[mi + 1];
          const el1 = tokenRefs.current[m1.start];
          const el2 = tokenRefs.current[m2.start];
          if (!el1 || !el2) return;

          const r1 = el1.getBoundingClientRect();
          const r2 = el2.getBoundingClientRect();

          // Centre of each token dot relative to the SVG overlay
          const x1 = r1.left - containerRect.left + r1.width / 2;
          const x2 = r2.left - containerRect.left + r2.width / 2;
          const y  = r1.top  - containerRect.top  + r1.height / 2;

          // Arc bows upward (negative y = up in SVG)
          const ry = Math.min(60, Math.abs(x2 - x1) * 0.45 + 18);
          maxArcHeight = Math.max(maxArcHeight, ry);
          const mx = (x1 + x2) / 2;
          const d = `M ${x1} ${y} Q ${mx} ${y - ry} ${x2} ${y}`;
          newArcs.push({ d, color, ci });
        });
      });

      setArcs(newArcs);
      setSvgHeight(maxArcHeight + 10);
    }

    // Wait one frame for refs to be positioned, then compute
    const id = requestAnimationFrame(compute);
    window.addEventListener('resize', compute);
    return () => { cancelAnimationFrame(id); window.removeEventListener('resize', compute); };
  }, [tokens, clusters]);

  if (!clusters.length) return (
    <div style={{ textAlign: 'center', padding: '40px 0', color: '#9994B8' }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>∅</div>
      <p style={{ fontSize: 14 }}>No coreference chains detected.</p>
    </div>
  );

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
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

      {/* Token area with SVG arc overlay */}
      <div
        style={{
          background: 'rgba(0,0,0,0.3)', borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.06)',
          padding: '16px 20px',
          position: 'relative',
        }}
      >
        {/* SVG overlay for arcs — sits above the tokens */}
        <div
          ref={containerRef}
          style={{ position: 'relative' }}
        >
          {/* Arc SVG — absolutely positioned, pointer-events none so tokens stay clickable */}
          <svg
            style={{
              position: 'absolute',
              top: `-${svgHeight}px`,
              left: 0, width: '100%',
              height: svgHeight,
              overflow: 'visible',
              pointerEvents: 'none',
            }}
          >
            {arcs.map((arc, ai) => (
              <path
                key={ai}
                d={arc.d}
                fill="none"
                stroke={arc.color}
                strokeWidth={hovered === arc.ci ? 2.5 : 1.5}
                opacity={hovered === null || hovered === arc.ci ? 0.9 : 0.15}
                style={{
                  filter: hovered === arc.ci ? `drop-shadow(0 0 5px ${arc.color})` : 'none',
                  transition: 'all 0.2s',
                }}
              />
            ))}
          </svg>

          {/* Tokens — flex-wrap so all tokens show */}
          <div
            dir="rtl"
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '10px 14px',
              paddingTop: `${svgHeight}px`,   /* push tokens down to make room for arcs */
              fontFamily: 'Noto Naskh Arabic, serif',
              fontSize: 16,
            }}
          >
            {tokens.map((tok, i) => {
              const color = mentionColor[i];
              return (
                <span
                  key={i}
                  ref={el => { tokenRefs.current[i] = el; }}
                  onMouseEnter={() => {
                    // Find which cluster this token belongs to
                    const ci = clusters.findIndex(c =>
                      c.mentions.some(m => i >= m.start && i <= m.end)
                    );
                    setHovered(ci >= 0 ? ci : null);
                  }}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    color: color || '#6B7280',
                    fontWeight: color ? 700 : 400,
                    cursor: color ? 'default' : 'default',
                    padding: '2px 4px',
                    borderRadius: 4,
                    background: color
                      ? (hovered === clusters.findIndex(c =>
                          c.mentions.some(m => i >= m.start && i <= m.end))
                        ? `${color}22` : `${color}11`)
                      : 'transparent',
                    transition: 'background 0.15s',
                    position: 'relative',
                  }}
                >
                  {/* Dot above highlighted tokens */}
                  {color && (
                    <span style={{
                      position: 'absolute',
                      top: -10, left: '50%',
                      transform: 'translateX(-50%)',
                      width: 6, height: 6,
                      borderRadius: '50%',
                      background: color,
                      boxShadow: `0 0 6px ${color}`,
                      display: 'block',
                    }} />
                  )}
                  {tok}
                </span>
              );
            })}
          </div>
        </div>
      </div>

      {/* Cluster pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 16 }}>
        {clusters.map((c, ci) => {
          const color = PALETTE[ci % PALETTE.length];
          return (
            <button
              key={ci}
              onMouseEnter={() => setHovered(ci)}
              onMouseLeave={() => setHovered(null)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: hovered === ci ? `${color}22` : 'rgba(255,255,255,0.04)',
                border: `1px solid ${hovered === ci ? color : 'rgba(255,255,255,0.1)'}`,
                borderRadius: 8, padding: '6px 14px', cursor: 'pointer',
                transition: 'all 0.2s',
                boxShadow: hovered === ci ? `0 0 12px ${color}40` : 'none',
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%',
                background: color, boxShadow: `0 0 6px ${color}`, flexShrink: 0 }} />
              <span style={{
                fontSize: 12, fontWeight: 600,
                color: hovered === ci ? color : '#9994B8',
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                CHAIN {ci + 1}
              </span>
              <span style={{ fontSize: 11, color: color, opacity: 0.7 }}>
                mentions {c.n_mentions}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
