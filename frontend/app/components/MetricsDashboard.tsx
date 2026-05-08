"use client";

interface Props {
  latency_ms: number;
  n_entities?: number;
  n_tokens?: number;
  n_clusters?: number;
}

const CARDS = [
  { key: 'latency',  label: 'LATENCY',  icon: 'lightning-charge-fill', color: '#60A5FA', glow: '#60A5FA30' },
  { key: 'entities', label: 'ENTITIES', icon: 'tag-fill',               color: '#34D399', glow: '#34D39930' },
  { key: 'tokens',   label: 'TOKENS',   icon: 'type',                   color: '#C084FC', glow: '#C084FC30' },
  { key: 'chains',   label: 'CHAINS',   icon: 'link-45deg',             color: '#FB923C', glow: '#FB923C30' },
];

export default function MetricsDashboard({ latency_ms, n_entities, n_tokens, n_clusters }: Props) {
  const values = [
    `${latency_ms.toFixed(1)}ms`,
    n_entities ?? '—',
    n_tokens   ?? '—',
    n_clusters ?? '—',
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}
      className="fade-up-2">
      {CARDS.map((card, i) => (
        <div key={card.key} className="glass" style={{
          borderRadius: 16, padding: '18px 16px', textAlign: 'center',
          border: `1px solid ${card.color}22`,
          boxShadow: `0 0 20px ${card.glow}`,
          transition: 'transform 0.2s, box-shadow 0.2s',
          cursor: 'default',
        }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLElement).style.transform = 'translateY(-3px)';
            (e.currentTarget as HTMLElement).style.boxShadow = `0 0 30px ${card.glow}`;
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
            (e.currentTarget as HTMLElement).style.boxShadow = `0 0 20px ${card.glow}`;
          }}
        >
          <i
            className={`bi bi-${card.icon}`}
            style={{
              fontSize: 24, color: card.color,
              display: 'block', marginBottom: 10,
              filter: `drop-shadow(0 0 8px ${card.color})`,
            }}
          />
          <div style={{
            fontSize: 26, fontWeight: 800, color: card.color,
            fontFamily: 'JetBrains Mono, monospace',
            textShadow: `0 0 20px ${card.color}`,
          }}>
            {values[i]}
          </div>
          <div style={{
            fontSize: 10, color: '#9994B8', marginTop: 6,
            fontWeight: 600, letterSpacing: '0.12em',
          }}>
            {card.label}
          </div>
        </div>
      ))}
    </div>
  );
}
