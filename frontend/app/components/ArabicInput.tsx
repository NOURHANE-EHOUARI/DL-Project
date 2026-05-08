"use client";

interface ArabicInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
}

const EXAMPLES = [
  "زار الرئيس محمد بن سلمان مدينة الرياض لافتتاح مشاريع تنموية جديدة.",
  "أعلنت شركة أرامكو السعودية عن نتائج مالية قياسية للربع الثالث.",
  "محمد ذهب إلى المدرسة وهو يحب القراءة كثيراً في المساء.",
  "أكد وزير الخارجية أن المملكة ملتزمة بدعم الاستقرار الإقليمي.",
];

export default function ArabicInput({ value, onChange, onSubmit, loading }: ArabicInputProps) {
  return (
    <div className="glass rounded-2xl p-6 fade-up-1" style={{ border: '1px solid rgba(124,58,237,0.3)' }}>
      {/* Label */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: 'linear-gradient(135deg,#A855F7,#F97316)',
          }} />
          <span style={{ fontSize: 12, color: '#9994B8', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Arabic Text Input
          </span>
        </div>
        <span style={{ fontSize: 11, color: '#9994B866', fontFamily: 'JetBrains Mono, monospace' }}>
          {value.length} / 2000
        </span>
      </div>

      {/* Textarea */}
      <div style={{ position: 'relative' }}>
        <textarea
          dir="rtl"
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder="اكتب أو الصق النص العربي هنا..."
          style={{
            width: '100%',
            height: 120,
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 12,
            padding: '16px 20px',
            color: '#F1F0FF',
            fontFamily: 'Noto Naskh Arabic, serif',
            fontSize: 18,
            lineHeight: 1.8,
            resize: 'none',
            outline: 'none',
            transition: 'border-color 0.2s',
          }}
          onFocus={e => e.target.style.borderColor = 'rgba(124,58,237,0.5)'}
          onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
        />
      </div>

      {/* Examples */}
      <div className="mt-4">
        <p style={{ fontSize: 11, color: '#9994B8', marginBottom: 8, letterSpacing: '0.05em' }}>
          EXAMPLES
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => onChange(ex)}
              style={{
                fontSize: 12,
                background: 'rgba(124,58,237,0.1)',
                border: '1px solid rgba(124,58,237,0.25)',
                borderRadius: 8,
                padding: '5px 12px',
                color: '#C084FC',
                cursor: 'pointer',
                fontFamily: 'Noto Naskh Arabic, serif',
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => {
                (e.target as HTMLElement).style.background = 'rgba(124,58,237,0.2)';
                (e.target as HTMLElement).style.borderColor = 'rgba(124,58,237,0.5)';
              }}
              onMouseLeave={e => {
                (e.target as HTMLElement).style.background = 'rgba(124,58,237,0.1)';
                (e.target as HTMLElement).style.borderColor = 'rgba(124,58,237,0.25)';
              }}
            >
              {ex.slice(0, 28)}...
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: 20 }}>
        <button
          onClick={onSubmit}
          disabled={loading || !value.trim()}
          className="border-glow"
          style={{
            padding: '12px 32px',
            borderRadius: 12,
            background: loading || !value.trim()
              ? 'rgba(124,58,237,0.2)'
              : 'linear-gradient(135deg,#7C3AED,#F97316)',
            color: '#fff',
            fontWeight: 700,
            fontSize: 14,
            cursor: loading || !value.trim() ? 'not-allowed' : 'pointer',
            opacity: loading || !value.trim() ? 0.5 : 1,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            border: 'none',
            letterSpacing: '0.05em',
            transition: 'opacity 0.2s, transform 0.15s',
            position: 'relative',
          }}
          onMouseEnter={e => { if (!loading && value.trim()) (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; }}
        >
          {loading ? (
            <>
              <svg className="spin" width={16} height={16} viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="3" strokeDasharray="30 70" />
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg width={16} height={16} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Analyze Text
            </>
          )}
        </button>
      </div>
    </div>
  );
}
