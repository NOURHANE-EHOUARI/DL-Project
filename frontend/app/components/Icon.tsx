"use client";

// Bootstrap Icons component — uses the bootstrap-icons CSS classes
// Full icon list: https://icons.getbootstrap.com/

interface IconProps {
  name: string;           // e.g. "tag-fill", "lightning-charge-fill"
  size?: number;
  color?: string;
  style?: React.CSSProperties;
  className?: string;
}

export default function Icon({ name, size = 16, color, style, className }: IconProps) {
  return (
    <i
      className={`bi bi-${name} ${className || ''}`}
      style={{ fontSize: size, color, lineHeight: 1, ...style }}
    />
  );
}
