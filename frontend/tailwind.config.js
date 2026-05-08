/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        arabic: ['Noto Naskh Arabic', 'serif'],
      },
      colors: {
        ner: {
          per:  '#4A90D9',
          org:  '#27AE60',
          loc:  '#E07B54',
          misc: '#9B59B6',
          date: '#F39C12',
        },
      },
    },
  },
  plugins: [],
}
