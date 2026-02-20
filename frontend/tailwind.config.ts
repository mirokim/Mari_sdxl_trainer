import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        notion: {
          bg: '#ffffff',
          'bg-dark': '#191919',
          sidebar: '#f7f6f3',
          'sidebar-dark': '#202020',
          hover: '#efefef',
          'hover-dark': '#2f2f2f',
          border: '#e3e3e3',
          'border-dark': '#333333',
          text: '#37352f',
          'text-dark': '#e3e3e3',
          'text-secondary': '#787774',
          'text-secondary-dark': '#9b9a97',
          accent: '#2383e2',
          'accent-hover': '#0b6bcb',
          success: '#0f7b0f',
          warning: '#cf9f02',
          error: '#eb5757',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Pretendard', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      fontSize: {
        'notion-title': ['2rem', { lineHeight: '1.2', fontWeight: '700' }],
        'notion-h1': ['1.5rem', { lineHeight: '1.3', fontWeight: '600' }],
        'notion-h2': ['1.25rem', { lineHeight: '1.3', fontWeight: '600' }],
        'notion-body': ['0.9375rem', { lineHeight: '1.6' }],
        'notion-small': ['0.8125rem', { lineHeight: '1.5' }],
      },
      spacing: {
        'sidebar': '240px',
        'sidebar-collapsed': '48px',
      },
      animation: {
        'fade-in': 'fadeIn 0.15s ease-in-out',
        'slide-in': 'slideIn 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-8px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
