/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary backgrounds
        'primary-bg': '#0A0B0E',
        'secondary-bg': '#13141A',
        
        // Accent colors
        'accent': '#6366F1',
        'accent-hover': '#5855EB',
        'accent-light': '#818CF8',
        
        // Status colors
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        
        // Text colors
        'text-primary': '#FFFFFF',
        'text-secondary': '#A1A1AA',
        'text-tertiary': '#71717A',
        
        // Border colors
        'border-primary': '#27272A',
        'border-secondary': '#3F3F46',
        
        // Surface colors
        'surface': '#18181B',
        'surface-hover': '#27272A',
        'surface-active': '#3F3F46',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      spacing: {
        '18': '4.5rem',   // 72px
        '88': '22rem',    // 352px
        '128': '32rem',   // 512px
      },
      borderRadius: {
        'card': '8px',
        'modal': '12px',
        'button': '4px',
      },
      backdropBlur: {
        'header': 'blur(12px)',
      },
      boxShadow: {
        'soft': '0 2px 4px rgba(0, 0, 0, 0.1)',
        'medium': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'strong': '0 8px 16px rgba(0, 0, 0, 0.15)',
        'glow': '0 0 20px rgba(99, 102, 241, 0.3)',
        'avatar': '0 8px 32px rgba(0, 0, 0, 0.2)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-in': 'slideIn 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(99, 102, 241, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.6)' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      transitionDuration: {
        '400': '400ms',
      },
      glassmorphism: {
        'light': 'backdrop-blur-md bg-white/10 border border-white/20',
        'dark': 'backdrop-blur-md bg-black/10 border border-white/10',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.text-primary'),
            h1: { color: theme('colors.text-primary') },
            h2: { color: theme('colors.text-primary') },
            h3: { color: theme('colors.text-primary') },
            h4: { color: theme('colors.text-primary') },
            h5: { color: theme('colors.text-primary') },
            h6: { color: theme('colors.text-primary') },
            strong: { color: theme('colors.text-primary') },
            a: { 
              color: theme('colors.accent'),
              '&:hover': { color: theme('colors.accent-light') }
            },
            code: { 
              color: theme('colors.accent-light'),
              backgroundColor: theme('colors.surface'),
              padding: '0.125rem 0.25rem',
              borderRadius: '0.25rem',
              fontWeight: '400'
            },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
            pre: {
              backgroundColor: theme('colors.surface'),
              color: theme('colors.text-primary'),
              padding: '1rem',
              borderRadius: theme('borderRadius.card'),
              border: '1px solid ' + theme('colors.border-primary')
            },
            blockquote: {
              borderLeftColor: theme('colors.accent'),
              color: theme('colors.text-secondary'),
              fontStyle: 'italic'
            },
            'ul > li': {
              '&::before': { 
                backgroundColor: theme('colors.accent')
              }
            },
            'ol > li': {
              '&::before': { 
                color: theme('colors.text-secondary')
              }
            },
            hr: {
              borderColor: theme('colors.border-primary')
            }
          }
        },
        invert: {
          css: {
            color: theme('colors.text-primary'),
          }
        }
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    function({ addUtilities }) {
      const newUtilities = {
        '.glass-light': {
          'backdrop-filter': 'blur(12px)',
          'background': 'rgba(255, 255, 255, 0.1)',
          'border': '1px solid rgba(255, 255, 255, 0.2)',
        },
        '.glass-dark': {
          'backdrop-filter': 'blur(12px)',
          'background': 'rgba(0, 0, 0, 0.1)',
          'border': '1px solid rgba(255, 255, 255, 0.1)',
        },
        '.scrollbar-thin': {
          'scrollbar-width': 'thin',
          'scrollbar-color': '#3F3F46 transparent',
        },
        '.scrollbar-thin::-webkit-scrollbar': {
          'width': '6px',
        },
        '.scrollbar-thin::-webkit-scrollbar-track': {
          'background': 'transparent',
        },
        '.scrollbar-thin::-webkit-scrollbar-thumb': {
          'background-color': '#3F3F46',
          'border-radius': '3px',
        },
        '.scrollbar-thin::-webkit-scrollbar-thumb:hover': {
          'background-color': '#52525B',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}