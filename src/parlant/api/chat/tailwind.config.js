/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: ['class'],
	content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
	theme: {
		extend: {
			screens: {
				mobile: '751px',
				tablet: '1080px',
			},
			keyframes: {
				'fade-in': {
					'0%': {opacity: 0},
					'100%': {opacity: 1},
				},
				'scroll-down': {
					'0%': {height: 0},
					'100%': {height: '100%'},
				},
				'background-shift': {
					'0%, 100%': {'background-position-x': '20%'},
					'50%': {'background-position-x': '80%'},
				},
			},
			animation: {
				'fade-in': 'fade-in 300ms linear',
				'scroll-down': 'scroll-down 300ms linear',
				'background-shift': 'background-shift 5s linear infinite',
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)',
			},
			colors: {
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				main: 'var(--main)',
				'blue-main': '#1E00FF',
				'black-main': '#151515',
				'gray-0': '#656565',
				'gray-1': '#A9A9A9',
				'gray-2': '#CDCDCD',
				'gray-3': '#EBECF0',
				'gray-4': '#F5F6F8',
				'gray-5': '#FBFBFB',
				muted: '#EBECF0',
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))',
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))',
				},
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))',
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))',
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))',
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))',
				},
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				chart: {
					1: 'hsl(var(--chart-1))',
					2: 'hsl(var(--chart-2))',
					3: 'hsl(var(--chart-3))',
					4: 'hsl(var(--chart-4))',
					5: 'hsl(var(--chart-5))',
				},
			},
			fontFamily: {
				'ubuntu-sans': 'Ubuntu Sans',
				'ubuntu-mono': 'Ubuntu Mono',
				inter: 'inter',
			},
		},
	},
	plugins: [require('tailwindcss-animate')],
};
