import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import {Toaster} from './components/ui/sonner.tsx';

createRoot(document.getElementById('root')!).render(
	<StrictMode>
		<App />
		<Toaster richColors position='bottom-center' toastOptions={{className: 'rounded-full w-fit px-[34px] !bg-black-main'}} className='mb-[80px] transition-none animate-none rounded-full' />
	</StrictMode>
);
