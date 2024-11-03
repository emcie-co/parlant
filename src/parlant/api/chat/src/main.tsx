import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { Toaster } from './components/ui/sonner.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    <Toaster richColors position='bottom-right'/>
  </StrictMode>,
);
