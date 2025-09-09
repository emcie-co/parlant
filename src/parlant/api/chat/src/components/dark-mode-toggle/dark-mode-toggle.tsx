import { ReactNode, useEffect, useState } from 'react';

/**
 * A toggle component that allows users to switch between light and dark themes.
 * Persists the theme preference in localStorage and applies the appropriate CSS classes.
 * @returns {ReactNode} The rendered dark mode toggle button component
 */
const DarkModeToggle = (): ReactNode =>{
    /**
     * Retrieves the initial theme from localStorage or defaults to 'light'.
     * @returns {string} The stored theme value or 'light' as default
     */
    const getInitialTheme = () => localStorage.getItem('theme') || 'light';
    const [theme, setTheme] = useState(getInitialTheme);
  
    useEffect(() => {
      const root = window.document.documentElement;
  
      if (theme === 'dark') {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
  
      localStorage.setItem('theme', theme);
    }, [theme]);
  
    /**
     * Toggles the current theme between 'light' and 'dark'.
     */
    const toggleTheme = () => {
      setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
    };

    return (
        <div className="flex items-center justify-center bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
            <button onClick={toggleTheme} 
                className="px-4 py-2 bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded">
                Switch to {theme === 'light' ? 'Dark' : 'Light'} Mode
            </button>
        </div>
    );
};

export default DarkModeToggle;