import {useEffect, useState} from 'react';

const LIMIT = 30;

export function useLocalStorage<T>(key: string, initialValue: T) {
	const [storedValue, setStoredValue] = useState<T>(() => {
		try {
			const item = globalThis.localStorage?.getItem(key);
			return item ? JSON.parse(item) : initialValue;
		} catch (error) {
			console.error('Error reading from localStorage', error);
			return initialValue;
		}
	});

	const addVal = () => {
		try {
			if (Array.isArray(storedValue) && storedValue?.length > LIMIT) storedValue.shift();
			localStorage.setItem(key, JSON.stringify(storedValue));
		} catch (e) {
			console.error('Error writing to localStorage', e);
			if (e instanceof DOMException && e.name === 'QuotaExceededError') {
				const logs = JSON.parse(localStorage.logs || '{}');
				if (Object.keys(logs)[0]) {
					console.log('deleting first log');
					delete logs[Object.keys(logs)[0]];
					localStorage.setItem('logs', JSON.stringify(logs));
					addVal();
				}
			}
		}
	};

	useEffect(() => {
		addVal();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [key, storedValue]);

	return [storedValue, setStoredValue];
}
