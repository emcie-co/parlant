/**
 * Detect base path from current URL for path-based routing.
 * Extracts the first path segment if it's not a known route.
 * e.g., "/QfnuAKfKSf/chat" -> "/QfnuAKfKSf"
 */
const getBasePath = (): string => {
	const path = window.location.pathname;
	const segments = path.split('/').filter(Boolean);
	
	// If first segment isn't a known route, use it as base path
	if (segments.length > 0 && !['chat', 'docs', 'api', 'healthz'].includes(segments[0])) {
		return '/' + segments[0];
	}
	return '';
};

export const BASE_URL = import.meta.env.VITE_BASE_URL || getBasePath();

const request = async (url: string, options: RequestInit = {}) => {
	try {
		const response = await fetch(url, options);
		if (!response.ok) {
			throw new Error(`HTTP error! Status: ${response.status}`);
		}
		if (options.method === 'PATCH' || options.method === 'DELETE') return;
		return await response.json();
	} catch (error) {
		console.error('Fetch error:', error);
		throw error;
	}
};

export const getData = async (endpoint: string) => {
	return request(`${BASE_URL}/${endpoint}`);
};

export const postData = async (endpoint: string, data?: object) => {
	return request(`${BASE_URL}/${endpoint}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(data),
	});
};

export const patchData = async (endpoint: string, data: object) => {
	return request(`${BASE_URL}/${endpoint}`, {
		method: 'PATCH',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(data),
	});
};

export const deleteData = async (endpoint: string) => {
	return request(`${BASE_URL}/${endpoint}`, {
		method: 'DELETE',
	});
};
