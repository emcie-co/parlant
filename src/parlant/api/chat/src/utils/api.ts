export const BASE_URL = 'http://127.0.0.1:8800';
// export const BASE_URL = 'http://localhost:8800';

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
