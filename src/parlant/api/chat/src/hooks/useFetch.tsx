import {BASE_URL} from '@/utils/api';
import {useState, useEffect, useCallback, useRef, ReactElement} from 'react';
import {toast} from 'sonner';

interface useFetchResponse<T> {
	data: T | null;
	loading: boolean;
	error: null | {message: string};
	refetch: () => void;
	ErrorTemplate: (() => ReactElement) | null;
	abortFetch: () => void;
}

function objToUrlParams(obj: Record<string, unknown>) {
	const params = [];
	for (const key in obj) {
		if (Object.prototype.hasOwnProperty.call(obj, key)) {
			const value = encodeURIComponent(`${obj[key]}`);
			params.push(`${key}=${value}`);
		}
	}
	return `?${params.join('&')}`;
}

const ABORT_REQ_CODE = 20;
const NOT_FOUND_CODE = 404;
const TIMEOUT_ERROR_MESSAGE = 'Error: Gateway Timeout';

export default function useFetch<T>(url: string, body?: Record<string, unknown>, dependencies: unknown[] = [], retry = false, initiate = true, checkErr = true): useFetchResponse<T> {
	const [data, setData] = useState<T | null>(null);
	const [loading, setLoading] = useState<boolean>(false);
	const [error, setError] = useState<null | {message: string}>(null);
	const [refetchData, setRefetchData] = useState(false);
	const params = body ? objToUrlParams(body) : '';

	const controllerRef = useRef<AbortController | null>(null);

	useEffect(() => {
		if (error && error.message !== TIMEOUT_ERROR_MESSAGE) throw new Error(`Failed to fetch "${url}"`);
	}, [error, url]);

	const ErrorTemplate = () => {
		return (
			<div>
				<div>Something went wrong</div>
				<div role='button' onClick={() => setRefetchData((r) => !r)} className='underline cursor-pointer'>
					Click to retry
				</div>
			</div>
		);
	};

	const refetch = () => setRefetchData((r) => !r);

	useEffect(() => {
		if (retry && error?.message === TIMEOUT_ERROR_MESSAGE) {
			setRefetchData((r) => !r);
			error.message = '';
		}
	}, [retry, error]);

	const fetchData = useCallback(
		(customParams = '') => {
			const controller = new AbortController();
			controllerRef.current = controller;
			const {signal} = controller;
			setTimeout(() => setLoading(true), 0);
			setError(null);
			const reqParams = customParams || params;

			fetch(`${BASE_URL}/${url}${reqParams}`, {signal})
				.then(async (response) => {
					if (!response.ok) {
						if (response.status === NOT_FOUND_CODE) {
							throw {code: NOT_FOUND_CODE, message: response.statusText};
						}
						throw new Error(`Error: ${response.statusText}`);
					}
					const result = await response.json();
					setData(result);
				})
				.catch((err) => {
					if (checkErr && err.code !== ABORT_REQ_CODE) setError({message: err.message});
					else if (err.code !== ABORT_REQ_CODE && err.code !== NOT_FOUND_CODE && retry) fetchData();

					if (err.code === NOT_FOUND_CODE) toast.error('resource not found. please try to refresh the page');
				})
				.finally(() => checkErr && setLoading(false));
		},
		// eslint-disable-next-line react-hooks/exhaustive-deps
		[url, refetchData, ...dependencies]
	);

	useEffect(() => {
		if (!initiate) return;
		fetchData();

		return () => {
			controllerRef.current?.abort();
		};
	}, [fetchData, initiate]);

	const abortFetch = () => {
		controllerRef.current?.abort();
	};

	return {data, loading, error, refetch, ErrorTemplate: error && ErrorTemplate, abortFetch};
}
