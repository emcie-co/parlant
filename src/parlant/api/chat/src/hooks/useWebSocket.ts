import {useEffect, useRef, useState, useCallback} from 'react';

interface WebSocketOptions {
	onMessage?: (message: string) => void;
	onError?: (error: Event) => void;
	onOpen?: () => void;
	onClose?: (event: CloseEvent) => void;
}

export const useWebSocket = (url: string, defaultRunning?: boolean, options?: WebSocketOptions | null, lastMessageFn?: (message: any) => void) => {
	const [isConnected, setIsConnected] = useState(false);
	const [lastMessage, setLastMessage] = useState<string | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const socketRef = useRef<WebSocket | null>(null);

	const sendMessage = useCallback((message: string) => {
		if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
			socketRef.current.send(message);
		} else {
			console.warn('WebSocket is not open. Unable to send message:', message);
		}
	}, []);

	const reconnect = () => {
		start();
		setTimeout(() => {
			if (!socketRef?.current?.readyState || !{[socketRef.current.OPEN]: true, [socketRef.current?.CONNECTING]: true}[socketRef.current.readyState]) {
				reconnect();
			}
		}, 5000);
	};

	useEffect(() => {
		if (defaultRunning) start();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const start = useCallback(() => {
		if (isRunning || (socketRef.current?.readyState != null && {[socketRef.current.OPEN]: true, [socketRef.current?.CONNECTING]: false}[socketRef.current.readyState])) {
			console.warn('WebSocket is already running.');
			return;
		}

		if ((socketRef.current && socketRef.current.readyState === socketRef.current.OPEN) || (socketRef.current && socketRef.current.readyState === socketRef.current?.CONNECTING)) socketRef.current.close();
		const socket = new WebSocket(url);
		socketRef.current = socket;

		socket.addEventListener('open', () => {
			setIsConnected(true);
			options?.onOpen?.();
		});

		socket.addEventListener('message', (event) => {
			const data = JSON.parse(event.data || '{}');
			setLastMessage(event.data);
			lastMessageFn?.(data);
			options?.onMessage?.(event.data);
		});

		socket.addEventListener('error', (event) => {
			console.error('WebSocket error:', event);
			options?.onError?.(event);
		});

		socket.addEventListener('close', (event) => {
			console.info('WebSocket closed:', event);
			setIsConnected(false);
			options?.onClose?.(event);
			setTimeout(() => {
				if (socketRef?.current?.readyState === 0 || socketRef?.current?.readyState === 1) return;
				reconnect();
			}, 5000);
		});

		setIsRunning(true);
	}, [url, options, isRunning]);

	const pause = useCallback(() => {
		if (socketRef.current) {
			socketRef.current.close();
			socketRef.current = null;
		}
		setIsConnected(false);
		setIsRunning(false);
	}, []);

	useEffect(() => {
		return () => {
			if (socketRef.current) {
				socketRef.current.close();
			}
		};
	}, []);

	return {isConnected, lastMessage, sendMessage, start, pause, isRunning};
};
