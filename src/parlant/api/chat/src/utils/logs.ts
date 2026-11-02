/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable no-useless-escape */
import { hasOtherOpenedTabs } from '@/lib/broadcast-channel';
import {Log} from './interfaces';

const logLevels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'];
export const DB_NAME = 'Parlant';
const STORE_NAME = 'logs';
const MAX_RECORDS = 2000;
const CHECK_INTERVAL = 10 * 60 * 1000;

export function getIndexedDBSize(databaseName = DB_NAME, tableName = STORE_NAME): Promise<number> {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(databaseName);

		request.onerror = (event) => {
			const target = event?.target as IDBOpenDBRequest;
			const error = target?.error;
			reject(new Error(`Failed to open database: ${error}`));
		};

		request.onsuccess = (event) => {
			const target = event?.target as IDBOpenDBRequest;
			const db = target?.result;

			if (!db.objectStoreNames.contains(tableName)) {
				db.close();
				reject(new Error(`Table "${tableName}" does not exist in database "${databaseName}"`));
				return;
			}

			const transaction = db.transaction(tableName, 'readonly');
			const store = transaction.objectStore(tableName);

			const getAllRequest = store.getAll();

			getAllRequest.onerror = (event: Event) => {
				db.close();
				const target = event.target as IDBRequest;
				reject(new Error(`Failed to read data: ${target.error}`));
			};

			getAllRequest.onsuccess = (event: Event) => {
				const target = event.target as IDBRequest;
				const records = target.result;
				let totalSize = 0;

				records.forEach((record: Record<string, unknown>) => {
					const serialized = JSON.stringify(record);
					totalSize += serialized.length * 2;
				});

				const sizeInMB = totalSize / (1024 * 1024);

				db.close();
				resolve(sizeInMB);
			};
		};
	});
}

export function clearIndexedDBData(dbName = DB_NAME, objectStoreName = STORE_NAME) {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(dbName);

		request.onerror = (event) => {
			const target = event?.target as IDBOpenDBRequest;
			const error = target?.error;
			reject(error);
		};

		request.onsuccess = (event) => {
			const target = event?.target as IDBOpenDBRequest;
			const db = target?.result;
			const transaction = db.transaction(objectStoreName, 'readwrite');
			const objectStore = transaction.objectStore(objectStoreName);
			const clearRequest = objectStore.clear();

			clearRequest.onsuccess = () => {
				resolve(null);
			};

			clearRequest.onerror = (clearEvent: Event) => {
				const target = clearEvent.target as IDBRequest;
				reject(target.error);
			};

			transaction.oncomplete = () => {
				db.close();
			};
		};
	});
}

// Reuse a single connection. openDB is called on every log read AND every log
// write (handleChatLogs); during a live turn that's hundreds of opens, each
// spinning up — and leaking — a fresh IndexedDB connection. Memoize the open
// promise and drop it if the connection ever closes so it reopens cleanly.
let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(storeName = STORE_NAME) {
	if (dbPromise) return dbPromise;

	dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, 1);

		request.onupgradeneeded = () => {
			const db = request.result;

			if (!db.objectStoreNames.contains(storeName)) {
				const store = db.createObjectStore(storeName, {autoIncrement: true});

				store.createIndex('timestampIndex', 'timestamp', {unique: false});
			}
		};

		request.onsuccess = () => {
			const db = request.result;
			db.onclose = () => {
				dbPromise = null;
			};
			// Another tab requesting a version change would otherwise block it forever.
			db.onversionchange = () => {
				db.close();
				dbPromise = null;
			};
			resolve(db);
		};
		request.onerror = () => {
			dbPromise = null;
			reject(request.error);
		};
	});

	return dbPromise;
}

async function getLogs(trace_id: string): Promise<Log[]> {
	const db = await openDB();
	return new Promise((resolve, reject) => {
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const request = store.get(trace_id);
		request.onsuccess = () => resolve(request.result?.values || []);
		request.onerror = () => reject(request.error);
	});
}

// Incoming logs are buffered and flushed to IndexedDB in batches rather than
// written one-transaction-per-log. A live, in-progress turn emits logs far faster
// than per-log read-modify-write transactions can keep up; the WebSocket receive
// queue (and the inspector that reads from IndexedDB) then falls behind and only
// catches up when the turn ends — which is the "blank logs for ~30s" symptom.
// Buffering keeps the WS handler O(1) and collapses a burst into a handful of
// grouped transactions, and it relieves the TCP backpressure that otherwise
// stalls the server's log drain.
const pendingLogs: Log[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_INTERVAL = 100;

const flushPendingLogs = async () => {
	flushTimer = null;
	if (!pendingLogs.length) return;

	const batch = pendingLogs.splice(0, pendingLogs.length);

	// Group by trace_id so each bucket is read-modify-written once for the batch.
	const byTrace = new Map<string, Log[]>();
	for (const log of batch) {
		const existing = byTrace.get(log.trace_id);
		if (existing) existing.push(log);
		else byTrace.set(log.trace_id, [log]);
	}

	const db = await openDB();
	const transaction = db.transaction(STORE_NAME, 'readwrite');
	const store = transaction.objectStore(STORE_NAME);
	const timestamp = Date.now();

	for (const [traceId, traceLogs] of byTrace) {
		const logEntry = store.get(traceId);
		logEntry.onsuccess = () => {
			const data = logEntry.result;
			if (!data?.values) {
				// Same gate as before, applied to the batch: drop leading HTTP noise
				// (except /events) when first creating a bucket.
				const seed = traceLogs.filter((l) => !l.message?.trim().startsWith('HTTP') || l.message?.includes('/events'));
				if (seed.length) store.put({timestamp, values: seed}, traceId);
			} else {
				data.values.push(...traceLogs);
				store.put({timestamp, values: data.values}, traceId);
			}
		};
		logEntry.onerror = () => console.error(logEntry.error);
	}

	// Overlapping-scope readwrite transactions are serialized by IndexedDB, so a
	// later flush always sees an earlier flush's writes — no lost updates.
	transaction.oncomplete = () => {
		for (const traceId of byTrace.keys()) {
			window.dispatchEvent(new CustomEvent('new-log', {detail: {trace_id: traceId}}));
		}
	};
	transaction.onerror = () => console.error(transaction.error);
};

export const handleChatLogs = async (log: Log) => {
	if (hasOtherOpenedTabs()) return;
	pendingLogs.push(log);
	if (!flushTimer) flushTimer = setTimeout(flushPendingLogs, FLUSH_INTERVAL);
};

export const getMessageLogs = async (trace_id: string): Promise<Log[]> => {
	return getLogs(trace_id);
};

// Pure, in-memory filter over an already-loaded log array. Kept separate from any
// IndexedDB read so callers that already hold the logs (e.g. the live inspector)
// can re-filter without paying for another DB round-trip per change.
export const filterLogs = (logs: Log[], filters: {level: string; types?: string[]; content?: string[]}): Log[] => {
	const escapedWords = filters?.content?.map((word) => word.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, '\\$1'));
	const pattern = escapedWords?.map((word) => `\\[?${word}\\]?`).join('.*?');
	const levelIndex = filters.level ? logLevels.indexOf(filters.level) : null;
	const validLevels = filters.level ? new Set(logLevels.filter((_, i) => i <= (levelIndex as number))) : null;
	const filterTypes = filters.types?.length ? new Set(filters.types) : null;

	return logs.filter((log) => {
		if (validLevels && !validLevels.has(log.level)) return false;
		if (pattern) {
			const allWordsMatch = escapedWords?.every((word) => {
				const regex = new RegExp(`\\[?${word}\\]?`, 'i'); // Allow optional brackets
				return regex.test(`[${log.level}]${log.message}`);
			  });
			if (!allWordsMatch) return false;
		}
		if (filterTypes) {
			const matches = [...log.message.matchAll(/\[([^\]]+)\]/g)].map(m => m?.[1]);
			const match = matches[0]?.startsWith('T+') ? matches[1] : matches[0];
			const type = match || 'General';
			return filterTypes.has(type);
		}
		return true;
	});
};

export const getMessageLogsWithFilters = async (trace_id: string, filters: {level: string; types?: string[]; content?: string[]}): Promise<Log[]> => {
	return filterLogs(await getMessageLogs(trace_id), filters);
};

// Count records cheaply, WITHOUT walking a cursor over the whole store on the
// main thread. The previous getAgentMessageLogsCount cursored every record (and
// collected their full values) on app load and every CHECK_INTERVAL. Combined
// with pruning that never matched — it keyed on a legacy "::" trace-id format that
// today's plain-uuid4 trace ids don't have, so nothing was ever deleted and the
// store grew without bound — that walk flooded the main thread for tens of seconds
// and blocked the log inspector from rendering, even for old messages with only a
// handful of logs. A keyed `count()` does the work in the IDB engine instead.
async function countLogRecords(): Promise<number> {
	const db = await openDB();
	return new Promise<number>((resolve, reject) => {
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const request = transaction.objectStore(STORE_NAME).count();
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
}

export async function getAllLogKeys(): Promise<IDBValidKey[]> {
	const db = await openDB();
	return new Promise((resolve, reject) => {
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const keysRequest = store.getAllKeys();

		keysRequest.onsuccess = () => {
			db.close();
			resolve(keysRequest.result);
		};

		keysRequest.onerror = () => {
			db.close();
			reject(keysRequest.error);
		};
	});
}

// Delete the oldest records via the timestamp index. Work is bounded by the
// number being removed (not the whole store), and capped per run so a large
// backlog drains over several runs instead of one long, read-blocking
// transaction. Does NOT close the shared connection.
const MAX_DELETES_PER_RUN = 2000;

async function deleteOldestRecords(numToDelete: number): Promise<void> {
	if (numToDelete <= 0) return;

	const db = await openDB();
	const limit = Math.min(numToDelete, MAX_DELETES_PER_RUN);

	return new Promise<void>((resolve, reject) => {
		const transaction = db.transaction(STORE_NAME, 'readwrite');
		const index = transaction.objectStore(STORE_NAME).index('timestampIndex');
		const cursorRequest = index.openCursor(); // ascending by timestamp -> oldest first
		let deleted = 0;

		cursorRequest.onsuccess = () => {
			const cursor = cursorRequest.result;
			if (cursor && deleted < limit) {
				cursor.delete();
				deleted++;
				cursor.continue();
			}
		};
		cursorRequest.onerror = () => reject(cursorRequest.error);
		transaction.oncomplete = () => resolve();
		transaction.onerror = () => reject(transaction.error);
	});
}

export async function checkAndCleanupLogs(): Promise<void> {
	try {
		const count = await countLogRecords();
		if (count <= MAX_RECORDS) return;
		await deleteOldestRecords(count - MAX_RECORDS);
	} catch (error) {
		console.error('Error during log cleanup:', error);
	}
}

let cleanupInterval: number | null = null;

export function startLogCleanup(): void {
	checkAndCleanupLogs();

	if (!cleanupInterval) {
		cleanupInterval = window.setInterval(checkAndCleanupLogs, CHECK_INTERVAL);
		console.log(`Log cleanup scheduled every ${CHECK_INTERVAL / 1000 / 60} minutes`);
	}
}

export function stopLogCleanup(): void {
	if (cleanupInterval) {
		window.clearInterval(cleanupInterval);
		cleanupInterval = null;
		console.log('Log cleanup stopped');
	}
}

startLogCleanup();
