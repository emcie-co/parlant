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

function openDB(storeName = STORE_NAME) {
	return new Promise<IDBDatabase>((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, 1);

		request.onupgradeneeded = () => {
			const db = request.result;

			if (!db.objectStoreNames.contains(storeName)) {
				const store = db.createObjectStore(storeName, {autoIncrement: true});

				store.createIndex('timestampIndex', 'timestamp', {unique: false});
			}
		};

		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
}

async function getLogs(correlation_id: string): Promise<Log[]> {
	const db = await openDB();
	return new Promise((resolve, reject) => {
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const request = store.get(correlation_id);
		request.onsuccess = () => resolve(request.result?.values || []);
		request.onerror = () => reject(request.error);
	});
}

export const handleChatLogs = async (log: Log) => {
	if (hasOtherOpenedTabs()) return;
	const db = await openDB();
	const transaction = db.transaction(STORE_NAME, 'readwrite');
	const store = transaction.objectStore(STORE_NAME);

	const logEntry = store.get(log.correlation_id);

	logEntry.onsuccess = () => {
		const data = logEntry.result;
		const timestamp = Date.now();
		if (!data?.values) {
			if (!log.message?.trim().startsWith('HTTP') || log.message?.includes('/events')) store.put({timestamp, values: [log]}, log.correlation_id);
		} else {
			data.values.push(log);
			store.put({timestamp, values: data.values}, log.correlation_id);
		}
	};
	logEntry.onerror = () => console.error(logEntry.error);
};

export const getMessageLogs = async (correlation_id: string): Promise<Log[]> => {
	return getLogs(correlation_id);
};

export const getMessageLogsWithFilters = async (correlation_id: string, filters: {level: string; types?: string[]; content?: string[]}): Promise<Log[]> => {
	const logs = await getMessageLogs(correlation_id);
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
			const match = log.message.match(/^\[([^\]]+)\]/);
			const type = match ? match[1] : 'General';
			return filterTypes.has(type);
		}
		return true;
	});
};

export async function getAgentMessageLogsCount(): Promise<Log[]> {
	const db = await openDB();
	return new Promise((resolve, reject) => {
		try {
			const transaction = db.transaction(STORE_NAME, 'readonly');
			const store = transaction.objectStore(STORE_NAME);
			const index = store.index('timestampIndex');
			const data = index.openCursor();

			const items: any[] = [];

			data.onsuccess = (event) => {
				const cursor = (event.target as IDBRequest).result;
				if (cursor) {
					if (cursor.primaryKey?.includes('::')) items.push(cursor.value);
					cursor.continue();
				} else {
					resolve(items);
				}
			};

			data.onerror = () => reject(data.error);
		} catch (error) {
			db.close();
			reject(error);
		}
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

export async function deleteOldestLogs(deleteTimestamp = 0): Promise<void> {
	if (!deleteTimestamp || deleteTimestamp <= 0) {
		console.log('No valid deletion timestamp provided, skipping cleanup');
		return;
	}

	try {
		const db = await openDB();
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const keysRequest = store.getAllKeys();
		const valuesRequest = store.getAll();

		return new Promise((resolve, reject) => {
			let keys: IDBValidKey[] = [];
			let values: any[] = [];

			keysRequest.onsuccess = () => {
				keys = keysRequest.result;
				if (values.length > 0) deleteOldest();
			};

			valuesRequest.onsuccess = () => {
				values = valuesRequest.result;
				if (keys.length > 0) deleteOldest();
			};

			const deleteOldest = () => {
				const keysToDelete = [];
				for (const i in keys) {
					const data = values[i];
					if (data.timestamp < deleteTimestamp) keysToDelete.push(keys[i]);
				}

				if (keysToDelete.length === 0) {
					db.close();
					resolve();
					return;
				}

				const deleteTransaction = db.transaction(STORE_NAME, 'readwrite');
				const deleteStore = deleteTransaction.objectStore(STORE_NAME);

				let completed = 0;
				let errors = 0;

				keysToDelete.forEach((key) => {
					const deleteRequest = deleteStore.delete(key);

					deleteRequest.onsuccess = () => {
						completed++;
						if (completed + errors === keysToDelete.length) {
							if (errors > 0) {
								console.warn(`Completed with ${errors} errors`);
							}
						}
					};

					deleteRequest.onerror = (event) => {
						errors++;
						console.error(`Failed to delete key ${key}:`, (event.target as IDBRequest).error);
					};
				});

				deleteTransaction.oncomplete = () => {
					db.close();
					console.log(`Successfully deleted ${completed} records older than ${new Date(deleteTimestamp).toISOString()}`);
					resolve();
				};

				deleteTransaction.onerror = (event) => {
					db.close();
					reject((event.target as IDBTransaction).error);
				};
			};

			transaction.onerror = (event) => {
				db.close();
				reject((event.target as IDBTransaction).error);
			};
		});
	} catch (error) {
		console.error('Error in deleteOldestLogs:', error);
		throw error;
	}
}

export async function checkAndCleanupLogs(): Promise<void> {
	try {
		const agentMessages = await getAgentMessageLogsCount();

		if (agentMessages[MAX_RECORDS]) {
			const recordsToDeleteDate = agentMessages[agentMessages.length - MAX_RECORDS]?.timestamp || 0;
			console.log(`Log count exceeds maximum (${MAX_RECORDS}), deleting logs before ${new Date(recordsToDeleteDate)?.toLocaleString()}`);
			await deleteOldestLogs(recordsToDeleteDate);
			console.log('Cleanup completed');
		}
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
