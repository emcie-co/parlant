import {Log} from './interfaces';
import {groupBy} from './obj';

const logLevels = ['WARNING', 'INFO', 'DEBUG'];

// const getLocalStorageSize = (): number => {
// 	let totalBytes = 0;

// 	for (let i = 0; i < localStorage.length; i++) {
// 		const key = localStorage.key(i) || '';
// 		const value = localStorage.getItem(key) || '';
// 		totalBytes += key.length + value.length;
// 	}

// 	return totalBytes / (1024 * 1024);
// };

const setLocalStorageItem = (logsJson: Record<string, string>, recursive?: boolean): boolean | void => {
	try {
		if (!localStorage) return;
		localStorage.setItem('logs', JSON.stringify(logsJson));
		if (recursive) return true;
	} catch (e) {
		if (e instanceof DOMException && e.name === 'QuotaExceededError') {
			const firstKey = Object.keys(logsJson)?.[0];
			if (firstKey) delete logsJson[firstKey];
			return setLocalStorageItem(logsJson, true);
		}
	}
};

export const handleChatLogs = (log: Log) => {
	if (!localStorage) return;
	if (!log.correlation_id.includes('::')) return;
	let eventsChanged = false;
	const logsJson = JSON.parse(localStorage.logs || '{}');
	if (!logsJson[log.correlation_id]) {
		eventsChanged = true;
		logsJson[log.correlation_id] = [];
	}

	logsJson[log.correlation_id].push(log);
	const setStorage = setLocalStorageItem(logsJson);
	eventsChanged ||= !!setStorage;
	return eventsChanged ? new Set(Object.keys(logsJson)) : null;
};

export const getMessageLogs = (correlation_id: string): Log[] => {
	if (!localStorage) return [];
	const logsJson = JSON.parse(localStorage.logs || '{}');
	return logsJson[correlation_id] || [];
};

export const getMessageLogsWithFilters = (correlation_id: string, filters: {level: string; types?: string[]; content?: string[]}): Log[] => {
	const logs = getMessageLogs(correlation_id);
	const escapedWords = filters?.content?.map((word) => word.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, '\\$1'));
	const pattern = escapedWords && escapedWords.map((word) => `\\[?${word}\\]?`).join('[\\s\\S]*');
	const levelIndex = filters.level ? logLevels.indexOf(filters.level) : null;
	const validLevels = filters.level ? new Set(logLevels.filter((_, i) => i <= (levelIndex as number))) : null;
	const filterTypes = filters.types?.length ? new Set(filters.types) : null;

	const filteredLogs = logs.filter((log) => {
		if (validLevels && !validLevels.has(log.level)) return false;
		if (pattern) {
			const regex = new RegExp(pattern, 'i');
			if (!regex.test(`[${log.level}]${log.message}`)) return false;
		}
		if (filterTypes) {
			const match = log.message.match(/^\[([^\]]+)\]/);
			const type = match ? match[1] : 'General';
			return filterTypes.has(type);
		}
		return true;
	});
	return filteredLogs;
};

export const getMessageLogsBy = (type: 'level' | 'source', correlation_id: string): any => {
	if (!correlation_id) return {};
	const logsJson = JSON.parse(localStorage.logs || '{}');
	if (type === 'level') {
		return groupBy(logsJson[correlation_id], (val: Log) => val.level);
	}
	const values = logsJson[correlation_id];
	return groupBy(values, (val: Log) => {
		const match = val.message.match(/^\[([^\]]+)\]/);
		return match ? match[1] : 'General';
	});
};
