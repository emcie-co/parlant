/* eslint-disable @typescript-eslint/no-explicit-any */
import {clsx, type ClassValue} from 'clsx';
import {toast} from 'sonner';
import {twMerge} from 'tailwind-merge';
import './broadcast-channel';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export const isSameDay = (dateA: string | Date, dateB: string | Date): boolean => {
	if (!dateA) return false;
	return new Date(dateA).toLocaleDateString() === new Date(dateB).toLocaleDateString();
};

export const copy = (text: string, element?: HTMLElement) => {
	if (navigator.clipboard && navigator.clipboard.writeText) {
		navigator.clipboard
			.writeText(text)
			.then(() => toast.info(text?.length < 100 ? `Copied text: ${text}` : 'Text copied'))
			.catch(() => {
				fallbackCopyText(text, element);
			});
	} else {
		fallbackCopyText(text, element);
	}
};

export const fallbackCopyText = (text: string, element?: HTMLElement) => {
	const textarea = document.createElement('textarea');
	textarea.value = text;
	(element || document.body).appendChild(textarea);
	textarea.style.position = 'fixed';
	textarea.select();
	try {
		const successful = document.execCommand('copy');
		if (successful) {
			toast.info(text?.length < 100 ? `Copied text: ${text}` : 'Text copied');
		} else {
			console.error('Fallback: Copy command failed.');
		}
	} catch (error) {
		console.error('Fallback: Unable to copy', error);
	} finally {
		(element || document.body).removeChild(textarea);
	}
};

export const timeAgo = (date: Date): string => {
	date = new Date(date);
	const now = new Date();
	const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
	const minutes = Math.floor(seconds / 60);
	const hours = Math.floor(minutes / 60);
	const days = Math.floor(hours / 24);
	// const weeks = Math.floor(days / 7);
	// const months = Math.floor(days / 30);
	const years = Math.floor(days / 365);

	if (seconds < 60) return 'less than a minute ago';
	if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
	if (hours < 24) return date.toLocaleTimeString('en-US', {hour: 'numeric', minute: 'numeric', hour12: false});
	else return date.toLocaleString('en-US', {year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', hour12: false});
	// if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
	// if (days === 1) return 'yesterday';
	// if (days < 7) return `${days} days ago`;
	// if (weeks === 1) return 'last week';
	// if (weeks < 4) return `${weeks} weeks ago`;
	// if (months === 1) return 'a month ago';
	// if (months < 12) return `${months} months ago`;
	// if (years === 1) return 'last year';
	return `${years} years ago`;
};

export const exportToCsv = (data: any[], filename: string, options: any = {}) => {
  try {
    const {
      headers = [],
      delimiter = ',',
      includeHeaders = true,
      dateFormat = 'iso'
    } = options;

    if (!data || data.length === 0) {
      throw new Error('No data to export');
    }

    const csvHeaders = headers.length > 0 ? headers : Object.keys(data[0]);
    
    const escapeField = (field: string) => {
      const stringField = String(field || '');
      if (stringField.includes(delimiter) || stringField.includes('"') || stringField.includes('\n')) {
        return `"${stringField.replace(/"/g, '""')}"`;
      }
      return stringField;
    };

    const formatValue = (value: string | Date) => {
      if (value instanceof Date) {
        return dateFormat === 'readable' ? value.toLocaleString() : value.toISOString();
      }
      return value;
    };

    const csvRows = [];
    
    if (includeHeaders) {
      csvRows.push(csvHeaders.map((header: string) => escapeField(header)).join(delimiter));
    }
    
    data.forEach(row => {
      const values = csvHeaders.map((header: string) => {
        const value = row[header];
        return escapeField(formatValue(value));
      });
      csvRows.push(values.join(delimiter));
    });

    const csvContent = csvRows.join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('CSV export failed:', error);
    throw error;
  }
};

function openIndexeddbDB(dbName: string, storeName: string, indexVals?: {name: string, keyPath: string}) {
	return new Promise<IDBDatabase>((resolve, reject) => {
		const request = indexedDB.open(dbName, 1);
    request.onupgradeneeded = () => {
			const db = request.result;

			if (!db.objectStoreNames.contains(storeName)) {
				const store = db.createObjectStore(storeName, {autoIncrement: true});
        if (indexVals) {
          store.createIndex(indexVals.name, indexVals.keyPath, {unique: false});
        }
			}
		};

		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
}

export const addItemToIndexedDB = async (
  dbName: string,
  storeName: string,
  key: string,
  value: any,
  mode: 'update' | 'multiple' = 'update',
  indexVals?: {name: string, keyPath: string},
) => {
  const db = await openIndexeddbDB(dbName, storeName, indexVals);
  const transaction = db.transaction(storeName, 'readwrite');
  const store = transaction.objectStore(storeName);

  if (mode === 'multiple') {
    const getRequest = store.get(key);
    getRequest.onsuccess = () => {
      let current = getRequest.result;
      if (!Array.isArray(current)) {
        current = current !== undefined ? [current] : [];
      }
      current.push(value);
      const putRequest = store.put(current, key);
      putRequest.onsuccess = () => {
        console.log('Item appended in IndexedDB');
      };
      putRequest.onerror = () => {
        console.error('Error appending item in IndexedDB');
      };
    };
    getRequest.onerror = () => {
      console.error('Error getting item for multiple mode in IndexedDB');
    };
  } else {
    const request = store.put(value, key);
    request.onerror = () => {
      console.error('Error updating item in IndexedDB');
    };
  }
};

export const deleteItemFromIndexedDB = async (dbName: string, storeName: string, key: string) => { 
  const db = await openIndexeddbDB(dbName, storeName);
  const transaction = db.transaction(storeName, 'readwrite');
  const store = transaction.objectStore(storeName);
  const request = store.delete(key);
  request.onerror = () => {
    console.error('Error deleting item in IndexedDB');
  };
};

export const getItemFromIndexedDB = async (dbName: string, storeName: string, key: string, indexVals?: {name: string, keyPath: string}) => {
  try {

    const db = await openIndexeddbDB(dbName, storeName, indexVals);
    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const response = await new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return response;
  } catch (error) {
    console.error('Error opening IndexedDB:', error);
    return null;
  }
};

export const getIndexedItemsFromIndexedDB = async (dbName: string, storeName: string, indexName: string, indexKey: string, indexVals?: {name: string, keyPath: string}, asObject?: boolean) => {
  try {
    const db = await openIndexeddbDB(dbName, storeName, indexVals);
    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const index = store.index(indexName);
    const response: any = await new Promise((resolve, reject) => {
      const request = index.getAll(indexKey);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return asObject ? response.reduce((acc: Record<string, string>, item: any) => {
      acc[item.correlationId] = item.flagValue;
      return acc;
    }, {} as Record<string, string>) : response;
  } catch (error) {
    console.error('Error opening IndexedDB:', error);
    return null;
  }
}