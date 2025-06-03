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