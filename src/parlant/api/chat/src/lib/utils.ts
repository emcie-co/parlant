import {clsx, type ClassValue} from 'clsx';
import {toast} from 'sonner';
import {twMerge} from 'tailwind-merge';

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
