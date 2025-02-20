import {clsx, type ClassValue} from 'clsx';
import {toast} from 'sonner';
import {twMerge} from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export const copy = (text: string) => {
	if (navigator.clipboard && navigator.clipboard.writeText) {
		navigator.clipboard
			.writeText(text)
			.then(() => toast.info(text?.length < 100 ? `Copied text: ${text}` : 'Text copied'))
			.catch(() => {
				fallbackCopyText(text);
			});
	} else {
		fallbackCopyText(text);
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
			toast.info(`Copied text: ${text}`);
		} else {
			console.error('Fallback: Copy command failed.');
		}
	} catch (error) {
		console.error('Fallback: Unable to copy', error);
	} finally {
		(element || document.body).removeChild(textarea);
	}
};
