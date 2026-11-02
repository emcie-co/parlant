export class DiagnosticReportDownloadError extends Error {
	constructor(public readonly status: number) {
		super(`Diagnostic report download failed with status ${status}`);
		this.name = 'DiagnosticReportDownloadError';
	}
}

const filenameFromDisposition = (value: string | null): string | undefined => {
	if (!value) return undefined;

	const encoded = value.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
	if (encoded) {
		try {
			return decodeURIComponent(encoded);
		} catch {
			return encoded;
		}
	}

	return value.match(/filename="?([^";]+)"?/i)?.[1];
};

const fallbackFilename = (url: string): string => {
	try {
		const name = new URL(url, window.location.origin).pathname.split('/').at(-1);
		return name?.endsWith('.zip') ? name : `${name || 'diagnostic-report'}.zip`;
	} catch {
		return 'diagnostic-report.zip';
	}
};

export const downloadDiagnosticReport = async (
	url: string,
	fetcher: typeof fetch = fetch
): Promise<string> => {
	const response = await fetcher(url);
	if (!response.ok) throw new DiagnosticReportDownloadError(response.status);

	const blob = await response.blob();
	const filename =
		filenameFromDisposition(response.headers.get('Content-Disposition')) ||
		fallbackFilename(url);
	const objectUrl = URL.createObjectURL(blob);
	const link = document.createElement('a');

	try {
		link.href = objectUrl;
		link.download = filename;
		link.hidden = true;
		document.body.appendChild(link);
		link.click();
	} finally {
		link.remove();
		URL.revokeObjectURL(objectUrl);
	}

	return filename;
};
