import {afterEach, describe, expect, it, vi} from 'vitest';
import {
	DiagnosticReportDownloadError,
	downloadDiagnosticReport,
} from './diagnostic-report-download';

describe('downloadDiagnosticReport', () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('downloads the response blob using the server-provided filename', async () => {
		const click = vi
			.spyOn(HTMLAnchorElement.prototype, 'click')
			.mockImplementation(() => undefined);
		vi.stubGlobal('URL', {
			...URL,
			createObjectURL: vi.fn(() => 'blob:report'),
			revokeObjectURL: vi.fn(),
		});
		const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
			new Response(new Blob(['zip-data']), {
				status: 200,
				headers: {
					'Content-Disposition': 'attachment; filename="trace-1.zip"',
				},
			})
		);

		const filename = await downloadDiagnosticReport('/debug/reports/trace-1', fetcher);

		expect(filename).toBe('trace-1.zip');
		expect(fetcher).toHaveBeenCalledWith('/debug/reports/trace-1');
		expect(click).toHaveBeenCalledOnce();
		expect(URL.createObjectURL).toHaveBeenCalledOnce();
		expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:report');
	});

	it('exposes the HTTP status when a download is rejected', async () => {
		const fetcher = vi
			.fn<typeof fetch>()
			.mockResolvedValue(new Response(null, {status: 403}));

		await expect(
			downloadDiagnosticReport('/debug/reports/trace-1', fetcher)
		).rejects.toEqual(new DiagnosticReportDownloadError(403));
	});
});
