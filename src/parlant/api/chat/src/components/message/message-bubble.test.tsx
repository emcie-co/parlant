import {fireEvent, render, screen, waitFor} from '@testing-library/react';
import {beforeEach, describe, expect, it, vi} from 'vitest';
import {EventInterface} from '@/utils/interfaces';
import MessageBubble from './message-bubble';
import {DiagnosticReportDownloadError} from '@/utils/diagnostic-report-download';

const downloadDiagnosticReport = vi.hoisted(() => vi.fn());
const toastError = vi.hoisted(() => vi.fn());

vi.mock('@/utils/diagnostic-report-download', async (importOriginal) => {
	const original = await importOriginal<
		typeof import('@/utils/diagnostic-report-download')
	>();
	return {...original, downloadDiagnosticReport};
});

vi.mock('sonner', () => ({
	toast: {error: toastError},
}));

const customerEvent = (withReport: boolean): EventInterface => ({
	id: 'message-1',
	source: 'customer',
	kind: 'message',
	trace_id: 'trace-1',
	serverStatus: 'error',
	offset: 0,
	creation_utc: new Date(),
	data: {message: 'Request that failed'},
	metadata: withReport
		? {__parlant__: {schema_version: 1, diagnostic_report_uri: '/debug/reports/trace-1'}}
		: undefined,
});

const renderBubble = (event: EventInterface) =>
	render(
		<MessageBubble
			event={event}
			isContinual={false}
			showLogs={() => undefined}
		/>
	);

describe('MessageBubble diagnostic report action', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it('is shown only when an errored customer message has a report URI', () => {
		const {rerender} = renderBubble(customerEvent(true));

		expect(
			screen.getByRole('button', {name: 'Download diagnostic report'})
		).not.toBeNull();

		rerender(
			<MessageBubble
				event={customerEvent(false)}
				isContinual={false}
				showLogs={() => undefined}
			/>
		);

		expect(
			screen.queryByRole('button', {name: 'Download diagnostic report'})
		).toBeNull();
	});

	it('shows a disabled loading state while the report downloads', async () => {
		let finishDownload: () => void = () => undefined;
		downloadDiagnosticReport.mockReturnValueOnce(
			new Promise<void>((resolve) => {
				finishDownload = resolve;
			})
		);
		renderBubble(customerEvent(true));

		fireEvent.click(
			screen.getByRole('button', {name: 'Download diagnostic report'})
		);

		const loadingButton = screen.getByRole('button', {name: 'Downloading...'});
		expect((loadingButton as HTMLButtonElement).disabled).toBe(true);
		expect(downloadDiagnosticReport).toHaveBeenCalledWith('/debug/reports/trace-1');

		finishDownload();
		await waitFor(() =>
			expect(
				(screen.getByRole('button', {
					name: 'Download diagnostic report',
				}) as HTMLButtonElement).disabled
			).toBe(false)
		);
	});

	it('explains when administrator access is required', async () => {
		downloadDiagnosticReport.mockRejectedValueOnce(new DiagnosticReportDownloadError(403));
		renderBubble(customerEvent(true));

		fireEvent.click(
			screen.getByRole('button', {name: 'Download diagnostic report'})
		);

		await waitFor(() =>
			expect(toastError).toHaveBeenCalledWith(
				'Administrator access is required to download this diagnostic report.'
			)
		);
	});

	it('shows a generic toast for other download failures', async () => {
		downloadDiagnosticReport.mockRejectedValueOnce(new Error('network failed'));
		renderBubble(customerEvent(true));

		fireEvent.click(
			screen.getByRole('button', {name: 'Download diagnostic report'})
		);

		await waitFor(() =>
			expect(toastError).toHaveBeenCalledWith(
				'Could not download the diagnostic report. Please try again.'
			)
		);
	});
});
