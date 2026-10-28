import {describe, expect, it} from 'vitest';
import {EventInterface} from './interfaces';
import {withLatestTraceStatus} from './event-status';

const event = (overrides: Partial<EventInterface> = {}): EventInterface => ({
	id: 'event-1',
	source: 'customer',
	kind: 'message',
	trace_id: 'trace-1',
	serverStatus: 'pending',
	offset: 0,
	creation_utc: new Date(),
	data: {message: 'Something went wrong'},
	...overrides,
});

describe('withLatestTraceStatus', () => {
	it('associates error details and the diagnostic report URI with the customer message', () => {
		const message = event({metadata: {__parlant__: {schema_version: 1, voice: {spoken_text: 'heard'}}}});
		const status = event({
			id: 'status-1',
			kind: 'status',
			source: 'ai_agent',
			data: {
				message: '',
				status: 'error',
				data: {exception: 'RuntimeError: sentinel'},
			},
			metadata: {
				__parlant__: {
					schema_version: 1,
					diagnostic_report_uri: '/debug/reports/trace-1',
				},
			},
		});

		const result = withLatestTraceStatus(message, [message, status]);

		expect(result.serverStatus).toBe('error');
		expect(result.error).toBe('RuntimeError: sentinel');
		expect(result.metadata).toEqual({
			__parlant__: {
				schema_version: 1,
				voice: {spoken_text: 'heard'},
				diagnostic_report_uri: '/debug/reports/trace-1',
			},
		});
	});

	it('does not add the metadata key when the status has no report', () => {
		const message = event();
		const status = event({
			kind: 'status',
			data: {message: '', status: 'error'},
		});

		const result = withLatestTraceStatus(message, [message, status]);

		expect(result.serverStatus).toBe('error');
		expect(result.metadata).toBeUndefined();
	});
});
