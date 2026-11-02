import {describe, expect, it} from 'vitest';

import {deliveredMessageText} from './delivered-text';
import type {EventInterface} from './interfaces';

const event = (message: string, spokenText?: string): EventInterface =>
	({
		source: 'ai_agent',
		kind: 'message',
		trace_id: 'trace-1',
		offset: 0,
		creation_utc: new Date(),
		serverStatus: 'ready',
		data: {message},
		...(spokenText === undefined
			? {}
			: {metadata: {__parlant__: {schema_version: 1, voice: {spoken_text: spokenText}}}}),
	}) as EventInterface;

describe('deliveredMessageText', () => {
	it('uses the message text when there is no voice annotation', () => {
		expect(deliveredMessageText(event('Hello there.'))).toBe('Hello there.');
	});

	it('uses the spoken text when the message was annotated', () => {
		expect(deliveredMessageText(event('Hello there. How can I help?', 'Hello there. '))).toBe('Hello there. ');
	});

	it('renders nothing for an empty spoken text, because none of it was heard', () => {
		// The distinction that matters: "" means the customer heard NOTHING, and
		// must not fall back to the generated text. A truthiness check (`||`)
		// would silently show words that were never spoken.
		expect(deliveredMessageText(event('This never reached the customer.', ''))).toBe('');
	});

	it('falls back to the message text when the voice metadata carries no spoken text', () => {
		const partial = {
			...event('Hello there.'),
			metadata: {__parlant__: {schema_version: 1, voice: {}}},
		} as EventInterface;

		expect(deliveredMessageText(partial)).toBe('Hello there.');
	});

	it('returns an empty string for an event with no message at all', () => {
		expect(deliveredMessageText({data: {}} as EventInterface)).toBe('');
		expect(deliveredMessageText(undefined)).toBe('');
	});
});
