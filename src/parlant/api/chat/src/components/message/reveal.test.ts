import {describe, expect, it} from 'vitest';

import {revealLengthForText} from './reveal';

describe('revealLengthForText', () => {
	it('keeps the revealed length while a streaming message grows', () => {
		// Mid-stream: 12 chars revealed of a 40-char message. Nothing to clamp.
		expect(revealLengthForText(40, 12)).toBe(12);
	});

	it('clamps to the shorter text when a message is annotated after it rendered', () => {
		// The voice case: the bubble fully revealed 80 chars, then the delivery
		// annotation arrived and the text became the 30-char spoken portion.
		// Resetting to 0 here would blank the bubble permanently, because the
		// reveal effect only re-runs when the text changes again — and it won't.
		expect(revealLengthForText(30, 80)).toBe(30);
	});

	it('leaves a fully revealed message alone', () => {
		expect(revealLengthForText(30, 30)).toBe(30);
	});

	it('handles a message annotated as never spoken', () => {
		// spoken_text === "" -> nothing was heard -> nothing to reveal.
		expect(revealLengthForText(0, 80)).toBe(0);
	});
});
