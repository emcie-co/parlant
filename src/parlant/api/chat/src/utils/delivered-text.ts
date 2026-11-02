import type {EventInterface} from './interfaces';

/**
 * The message text the UI should show as what was actually said.
 *
 * Voice delivery annotates a message event with the portion that was really
 * played before an interruption (`metadata.__parlant__.voice.spoken_text`, empty when
 * nothing was heard). The event's `data.message` keeps the model's full
 * output; the annotation is the delivery truth. Absent the annotation, the
 * message text IS the delivered truth.
 *
 * Mirrors `delivered_message_text` in parlant/core/sessions.py — keep the two
 * in step. Note the `typeof === 'string'` check rather than a truthy one: an
 * empty spoken text means the customer heard nothing, which must render as
 * nothing rather than falling back to words never spoken.
 */
export const deliveredMessageText = (event?: EventInterface | null): string => {
	const spokenText = event?.metadata?.__parlant__?.voice?.spoken_text;

	if (typeof spokenText === 'string') return spokenText;

	return event?.data?.message || '';
};
