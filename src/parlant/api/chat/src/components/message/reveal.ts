/**
 * How much of `textLength` should be considered already revealed, given that
 * `revealedLength` characters were revealed before the text changed.
 *
 * Text normally only grows while streaming, so this is usually the identity.
 * It matters when the text SHRINKS: a voice message is annotated with what was
 * actually spoken (`metadata.__parlant__.voice.spoken_text`) after it has already been
 * rendered in full, replacing the text with a shorter one. Clamping keeps the
 * shorter text fully visible; resetting to 0 would blank the bubble for good,
 * since the reveal effect only re-runs when the text changes again.
 */
export const revealLengthForText = (textLength: number, revealedLength: number): number => Math.min(revealedLength, textLength);
