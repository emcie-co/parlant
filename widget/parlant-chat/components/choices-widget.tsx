// The built-in "choices" widget: a tool offers a few options, the visitor clicks one, and
// that option is sent as the visitor's own chat message.
//
//   { "widget": { "type": "choices", "id": "meeting-dates", "prompt": "Which day suits you?",
//                 "options": [{ "id": "2026-11-03", "label": "Tue, Nov 3" },
//                             { "id": "2026-11-04", "label": "Wed, Nov 4",
//                               "reply": "Wednesday, Nov 4 works" }] } }
//
// Clicking sends `reply ?? label`. Once an option is picked, or any later customer message
// exists, the buttons are disabled; an earlier answer that matches an option's text marks it.

import { useRef, useState } from "react";

import { cn } from "../lib/cn";
import type { WidgetContext, WidgetProps } from "./tool-widget";

export const MAX_CHOICES = 8;

export interface ChoiceOption {
  id: string;
  label: string;
  reply?: string;
}

export interface Choices {
  id?: string;
  prompt?: string;
  options: ChoiceOption[];
}

const nonEmpty = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;

/** Validate a "choices" payload; `null` when it's malformed (so nothing renders). */
export function parseChoices(props: WidgetProps): Choices | null {
  const options = props?.options;
  if (!Array.isArray(options) || options.length < 1 || options.length > MAX_CHOICES) return null;

  const parsed: ChoiceOption[] = [];
  const ids = new Set<string>();
  for (const o of options) {
    if (!o || typeof o !== "object") return null;
    if (!nonEmpty(o.id) || !nonEmpty(o.label) || ids.has(o.id)) return null;
    if (o.reply !== undefined && o.reply !== null && !nonEmpty(o.reply)) return null;
    ids.add(o.id);
    parsed.push({ id: o.id, label: o.label, reply: nonEmpty(o.reply) ? o.reply : undefined });
  }

  return {
    id: nonEmpty(props.id) ? props.id : undefined,
    prompt: nonEmpty(props.prompt) ? props.prompt : undefined,
    options: parsed,
  };
}

/** The text an option sends as the visitor's message. */
export const choiceText = (o: ChoiceOption): string => (o.reply ?? o.label).trim();

export function ChoicesWidget({
  choices,
  toolCallId,
  context,
}: {
  choices: Choices;
  toolCallId: string;
  context: WidgetContext;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  // A ref, not just state: a double click lands before the re-render that disables.
  const sentRef = useRef(false);

  const answer = context.nextCustomerMessage?.trim();
  const chosen =
    picked ??
    (answer !== undefined ? choices.options.find((o) => choiceText(o) === answer)?.id : undefined) ??
    null;
  const disabled = picked !== null || !context.isLatest;

  const choose = (o: ChoiceOption) => {
    if (sentRef.current || disabled) return;
    sentRef.current = true;
    setPicked(o.id);
    context.sendMessage(choiceText(o));
  };

  return (
    <div
      role="group"
      aria-label={choices.prompt ?? "Choose an option"}
      aria-disabled={disabled || undefined}
      data-choices-id={choices.id}
      className={cn("parlant-choices", disabled && "parlant-choices--disabled")}
    >
      {choices.options.map((o) => (
        <button
          key={`${toolCallId}:${o.id}`}
          type="button"
          disabled={disabled}
          aria-pressed={chosen === o.id}
          onClick={() => choose(o)}
          className={cn("parlant-choice", chosen === o.id && "parlant-choice--chosen")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
