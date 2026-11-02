// Minimal className joiner (no dependency). Swap for clsx + tailwind-merge if you prefer.
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
