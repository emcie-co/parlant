// Hands the widget's UTM setting down to everything that renders a link, without threading it
// through every component in between. ChatWidget provides it; the Markdown renderer and the
// custom-widget context read it.

import { createContext, useContext } from "react";

/** Returns the link to render for `href`: with the configured UTM parameters, if any apply. */
export type LinkTagger = (href: string) => string;

const untouched: LinkTagger = (href) => href;

export const LinkTaggerContext = createContext<LinkTagger>(untouched);

export function useLinkTagger(): LinkTagger {
  return useContext(LinkTaggerContext);
}
