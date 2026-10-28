// UTM tagging for the links the widget renders. Pure and dependency-free, so it can tag a
// link anywhere: the Markdown renderer uses it, and custom widgets get it on their context.

/** UTM parameters to add to links, without the `utm_` prefix (e.g. `{ source: "site-chat" }`
 *  becomes `utm_source=site-chat`). Empty values are skipped. */
export interface UtmParams {
  source?: string;
  medium?: string;
  campaign?: string;
  term?: string;
  content?: string;
  id?: string;
}

/** Which links get UTM parameters: every http(s) link ("all"), or only links to a host other
 *  than the page the widget is embedded in ("external"). Hosts are compared exactly, so
 *  `www.example.com` and `docs.example.com` are different hosts. */
export type UtmLinks = "all" | "external";

const UTM_KEYS: (keyof UtmParams)[] = ["source", "medium", "campaign", "term", "content", "id"];

export interface AppendUtmOptions {
  /** Default "all". */
  links?: UtmLinks;
  /** The embedding page, for `links: "external"`. Defaults to `window.location.href`. */
  pageUrl?: string;
}

/** Return `href` with `utm` added to its query. Only http(s) URLs are tagged; anything else
 *  (mailto:, tel:, relative paths, unparseable text) is returned unchanged. A `utm_*` key the
 *  link already has is kept, and so are the rest of its query and its #fragment. With no
 *  parameters to add, `href` is returned byte for byte. */
export function appendUtm(
  href: string,
  utm: UtmParams | undefined,
  options: AppendUtmOptions = {},
): string {
  const entries = UTM_KEYS.flatMap((key) => {
    const value = utm?.[key]?.trim();
    return value ? [[`utm_${key}`, value] as const] : [];
  });
  if (entries.length === 0) return href;

  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return href;
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") return href;

  if ((options.links ?? "all") === "external") {
    const page = options.pageUrl ?? (typeof window !== "undefined" ? window.location.href : "");
    try {
      if (new URL(page).host === url.host) return href;
    } catch {
      /* no usable page URL: treat the link as external */
    }
  }

  const missing = entries.filter(([key]) => !url.searchParams.has(key));
  if (missing.length === 0) return href;
  // Append to the query as written rather than re-serializing it through searchParams, which
  // would re-encode the link's own parameters (e.g. %20 becomes +).
  const added = new URLSearchParams(missing.map(([key, value]) => [key, value])).toString();
  url.search = url.search.length > 1 ? `${url.search}&${added}` : `?${added}`;
  return url.toString();
}
