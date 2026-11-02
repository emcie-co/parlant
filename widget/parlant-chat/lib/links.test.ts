import { describe, expect, it } from "vitest";

import { appendUtm } from "./links";

const PAGE = "https://www.example.com/pricing";

describe("appendUtm", () => {
  it("test_that_utm_params_are_appended_to_an_https_link", () => {
    expect(
      appendUtm("https://docs.example.com/start", { source: "site-chat", medium: "chat-widget" }),
    ).toBe("https://docs.example.com/start?utm_source=site-chat&utm_medium=chat-widget");
  });

  it("test_that_utm_params_are_appended_to_an_http_link", () => {
    expect(appendUtm("http://example.org/", { campaign: "spring" })).toBe(
      "http://example.org/?utm_campaign=spring",
    );
  });

  it("test_that_a_links_existing_query_and_fragment_are_kept", () => {
    expect(
      appendUtm("https://example.org/docs?lang=en#install", { source: "site-chat" }),
    ).toBe("https://example.org/docs?lang=en&utm_source=site-chat#install");
  });

  it("test_that_a_utm_key_already_on_the_link_is_not_overridden", () => {
    expect(
      appendUtm("https://example.org/?utm_source=newsletter", {
        source: "site-chat",
        medium: "chat-widget",
      }),
    ).toBe("https://example.org/?utm_source=newsletter&utm_medium=chat-widget");
  });

  it("test_that_every_supported_utm_key_gets_its_prefix", () => {
    const href = appendUtm("https://example.org/", {
      source: "s",
      medium: "m",
      campaign: "c",
      term: "t",
      content: "x",
      id: "i",
    });
    expect([...new URL(href).searchParams.entries()]).toEqual([
      ["utm_source", "s"],
      ["utm_medium", "m"],
      ["utm_campaign", "c"],
      ["utm_term", "t"],
      ["utm_content", "x"],
      ["utm_id", "i"],
    ]);
  });

  it("test_that_utm_values_are_url_encoded", () => {
    expect(appendUtm("https://example.org/", { campaign: "spring sale & more" })).toBe(
      "https://example.org/?utm_campaign=spring+sale+%26+more",
    );
  });

  it("test_that_mailto_tel_and_relative_links_are_left_unchanged", () => {
    const utm = { source: "site-chat" };
    expect(appendUtm("mailto:sales@example.org", utm)).toBe("mailto:sales@example.org");
    expect(appendUtm("tel:+15551234", utm)).toBe("tel:+15551234");
    expect(appendUtm("/docs/start", utm)).toBe("/docs/start");
    expect(appendUtm("not a url", utm)).toBe("not a url");
  });

  it("test_that_empty_utm_values_are_not_added", () => {
    expect(appendUtm("https://example.org/", { source: "site-chat", medium: "", campaign: "  " })).toBe(
      "https://example.org/?utm_source=site-chat",
    );
  });

  it("test_that_no_utm_params_leave_the_link_byte_for_byte_unchanged", () => {
    expect(appendUtm("https://example.org/a?b=1#c", undefined)).toBe("https://example.org/a?b=1#c");
    expect(appendUtm("https://example.org/a?b=1#c", {})).toBe("https://example.org/a?b=1#c");
  });

  it("test_that_all_links_mode_tags_links_to_the_embedding_pages_own_host", () => {
    expect(
      appendUtm("https://www.example.com/docs", { source: "site-chat" }, { links: "all", pageUrl: PAGE }),
    ).toBe("https://www.example.com/docs?utm_source=site-chat");
  });

  it("test_that_external_links_mode_leaves_links_to_the_embedding_pages_own_host_unchanged", () => {
    expect(
      appendUtm("https://www.example.com/docs", { source: "site-chat" }, { links: "external", pageUrl: PAGE }),
    ).toBe("https://www.example.com/docs");
  });

  it("test_that_external_links_mode_tags_links_to_any_other_host", () => {
    expect(
      appendUtm("https://docs.example.com/start", { source: "site-chat" }, { links: "external", pageUrl: PAGE }),
    ).toBe("https://docs.example.com/start?utm_source=site-chat");
  });
});
