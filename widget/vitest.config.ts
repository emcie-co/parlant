import { defineConfig, mergeConfig } from "vitest/config";

import viteConfig from "./vite.config";

export default mergeConfig(
  viteConfig,
  defineConfig({
    // Vitest stubs stylesheets to "" by default; styles.test.ts reads this one's text.
    test: { css: { include: [/\/parlant-chat\/styles\.css/] } },
  }),
);
