/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PARLANT_URL: string;
  readonly VITE_AGENT_ID: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
