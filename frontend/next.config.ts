import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The browser tests reach `next dev` at 127.0.0.1, which Next 16 otherwise treats as a
  // foreign origin and refuses its dev resources (no hydration). Ignored by `next start`.
  allowedDevOrigins: ["127.0.0.1"],
};

export default config;
