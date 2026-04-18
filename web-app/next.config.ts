import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // @ts-ignore
  experimental: {
    reactCompiler: false,
  },
};

export default nextConfig;
