import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 消除 workspace root 推断警告
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
