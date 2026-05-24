import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI 奶茶配方生成",
  description: "用自然语言生成现代茶饮配方方案",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
