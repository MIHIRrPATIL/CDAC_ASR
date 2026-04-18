import type { Metadata } from "next";
import "./globals.css";
import { ClientLayout } from "@/app/client-layout";

export const metadata: Metadata = {
  title: "VoiceScore - Pronunciation Analyzer",
  description: "AI-powered clinical pronunciation scoring",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
