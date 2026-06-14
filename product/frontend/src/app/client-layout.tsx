"use client";

import { AuthProvider } from "@/lib/auth-context";
import { DynamicNavbar } from "@/components/dynamic-navbar";
import InfiniteGridBackground from "@/components/InfiniteGridBackground";
import { usePathname } from "next/navigation";

export function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isAuthPage = pathname === "/auth";
  const isLandingPage = pathname === "/";
  const hideNav = isAuthPage || isLandingPage;

  return (
    <AuthProvider>
      {!hideNav && <InfiniteGridBackground />}
      {!hideNav && <DynamicNavbar />}
      {children}
    </AuthProvider>
  );
}
