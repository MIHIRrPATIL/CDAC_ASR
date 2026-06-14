"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  Activity,
  Mic,
  BarChart3,
  Info,
  Server,
  Settings,
  FileText,
  ArrowLeft,
  MoreHorizontal,
  Menu,
  X,
  Rocket,
  LogOut,
  LogIn,
} from "lucide-react";

// ASR Navigation Items
const mainNavItems = [
  { label: "Analyzer", href: "/analyzer", icon: Mic },
  { label: "Dashboard", href: "/dashboard", icon: BarChart3 },
  { label: "Features", href: "/features", icon: Rocket },
];

const moreNavItems = [
  { label: "Documentation", href: "/docs", icon: FileText },
  { label: "Settings", href: "/settings", icon: Settings },
];

export function DynamicNavbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAuthenticated, logout } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [isCompressed, setIsCompressed] = useState(false);
  const [isMoreHovered, setIsMoreHovered] = useState(false);
  const lastScrollY = useRef(0);
  const menuTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      const scrollDelta = currentScrollY - lastScrollY.current;

      if (scrollDelta < -5) {
        setIsVisible(true);
        setIsCompressed(false);
      } else if (scrollDelta > 5 && currentScrollY > 100) {
        setIsCompressed(true);
      } else if (scrollDelta > 10 && currentScrollY > 200) {
        setIsVisible(false);
      }

      lastScrollY.current = currentScrollY;
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleNavEnter = () => {
    if (menuTimeoutRef.current) clearTimeout(menuTimeoutRef.current);
  };

  const handleNavLeave = () => {
    menuTimeoutRef.current = setTimeout(() => {
      setIsMoreHovered(false);
    }, 300);
  };

  const handleMoreEnter = () => {
    if (menuTimeoutRef.current) clearTimeout(menuTimeoutRef.current);
    setIsMoreHovered(true);
  };

  const handleLogout = async () => {
    await logout();
    router.push("/auth");
  };

  const isDashboardRoute =
    pathname?.startsWith("/dashboard") || pathname?.startsWith("/analyzer");

  return (
    <nav
      className={`fixed top-0 z-50 w-full transition-all duration-300 ease-out ${isDashboardRoute ? "bg-background/80" : ""} ${isVisible ? "translate-y-0" : "-translate-y-full"} ${isCompressed ? "py-2" : "py-4"}`}
    >
      <div
        className={`absolute inset-0 backdrop-blur-xl border-b transition-colors ${isDashboardRoute ? "bg-background/80 border-border/30" : "bg-background/40 border-border/20"}`}
      />

      <div className="relative max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className={`flex items-center gap-2 font-bold transition-all duration-300 ${isCompressed ? "text-sm" : "text-lg"} ${isDashboardRoute ? "text-foreground" : "text-white"}`}
          >
            <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-white">
              <Activity size={20} />
            </div>
            <span
              style={{
                fontSize: "1.25rem",
                letterSpacing: "-0.02em",
                color: isDashboardRoute ? "var(--foreground)" : "currentColor",
              }}
            >
              LinguaScore
            </span>
          </Link>

          <div
            onMouseLeave={handleNavLeave}
            onMouseEnter={handleNavEnter}
            className={`hidden xl:flex items-center relative overflow-hidden border rounded-full backdrop-blur-md transition-all duration-500 shadow-lg w-[700px] ${isCompressed ? "px-2 py-1.5" : "px-4 py-2"} ${!isMoreHovered ? "bg-muted/30 border-primary/20 hover:bg-muted/40 hover:shadow-xl" : "border-orange-400/50 shadow-orange-500/20"}`}
          >
            <div
              className={`absolute inset-0 bg-orange-600 transition-transform duration-500 origin-right ${isMoreHovered ? "scale-x-100" : "scale-x-0"}`}
            />

            <div
              className={`flex items-center flex-1 transition-all duration-300 ${isMoreHovered ? "opacity-0 translate-x-4 pointer-events-none absolute inset-y-0 left-0 w-full" : "opacity-100 translate-x-0 relative w-full"}`}
            >
              {mainNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex-1 px-1"
                  >
                    <div
                      className={`flex items-center justify-center gap-2 rounded-full font-medium transition-all duration-300 ${isCompressed ? "text-xs px-2 py-1" : "text-sm px-3 py-1.5"} ${isActive ? "bg-linear-to-r from-orange-500 to-orange-400 text-white shadow-md scale-105" : isDashboardRoute ? "text-foreground/80 hover:text-foreground hover:bg-foreground/10" : "text-muted-foreground hover:text-foreground hover:bg-white/20"}`}
                    >
                      <Icon
                        className={`shrink-0 transition-all ${isCompressed ? "w-3 h-3" : "w-4 h-4"}`}
                      />
                      <span
                        className={`transition-all ${isCompressed ? "text-xs" : "text-sm"}`}
                      >
                        {item.label}
                      </span>
                    </div>
                  </Link>
                );
              })}
            </div>

            <div
              className={`flex items-center flex-1 transition-all duration-300 ${!isMoreHovered ? "opacity-0 -translate-x-4 pointer-events-none absolute inset-y-0 left-0 w-full" : "opacity-100 translate-x-0 relative w-full"}`}
            >
              {moreNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex-1 px-1"
                  >
                    <div
                      className={`flex items-center justify-center gap-2 rounded-full font-medium transition-all duration-300 z-10 relative w-full ${isCompressed ? "text-xs px-2 py-1" : "text-sm px-3 py-1.5"} ${isActive ? "bg-white text-orange-600 shadow-md" : "text-white/80 hover:text-white hover:bg-white/10"}`}
                    >
                      <Icon
                        className={`shrink-0 transition-all ${isCompressed ? "w-3 h-3" : "w-4 h-4"}`}
                      />
                      <span
                        className={`transition-all ${isCompressed ? "text-xs" : "text-sm"}`}
                      >
                        {item.label}
                      </span>
                    </div>
                  </Link>
                );
              })}
            </div>

            <div
              className={`h-4 w-px mx-4 transition-colors duration-300 z-10 relative ${isMoreHovered ? "bg-white/30" : "bg-foreground/20"}`}
            />

            <div
              className="flex items-center z-10 relative"
              onMouseEnter={handleMoreEnter}
            >
              <div
                className={`flex items-center gap-2 rounded-full font-medium transition-all duration-300 cursor-pointer ${isCompressed ? "text-xs px-2 py-1" : "text-sm px-3 py-1.5"} ${isMoreHovered ? "text-white bg-white/10" : isDashboardRoute ? "text-foreground/80 hover:text-foreground hover:bg-foreground/10" : "text-muted-foreground hover:text-foreground hover:bg-white/20"}`}
              >
                {isMoreHovered ? (
                  <ArrowLeft
                    className={`shrink-0 transition-all ${isCompressed ? "w-3 h-3" : "w-4 h-4"}`}
                  />
                ) : (
                  <MoreHorizontal
                    className={`shrink-0 transition-all ${isCompressed ? "w-3 h-3" : "w-4 h-4"}`}
                  />
                )}
                <span
                  className={`hidden lg:inline transition-all ${isCompressed ? "text-xs" : "text-sm"}`}
                >
                  {isMoreHovered ? "Back" : "More"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            {isAuthenticated ? (
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 bg-foreground/10 hover:bg-foreground/20 text-foreground px-4 py-2 rounded-lg text-sm font-semibold transition-all"
              >
                <LogOut size={16} />
                Logout
              </button>
            ) : (
              <Link href="/auth">
                <button className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-lg text-sm font-semibold hover:opacity-90 transition-opacity">
                  <LogIn size={16} />
                  Login
                </button>
              </Link>
            )}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className={`md:hidden p-2 rounded-md ${isDashboardRoute ? "text-foreground" : "text-white"}`}
            >
              {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
