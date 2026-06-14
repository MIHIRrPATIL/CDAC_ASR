"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Home" },
    { href: "/features", label: "Features" },
    { href: "/analyzer", label: "Analyzer" },
  ];

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "24px 48px",
        background: "var(--background)",
        borderBottom: "1px solid var(--surface-container-highest)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      <Link
        href="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          textDecoration: "none",
        }}
      >
        <Activity size={24} color="var(--primary-container)" />
        <span
          style={{
            fontWeight: 700,
            fontSize: "1.25rem",
            color: "var(--on-background)",
            letterSpacing: "-0.02em",
          }}
        >
          LinguaScore
        </span>
      </Link>

      <div style={{ display: "flex", gap: "32px", alignItems: "center" }}>
        {links.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              style={{
                fontSize: "0.875rem",
                fontWeight: 500,
                color: isActive
                  ? "var(--primary-container)"
                  : "var(--on-surface-variant)",
                textDecoration: "none",
                transition: "color 0.2s",
              }}
            >
              {link.label}
            </Link>
          );
        })}
        <Link
          href="/analyzer"
          className="btn-primary"
          style={{ textDecoration: "none", marginLeft: "16px" }}
        >
          Try Analyzer
        </Link>
      </div>
    </motion.nav>
  );
}
