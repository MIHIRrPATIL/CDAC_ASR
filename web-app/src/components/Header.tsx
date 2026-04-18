import React from "react";

export default function Header() {
  return (
    <header
      style={{
        background: "var(--header-bg)",
        margin: "-48px -32px 0",
        padding: "24px 32px 80px",
        borderRadius: "0 0 16px 16px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          marginBottom: "4px",
        }}
      >
        <div
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            background: "var(--primary-fixed)",
          }}
        />
        <h1
          style={{
            fontSize: "1.25rem",
            fontWeight: 600,
            color: "#ffffff",
            letterSpacing: "-0.01em",
          }}
        >
          Pronunciation Analyzer
        </h1>
      </div>
      <p
        style={{
          fontSize: "0.8125rem",
          color: "rgba(255,255,255,0.6)",
          paddingLeft: "18px",
        }}
      >
        Record or upload your speech to refine your fluency.
      </p>
    </header>
  );
}
