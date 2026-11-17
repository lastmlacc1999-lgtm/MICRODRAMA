import "./globals.css";

// The HTML must not be cached across deployments: its script references need to
// match the current, content-hashed client bundle after an upgrade.
export const dynamic = "force-dynamic";

export const metadata = {
  title: "MicroDrama AI — Agentic Micro-Drama Generator",
  description:
    "Transform an idea into video using Gemini or A2E images, A2E video, and optional AI story planning.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen" style={{ backgroundColor: "#0a0a0f" }}>
        {children}
      </body>
    </html>
  );
}
