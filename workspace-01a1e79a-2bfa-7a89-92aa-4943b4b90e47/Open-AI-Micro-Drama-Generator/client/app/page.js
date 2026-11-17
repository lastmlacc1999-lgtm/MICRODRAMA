"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import IdeaForm from "../components/IdeaForm";
import A2EKeySettings from "../components/A2EKeySettings";
import ProjectList from "../components/ProjectList";
import { Film, Zap, GitBranch, Layers } from "lucide-react";

export default function HomePage() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [server, setServer] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState(false);

  useEffect(() => {
    fetch("/api/health", { cache: "no-store" }).then((res) => res.ok ? res.json() : null)
      .then(setServer).catch(() => setServer(null));
    fetch("/api/models", { cache: "no-store" })
      .then((res) => { if (!res.ok) throw new Error("Model catalog unavailable"); return res.json(); })
      .then(setCatalog).catch(() => setCatalogError(true));
  }, []);

  const handleSubmit = async (formData) => {
    setIsSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(typeof err.detail === "string" ? err.detail : "Invalid request; check your inputs.");
      }
      const data = await res.json();
      router.push(`/generate/${data.job_id}`);
    } catch (err) {
      setError(err.message);
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen" style={{ backgroundColor: "#0a0a0f" }}>
      {/* Hero header */}
      <div className="relative overflow-hidden">
        {/* Background glow effects */}
        <div
          className="absolute inset-0 pointer-events-none"
          aria-hidden="true"
        >
          <div
            className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full opacity-20"
            style={{
              background:
                "radial-gradient(ellipse at center, #7c3aed 0%, transparent 70%)",
              filter: "blur(60px)",
            }}
          />
        </div>

        <div className="relative max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-6"
            style={{ backgroundColor: "rgba(124, 58, 237, 0.15)", border: "1px solid rgba(124, 58, 237, 0.3)", color: "#a78bfa" }}>
            <Zap size={12} />
            Gemini or GPT planning · A2E video
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-6xl md:text-7xl font-black tracking-tight mb-6">
            <span className="gradient-text">MicroDrama AI</span>
          </h1>
          <p className="text-xl md:text-2xl font-light mb-3" style={{ color: "#94a3b8" }}>
            Turn a story into moving pictures
          </p>
          <p className="text-base max-w-2xl mx-auto mb-12" style={{ color: "#64748b" }}>
            Start with an idea or a scene. Choose Gemini or GPT for story planning, Gemini or A2E for frames,
            and an A2E video model for your clips.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap justify-center gap-3 mb-16">
            {[
              { icon: <Film size={14} />, label: "Idea or Script" },
              { icon: <GitBranch size={14} />, label: "Shot Planning" },
              { icon: <Layers size={14} />, label: "Frame Generator" },
              { icon: <Zap size={14} />, label: "Video Generator" },
            ].map((f) => (
              <div
                key={f.label}
                className="flex items-center gap-2 px-4 py-2 rounded-full text-sm"
                style={{
                  backgroundColor: "#12121a",
                  border: "1px solid #22223a",
                  color: "#94a3b8",
                }}
              >
                <span style={{ color: "#7c3aed" }}>{f.icon}</span>
                {f.label}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Form section */}
      <div className="max-w-3xl mx-auto px-6 pb-24">
        <A2EKeySettings
          configured={server ? server.a2e_configured : null}
          onSaved={() => {
            setError(null);
            setServer((current) => ({ ...(current || {}), a2e_configured: true }));
          }}
        />
        <A2EKeySettings
          provider="gemini"
          configured={server ? server.gemini_configured : null}
          onSaved={() => {
            setError(null);
            setServer((current) => ({ ...(current || {}), gemini_configured: true }));
          }}
        />
        <A2EKeySettings
          provider="openai"
          configured={server?.openai_configured ?? null}
          onSaved={() => {
            setError(null);
            setServer((current) => ({ ...(current || {}), openai_configured: true }));
          }}
        />
        <ProjectList />
        <div className="mb-5 rounded-xl px-4 py-3 text-sm" style={{
          backgroundColor: "rgba(124, 58, 237, 0.08)",
          border: "1px solid rgba(124, 58, 237, 0.2)", color: "#cbd5e1",
        }}>
          Choose simple planning, Gemini, OpenAI GPT, or a server-configured OpenAI-compatible LLM for each job.
        </div>
        {error && (
          <div
            className="mb-6 px-4 py-3 rounded-xl text-sm animate-fade-in"
            style={{
              backgroundColor: "rgba(239, 68, 68, 0.1)",
              border: "1px solid rgba(239, 68, 68, 0.3)",
              color: "#fca5a5",
            }}
          >
            {error}
          </div>
        )}
        <IdeaForm onSubmit={handleSubmit} isSubmitting={isSubmitting} catalog={catalog} catalogError={catalogError}
          geminiConfigured={server?.gemini_configured ?? null} openaiConfigured={server?.openai_configured ?? null} />
      </div>

      {/* Footer */}
      <footer className="text-center pb-8 text-sm" style={{ color: "#374151" }}>
        MicroDrama AI &mdash; Flexible image and video generation
      </footer>
    </main>
  );
}
