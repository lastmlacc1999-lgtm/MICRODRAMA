"use client";

import { useState } from "react";
import { Check, ExternalLink, Eye, EyeOff, KeyRound, Loader2 } from "lucide-react";

// Keys stay server-side after saving; each instance manages its own input.
const PROVIDERS = {
  a2e: {
    name: "A2E", description: "Required for video generation and A2E images.",
    placeholder: "sk_...", optional: false, url: "https://video.a2e.ai/account/token", link: "an A2E API token",
  },
  gemini: {
    name: "Gemini", description: "Optional: used for Gemini story planning and/or images.",
    placeholder: "Gemini API key", optional: true, url: "https://aistudio.google.com/api-keys", link: "a Gemini API key",
  },
  openai: {
    name: "OpenAI", description: "Optional: used for GPT story and shot planning. API billing is separate from ChatGPT.",
    placeholder: "sk-...", optional: true, url: "https://platform.openai.com/api-keys", link: "an OpenAI API key",
  },
};

export default function A2EKeySettings({ configured, onSaved, provider = "a2e" }) {
  const info = PROVIDERS[provider];
  const displayName = info.name;
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);

  const handleSave = async (event) => {
    event.preventDefault();
    if (!apiKey.trim() || saving) return;
    setSaving(true);
    setMessage(null);
    try {
      const response = await fetch(`/api/settings/${provider}-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Requested-With": "MicroDrama" },
        body: JSON.stringify({ api_key: apiKey.trim() }),
        cache: "no-store",
      });
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        throw new Error(typeof result.detail === "string" ? result.detail : "Unable to save the key.");
      }
      setApiKey("");
      setShowKey(false);
      setMessage({ type: "success", text: `Key saved for new jobs. It has not been verified with ${displayName} yet.` });
      onSaved();
    } catch (error) {
      setMessage({ type: "error", text: error.message || "Unable to save the key." });
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="mb-5 rounded-2xl p-6" style={{ backgroundColor: "#12121a", border: "1px solid #29243e" }}>
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: "rgba(124, 58, 237, 0.15)", color: "#a78bfa" }}>
            <KeyRound size={17} />
          </div>
          <div>
            <h2 className="text-sm font-semibold" style={{ color: "#e2e8f0" }}>{displayName} API key</h2>
            <p className="text-xs mt-1" style={{ color: "#94a3b8" }}>
              {info.description}
            </p>
          </div>
        </div>
        {configured !== null && (
          <span className="text-xs font-medium rounded-full px-2.5 py-1" style={{
            backgroundColor: configured ? "rgba(34, 197, 94, 0.1)" : "rgba(234, 179, 8, 0.1)",
            color: configured ? "#86efac" : "#fde68a",
          }}>
            {configured ? "Key configured · not verified" : info.optional ? "Optional" : "Key needed"}
          </span>
        )}
      </div>
      <form onSubmit={handleSave}>
        <label htmlFor={`${provider}-api-key`} className="block text-xs font-medium mb-2" style={{ color: "#94a3b8" }}>
          {configured ? `Update ${displayName} API key` : `${displayName} API key`}
        </label>
        <div className="flex flex-col sm:flex-row gap-2">
          <div className="relative flex-1">
            <input
              id={`${provider}-api-key`}
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(event) => { setApiKey(event.target.value); setMessage(null); }}
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              maxLength={2048}
              placeholder={info.placeholder}
              className="w-full rounded-xl pl-4 pr-12 py-3 text-sm outline-none"
              style={{ backgroundColor: "#0d0d14", border: "1px solid #29243e", color: "#e2e8f0" }}
            />
            <button
              type="button"
              onClick={() => setShowKey((visible) => !visible)}
              aria-label={`${showKey ? "Hide" : "Show"} ${displayName} API key`}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1"
              style={{ color: "#94a3b8" }}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
          <button
            type="submit"
            disabled={saving || !apiKey.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold"
            style={{ backgroundColor: "#7c3aed", color: "white", opacity: saving || !apiKey.trim() ? 0.5 : 1 }}
          >
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Check size={15} />}
            {saving ? "Saving..." : "Save key"}
          </button>
        </div>
      </form>
      {message && (
        <p role="status" className="text-xs mt-3" style={{ color: message.type === "error" ? "#fca5a5" : "#86efac" }}>
          {message.text}
        </p>
      )}
      <div className="mt-4 space-y-2 text-xs leading-relaxed" style={{ color: "#64748b" }}>
        <p>Saved unencrypted in the server&apos;s <code>.env</code> file and not shown again. This app has no login; use only on a private, trusted instance.</p>
        <a href={info.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1" style={{ color: "#a78bfa" }}>
          Get {info.link} <ExternalLink size={12} />
        </a>
      </div>
    </section>
  );
}
