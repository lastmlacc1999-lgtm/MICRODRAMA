"use client";

import { useState } from "react";
import { Sparkles, Loader2 } from "lucide-react";

const STYLES = [
  { value: "Cinematic", label: "Cinematic", desc: "Epic film quality" },
  { value: "Realistic", label: "Realistic", desc: "True-to-life" },
  { value: "Anime", label: "Anime", desc: "Japanese animation" },
  { value: "Fantasy", label: "Fantasy", desc: "Magical & fantastical" },
  { value: "Documentary", label: "Documentary", desc: "Journalistic style" },
];

const MODES = [
  {
    value: "idea2video",
    label: "Idea to Video",
    desc: "Plan shots from an idea; optional AI story writing",
  },
  {
    value: "script2video",
    label: "Script to Video",
    desc: "Start from your own scene script",
  },
];

export default function IdeaForm({ onSubmit, isSubmitting, catalog, catalogError, geminiConfigured, openaiConfigured }) {
  const [idea, setIdea] = useState("");
  const [userRequirement, setUserRequirement] = useState("");
  const [style, setStyle] = useState("Cinematic");
  const [mode, setMode] = useState("idea2video");
  const [script, setScript] = useState("");
  const [planningChoice, setPlanningChoice] = useState("");
  const [textModelChoices, setTextModelChoices] = useState({ gemini: "", openai: "" });
  const [imageProvider, setImageProvider] = useState("a2e");
  const [imageChoice, setImageChoice] = useState("");
  const [videoChoice, setVideoChoice] = useState("");
  const [durationChoice, setDurationChoice] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);

  const planningProvider = planningChoice || catalog?.defaults?.planning_provider || "simple";
  const selectableTextModel = planningProvider === "gemini" || planningProvider === "openai";
  const textModels = (planningProvider === "openai" ? catalog?.openai_text_models : catalog?.gemini_text_models) || [];
  const defaultTextModel = planningProvider === "openai" ? catalog?.defaults?.openai_text : catalog?.defaults?.gemini_text;
  const selectedTextModel = selectableTextModel
    ? textModels.find((item) => item.id === (textModelChoices[planningProvider] || defaultTextModel)) : null;
  const images = (catalog?.image_models || []).filter((item) => item.provider === imageProvider);
  const videoModels = catalog?.video_models || [];
  const defaultImage = imageProvider === "gemini" ? catalog?.defaults?.gemini_image : catalog?.defaults?.a2e_image;
  const selectedImage = images.find((item) => item.id === (imageChoice || defaultImage));
  const selectedVideo = videoModels.find((item) => item.id === (videoChoice || catalog?.defaults?.video));
  const selectedDuration = selectedVideo?.durations.includes(Number(durationChoice)) && durationChoice !== ""
    ? Number(durationChoice) : selectedVideo?.default_duration;
  const groups = (items) => [...new Set(items.map((item) => item.group))];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!acknowledged || !selectedImage || !selectedVideo ||
        (selectableTextModel && !selectedTextModel) ||
        (mode === "idea2video" ? !idea.trim() : !script.trim())) return;
    onSubmit({
      idea: idea.trim(),
      user_requirement: userRequirement.trim(),
      style,
      mode,
      script: script.trim(),
      planning_provider: planningProvider,
      planning_model: selectableTextModel ? selectedTextModel.id : null,
      image_provider: imageProvider,
      image_model: selectedImage.id,
      video_model: selectedVideo.id,
      clip_duration: selectedDuration ?? null,
    });
  };

  const inputStyle = {
    backgroundColor: "#12121a",
    border: "1px solid #22223a",
    borderRadius: "12px",
    color: "#e2e8f0",
    padding: "12px 16px",
    width: "100%",
    fontSize: "14px",
    transition: "border-color 0.2s",
    outline: "none",
    resize: "vertical",
  };

  const labelStyle = {
    display: "block",
    fontSize: "13px",
    fontWeight: "500",
    marginBottom: "8px",
    color: "#94a3b8",
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl p-6 space-y-6"
      style={{
        backgroundColor: "#12121a",
        border: "1px solid #1a1a26",
      }}
    >
      {/* Mode selector */}
      <div>
        <label style={labelStyle}>Generation Mode</label>
        <div className="grid grid-cols-2 gap-3">
          {MODES.map((m) => (
            <button
              key={m.value}
              type="button"
              aria-pressed={mode === m.value}
              onClick={() => setMode(m.value)}
              className="text-left p-4 rounded-xl transition-all"
              style={{
                backgroundColor: mode === m.value ? "rgba(124, 58, 237, 0.15)" : "#1a1a26",
                border: mode === m.value
                  ? "1px solid rgba(124, 58, 237, 0.5)"
                  : "1px solid #22223a",
                cursor: "pointer",
              }}
            >
              <div
                className="text-sm font-semibold mb-1"
                style={{ color: mode === m.value ? "#a78bfa" : "#e2e8f0" }}
              >
                {m.label}
              </div>
              <div className="text-xs" style={{ color: "#6b7280" }}>
                {m.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Idea input */}
      <div>
        <label htmlFor="idea-text" style={labelStyle}>
          {mode === "script2video" ? "Brief idea or title" : "Your Idea *"}
        </label>
        <textarea
          id="idea-text"
          value={idea}
          maxLength={3000}
          onChange={(e) => setIdea(e.target.value)}
          placeholder={
            mode === "idea2video"
              ? "e.g. A lone astronaut discovers an ancient alien structure on Mars..."
              : "Brief title or concept for your video"
          }
          rows={3}
          required={mode === "idea2video"}
          style={inputStyle}
          onFocus={(e) => (e.target.style.borderColor = "#7c3aed")}
          onBlur={(e) => (e.target.style.borderColor = "#22223a")}
        />
      </div>

      {/* Script input (script2video only) */}
      {mode === "script2video" && (
        <div>
          <label htmlFor="scene-script" style={labelStyle}>Scene Script *</label>
          <textarea
            id="scene-script"
            value={script}
            maxLength={8000}
            onChange={(e) => setScript(e.target.value)}
            placeholder="Write your scene script here. Be descriptive about setting, characters, actions, and mood..."
            rows={6}
            required
            style={inputStyle}
            onFocus={(e) => (e.target.style.borderColor = "#7c3aed")}
            onBlur={(e) => (e.target.style.borderColor = "#22223a")}
          />
        </div>
      )}

      {/* Style selector */}
      <div>
        <label style={labelStyle}>Visual Style</label>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {STYLES.map((s) => (
            <button
              key={s.value}
              type="button"
              aria-pressed={style === s.value}
              onClick={() => setStyle(s.value)}
              className="p-3 rounded-xl text-center transition-all"
              style={{
                backgroundColor: style === s.value ? "rgba(124, 58, 237, 0.15)" : "#1a1a26",
                border: style === s.value
                  ? "1px solid rgba(124, 58, 237, 0.5)"
                  : "1px solid #22223a",
                cursor: "pointer",
              }}
            >
              <div
                className="text-xs font-semibold mb-1"
                style={{ color: style === s.value ? "#a78bfa" : "#e2e8f0" }}
              >
                {s.label}
              </div>
              <div className="text-xs" style={{ color: "#94a3b8", fontSize: "11px" }}>
                {s.desc}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Per-job planning, image and video selections are backed by the server catalog. */}
      <section className="rounded-xl p-4 space-y-4" style={{ backgroundColor: "#0d0d14", border: "1px solid #29243e" }} aria-labelledby="models-heading">
        <div>
          <h2 id="models-heading" className="text-sm font-semibold" style={{ color: "#e2e8f0" }}>Generation models</h2>
          <p className="text-xs mt-1 leading-relaxed" style={{ color: "#94a3b8" }}>
            Pick separately for this job. This list is not an access check; provider availability and prices vary.
          </p>
        </div>
        {catalogError && <p role="alert" className="text-xs" style={{ color: "#fca5a5" }}>Could not load model options. Check that the backend is running, then refresh.</p>}
        {!catalog && !catalogError && <p className="text-xs" style={{ color: "#94a3b8" }}>Loading model options…</p>}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="planning-provider" style={labelStyle}>Story &amp; shot planning</label>
            <select id="planning-provider" value={planningProvider} disabled={!catalog}
              onChange={(e) => { setPlanningChoice(e.target.value); setAcknowledged(false); }} style={inputStyle}>
              <option value="simple">Simple · use my text directly</option>
              <option value="gemini">Google Gemini API</option>
              <option value="openai">OpenAI GPT API</option>
              <option value="configured" disabled={!catalog?.configured_llm_available}>
                OpenAI-compatible LLM{catalog?.configured_llm_available ? " · server configured" : " · not configured"}
              </option>
            </select>
          </div>
          {selectableTextModel ? (
            <div>
              <label htmlFor="planning-model" style={labelStyle}>{planningProvider === "openai" ? "OpenAI GPT" : "Gemini"} text model</label>
              <select id="planning-model" value={selectedTextModel?.id || ""} disabled={!catalog}
                onChange={(e) => { const value = e.target.value; setTextModelChoices((choices) => ({ ...choices, [planningProvider]: value })); setAcknowledged(false); }} style={inputStyle}>
                {!selectedTextModel && <option value="">Loading…</option>}
                {textModels.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
              </select>
            </div>
          ) : (
            <p className="text-xs leading-relaxed self-center" style={{ color: "#94a3b8" }}>
              {planningProvider === "configured"
                ? "Uses the OpenAI-compatible text model configured on this server."
                : "No text-model calls: the app makes a simple plan from the text you provide."}
            </p>
          )}
          <div>
            <label htmlFor="image-provider" style={labelStyle}>Image provider</label>
            <select id="image-provider" value={imageProvider} disabled={!catalog}
              onChange={(e) => { setImageProvider(e.target.value); setImageChoice(""); setAcknowledged(false); }} style={inputStyle}>
              <option value="a2e">A2E API</option>
              <option value="gemini">Google Gemini API</option>
            </select>
          </div>
          <div>
            <label htmlFor="image-model" style={labelStyle}>Image model</label>
            <select id="image-model" value={selectedImage?.id || ""} disabled={!catalog}
              onChange={(e) => { setImageChoice(e.target.value); setAcknowledged(false); }} style={inputStyle}>
              {!selectedImage && <option value="">Loading…</option>}
              {groups(images).map((group) => (
                <optgroup label={group} key={group}>
                  {images.filter((item) => item.group === group).map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
                </optgroup>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="video-model" style={labelStyle}>A2E video model</label>
            <select id="video-model" value={selectedVideo?.id || ""} disabled={!catalog}
              onChange={(e) => { setVideoChoice(e.target.value); setDurationChoice(""); setAcknowledged(false); }} style={inputStyle}>
              {!selectedVideo && <option value="">Loading…</option>}
              {groups(videoModels).map((group) => (
                <optgroup label={group} key={group}>
                  {videoModels.filter((item) => item.group === group).map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
                </optgroup>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="clip-duration" style={labelStyle}>Duration per clip</label>
            <select id="clip-duration" value={selectedDuration ?? ""} disabled={!selectedVideo || selectedVideo.durations.length === 0}
              onChange={(e) => { setDurationChoice(e.target.value); setAcknowledged(false); }} style={inputStyle}>
              {selectedVideo?.durations.length ? selectedVideo.durations.map((seconds) => (
                <option value={seconds} key={seconds}>{seconds} seconds</option>
              )) : <option value="">Provider-determined</option>}
            </select>
          </div>
        </div>
        <div className="space-y-1 text-xs leading-relaxed" style={{ color: "#94a3b8" }}>
          {selectableTextModel && (
            <>
              <p>{planningProvider === "openai" ? "OpenAI GPT" : "Gemini"} {mode === "idea2video" ? "develops the story, scenes, characters and shots" : "extracts characters and plans shots from your script; it does not rewrite the script"}. This is separate from your image choice.</p>
              {selectedTextModel?.note && <p><strong style={{ color: "#cbd5e1" }}>Text:</strong> {selectedTextModel.note}</p>}
            </>
          )}
          {(planningProvider === "gemini" || imageProvider === "gemini") && geminiConfigured === false &&
            <p style={{ color: "#fde68a" }}>Save a Gemini API key above before starting a Gemini job.</p>}
          {planningProvider === "openai" && openaiConfigured === false &&
            <p style={{ color: "#fde68a" }}>Save an OpenAI API key above before starting a GPT planning job.</p>}
          <p>One clip is generated per shot; a job can contain several shots. Image references encourage continuity but cannot guarantee it.</p>
          <p>Resolution and other provider settings are preset; available quality variants appear as separate model choices.</p>
          {selectedImage?.note && <p><strong style={{ color: "#cbd5e1" }}>Image:</strong> {selectedImage.note}</p>}
          {selectedVideo?.note && <p><strong style={{ color: "#cbd5e1" }}>Video:</strong> {selectedVideo.note}</p>}
          <p>When supported, non-native video audio is requested off. Some models may produce audio anyway; A2E native audio follows the server setting.</p>
        </div>
      </section>

      {/* Additional requirements */}
      <div>
        <label htmlFor="user-requirements" style={labelStyle}>
          Additional Requirements{" "}
          <span style={{ color: "#4b5563", fontWeight: 400 }}>(optional)</span>
        </label>
        <textarea
          id="user-requirements"
          value={userRequirement}
          maxLength={2000}
          onChange={(e) => setUserRequirement(e.target.value)}
          placeholder="Any specific instructions, mood, pacing, color palette, character details..."
          rows={2}
          style={inputStyle}
          onFocus={(e) => (e.target.style.borderColor = "#7c3aed")}
          onBlur={(e) => (e.target.style.borderColor = "#22223a")}
        />
      </div>

      {/* A single job can trigger multiple separately billed provider requests. */}
      <div className="rounded-xl px-4 py-4 text-sm" style={{ backgroundColor: "rgba(234, 179, 8, 0.06)", border: "1px solid rgba(234, 179, 8, 0.2)", color: "#cbd5e1" }}>
        <p className="mb-2 leading-relaxed">
          {imageProvider === "gemini"
            ? "Gemini may charge for images. A2E charges for videos and for issuing each frame-upload URL, even if an upload fails. Frames are uploaded to A2E storage."
            : "A2E charges credits for image and video tasks."}{" "}
          {planningProvider === "gemini" ? "Gemini may also charge separately for text planning."
            : planningProvider === "openai" ? "OpenAI may also charge separately for GPT text planning; API billing is separate from ChatGPT."
            : planningProvider === "configured" ? "Your configured text LLM may charge separately."
            : "Simple planning makes no text-model requests."}
        </p>
        <p className="mb-3 leading-relaxed">
          A2E estimates are checked before submissions, not guaranteed final prices; its job limit does not cover Gemini, OpenAI or other providers.
          Completed work is saved, but earlier requests may be billed if a job fails. A manual resume can charge for unfinished steps or re-uploading inaccessible images.
        </p>
        <label className="flex items-start gap-2 cursor-pointer">
          <input type="checkbox" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} className="mt-1" />
          <span>I understand that starting this job may incur provider charges.</span>
        </label>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isSubmitting || !acknowledged || !selectedImage || !selectedVideo ||
          (selectableTextModel && !selectedTextModel) || (mode === "idea2video" ? !idea.trim() : !script.trim())}
        className="w-full py-4 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all"
        style={{
          background: isSubmitting
            ? "#3b3b5c"
            : "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
          color: isSubmitting ? "#6b7280" : "white",
          cursor: isSubmitting || !acknowledged || !selectedImage || !selectedVideo ||
            (selectableTextModel && !selectedTextModel) ? "not-allowed" : "pointer",
          opacity: isSubmitting || !acknowledged || !selectedImage || !selectedVideo ||
            (selectableTextModel && !selectedTextModel) || (mode === "idea2video" ? !idea.trim() : !script.trim()) ? 0.55 : 1,
          boxShadow: isSubmitting ? "none" : "0 4px 20px rgba(124, 58, 237, 0.4)",
        }}
      >
        {isSubmitting ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            Starting pipeline...
          </>
        ) : (
          <>
            <Sparkles size={16} />
            Generate Video
          </>
        )}
      </button>
    </form>
  );
}
