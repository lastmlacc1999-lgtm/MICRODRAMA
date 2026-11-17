"use client";

import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import PipelineProgress from "../../../components/PipelineProgress";
import VideoResult from "../../../components/VideoResult";
import { ArrowLeft, RotateCcw } from "lucide-react";
import Link from "next/link";

function SavedWork({ project }) {
  if (!project) return null;
  const steps = project.completed_steps || [];
  const assets = project.assets || [];
  const tasks = project.a2e_tasks || [];
  return (
    <div className="mt-4 space-y-3 text-sm" style={{ color: "#cbd5e1" }}>
      <p>{steps.length} completed {steps.length === 1 ? "step" : "steps"} saved · A2E estimated total: {project.estimated_a2e_credits} credits</p>
      {steps.length > 0 && (
        <details className="text-xs leading-relaxed" style={{ color: "#94a3b8" }}>
          <summary className="cursor-pointer" style={{ color: "#c4b5fd" }}>View saved steps</summary>
          <ul className="mt-2 space-y-1 list-disc pl-5">{steps.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}</ul>
        </details>
      )}
      {tasks.length > 0 && (
        <details className="text-xs leading-relaxed" style={{ color: "#94a3b8" }}>
          <summary className="cursor-pointer" style={{ color: "#c4b5fd" }}>View saved A2E task IDs</summary>
          <ul className="mt-2 space-y-1">{tasks.map((task, index) => (
            <li key={`${task.task_id}-${index}`} className="break-all">{task.label}: {task.task_id} ({task.state})</li>
          ))}</ul>
        </details>
      )}
      {assets.length > 0 && (
        <div>
          <p className="text-xs font-semibold mb-2" style={{ color: "#cbd5e1" }}>Saved images and clips</p>
          <div className="flex flex-wrap gap-2">
            {assets.map((asset) => (
              <a key={asset.url} href={asset.url} target="_blank" rel="noopener noreferrer"
                className="rounded-lg px-3 py-2 text-xs break-words" style={{ backgroundColor: "#1a1a26", border: "1px solid #35304c", color: "#c4b5fd" }}>
                {asset.label} ↗
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function GeneratePage() {
  const { jobId } = useParams();
  const [status, setStatus] = useState("running");
  const [videoUrl, setVideoUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentMessage, setCurrentMessage] = useState("Starting pipeline...");
  const [logs, setLogs] = useState([]);
  const [project, setProject] = useState(null);
  const [projectUnavailable, setProjectUnavailable] = useState(null);
  const [resumeKey, setResumeKey] = useState(0);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState(null);
  const [acknowledgeCosts, setAcknowledgeCosts] = useState(false);
  const [confirmUncertain, setConfirmUncertain] = useState(false);

  const loadProject = useCallback(async () => {
    if (!jobId) return;
    try {
      const response = await fetch(`/api/projects/${jobId}`, { cache: "no-store" });
      if (response.status === 404) {
        setProject(null);
        setProjectUnavailable("missing");
        return;
      }
      if (!response.ok) throw new Error("Project unavailable");
      setProject(await response.json());
      setProjectUnavailable(null);
    } catch {
      setProject(null);
      setProjectUnavailable("connection");
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/status/${jobId}`);
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        setLogs((prev) => [...prev, {
          time: new Date().toLocaleTimeString(),
          message: event.message || event.stage || "",
          type: event.type,
        }]);
        if (event.type === "progress") {
          setProgress(event.progress ?? 0);
          setCurrentMessage(event.message || "");
        } else if (event.type === "complete") {
          setProgress(100);
          setStatus("completed");
          setVideoUrl(event.video_url);
          setCurrentMessage("Video generation complete!");
          loadProject();
          es.close();
        } else if (event.type === "error") {
          setStatus("failed");
          setErrorMsg(event.message || "Generation failed");
          setCurrentMessage("Generation paused");
          loadProject();
          es.close();
        }
      } catch (error) {
        console.error("Failed to parse SSE event:", error);
      }
    };
    es.onerror = () => {
      // EventSource reconnects automatically while a job runs.
      fetch(`/api/result/${jobId}`, { cache: "no-store" })
        .then(async (response) => {
          if (response.status === 404) {
            setStatus("failed");
            setErrorMsg("This project could not be found on this server.");
            loadProject();
            es.close();
            return;
          }
          if (!response.ok) return;
          const data = await response.json();
          if (data.status === "completed") {
            setStatus("completed");
            setVideoUrl(data.video_url);
            setProgress(100);
            loadProject();
            es.close();
          } else if (data.status === "failed" || data.status === "interrupted") {
            setStatus("failed");
            setErrorMsg(data.error || "Generation was interrupted.");
            setCurrentMessage("Generation paused");
            loadProject();
            es.close();
          }
        }).catch(() => {});
    };
    return () => es.close();
  }, [jobId, resumeKey, loadProject]);

  const resume = async () => {
    if (!project || !acknowledgeCosts || (project.uncertain_steps?.length > 0 && !confirmUncertain) || resuming) return;
    setResuming(true);
    setResumeError(null);
    try {
      const response = await fetch(`/api/projects/${jobId}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Requested-With": "MicroDrama" },
        body: JSON.stringify({ acknowledge_costs: true, retry_uncertain: confirmUncertain }),
        cache: "no-store",
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(typeof data.detail === "string" ? data.detail : "Could not resume the project.");
      }
      setStatus("running");
      setErrorMsg(null);
      setVideoUrl(null);
      setLogs([]);
      setCurrentMessage("Resuming from saved work...");
      setProject(null);
      setAcknowledgeCosts(false);
      setConfirmUncertain(false);
      setResumeKey((value) => value + 1);
    } catch (error) {
      setResumeError(error.message || "Could not resume the project.");
      loadProject();
    } finally {
      setResuming(false);
    }
  };

  const uncertain = project?.uncertain_steps || [];
  return (
    <main className="min-h-screen" style={{ backgroundColor: "#0a0a0f" }}>
      <div className="sticky top-0 z-10 px-6 py-4 flex items-center gap-4" style={{
        backgroundColor: "rgba(10, 10, 15, 0.9)", backdropFilter: "blur(12px)", borderBottom: "1px solid #1a1a26",
      }}>
        <Link href="/" className="flex shrink-0 items-center gap-2 text-sm" style={{ color: "#9ca3af" }}>
          <ArrowLeft size={16} /> Projects
        </Link>
        <div className="h-4 w-px" style={{ backgroundColor: "#22223a" }} />
        <span className="min-w-0 flex-1 truncate text-sm" title={`Job: ${jobId}`} style={{ color: "#64748b" }}>
          Job: <span style={{ color: "#a78bfa", fontFamily: "monospace" }}>{jobId}</span>
        </span>
        <span className="shrink-0 rounded-full px-3 py-1 text-xs font-medium" style={{
          backgroundColor: status === "running" ? "rgba(234,179,8,.15)" : status === "completed" ? "rgba(34,197,94,.15)" : "rgba(239,68,68,.15)",
          color: status === "running" ? "#fde047" : status === "completed" ? "#86efac" : "#fca5a5",
        }}>{status === "running" ? "Generating" : status === "completed" ? "Complete" : "Paused"}</span>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-10">
        {status === "completed" && videoUrl && (
          <div className="mb-10 animate-slide-up"><VideoResult videoUrl={videoUrl} jobId={jobId} /></div>
        )}
        {status === "completed" && project && (
          <section className="mb-10 rounded-2xl p-5" style={{ backgroundColor: "#12121a", border: "1px solid #29243e" }}>
            <h2 className="text-sm font-semibold" style={{ color: "#e2e8f0" }}>Saved project work</h2>
            <SavedWork project={project} />
          </section>
        )}
        {status === "failed" && (
          <section className="mb-10 p-5 rounded-2xl" style={{ backgroundColor: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.25)" }}>
            <h2 className="text-sm font-semibold" style={{ color: "#fca5a5" }}>Generation paused</h2>
            <p className="text-sm mt-1 leading-relaxed" style={{ color: "#cbd5e1" }}>{errorMsg}</p>
            {project ? (
              <>
                <p className="text-sm mt-3 leading-relaxed" style={{ color: "#cbd5e1" }}>
                  Your completed work is saved. Resume uses saved steps and checks known A2E tasks instead of submitting them again.
                </p>
                <SavedWork project={project} />
                {uncertain.length > 0 && (
                  <div className="mt-4 rounded-xl p-3 text-xs leading-relaxed" style={{ backgroundColor: "rgba(234,179,8,.08)", border: "1px solid rgba(234,179,8,.25)", color: "#fde68a" }}>
                    <p className="font-semibold mb-1">A request may already have been billed</p>
                    <p>Its outcome is uncertain, its output is missing, or the provider reported a failed task. Check the provider&apos;s task and billing history before trying it again:</p>
                    <ul className="list-disc pl-5 mt-2">{uncertain.map((item, index) => (
                      <li key={`${item.label}-${index}`} className="break-words">{item.label}{item.task_id ? ` — task ${item.task_id}` : ""}</li>
                    ))}</ul>
                    <label className="flex items-start gap-2 mt-3 cursor-pointer">
                      <input type="checkbox" checked={confirmUncertain} onChange={(event) => setConfirmUncertain(event.target.checked)} className="mt-0.5" />
                      <span>I checked the provider history and accept that retrying these requests may be charged again.</span>
                    </label>
                  </div>
                )}
                <label className="flex items-start gap-2 mt-4 text-xs leading-relaxed cursor-pointer" style={{ color: "#cbd5e1" }}>
                  <input type="checkbox" checked={acknowledgeCosts} onChange={(event) => setAcknowledgeCosts(event.target.checked)} className="mt-0.5" />
                  <span>I understand unfinished steps and re-uploads of expired or inaccessible images may incur new charges.</span>
                </label>
                {resumeError && <p role="alert" className="text-xs mt-3" style={{ color: "#fca5a5" }}>{resumeError}</p>}
                <button type="button" onClick={resume}
                  disabled={resuming || !acknowledgeCosts || (uncertain.length > 0 && !confirmUncertain)}
                  className="mt-4 inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold"
                  style={{ backgroundColor: "#7c3aed", color: "white", opacity: resuming || !acknowledgeCosts || (uncertain.length > 0 && !confirmUncertain) ? 0.5 : 1 }}>
                  <RotateCcw size={15} /> {resuming ? "Resuming..." : "Resume from saved work"}
                </button>
                <p className="text-xs mt-3 leading-relaxed" style={{ color: "#94a3b8" }}>
                  Nothing is retried automatically. If you need to update a key, return to Projects; the saved job will remain here.
                </p>
              </>
            ) : projectUnavailable ? (
              <p className="text-xs mt-4 leading-relaxed" style={{ color: "#cbd5e1" }}>
                {projectUnavailable === "missing"
                  ? "This server could not find or read the saved project. Check the original server and storage; older jobs may not have checkpoints. Review provider task history before starting again."
                  : "Could not load saved work right now. Check the backend connection and try again; do not assume your project is gone."}
              </p>
            ) : <p className="text-xs mt-3" style={{ color: "#94a3b8" }}>Checking for saved work…</p>}
          </section>
        )}
        <PipelineProgress progress={progress} currentMessage={currentMessage} status={status} logs={logs} />
      </div>
    </main>
  );
}
