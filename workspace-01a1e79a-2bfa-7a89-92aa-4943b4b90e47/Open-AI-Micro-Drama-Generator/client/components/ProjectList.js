"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { History, ArrowRight } from "lucide-react";

const STATUS = {
  running: "Generating",
  completed: "Complete",
  failed: "Needs attention",
  interrupted: "Interrupted",
  unavailable: "Unavailable",
};

export default function ProjectList() {
  const [projects, setProjects] = useState(null);
  const [failed, setFailed] = useState(false);
  const [visibleCount, setVisibleCount] = useState(20);

  useEffect(() => {
    let active = true;
    fetch("/api/projects", { cache: "no-store" })
      .then((res) => { if (!res.ok) throw new Error("Projects unavailable"); return res.json(); })
      .then((data) => { if (active) setProjects(data.projects || []); })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, []);

  return (
    <section className="mb-5 rounded-2xl p-6" style={{ backgroundColor: "#12121a", border: "1px solid #29243e" }} aria-labelledby="saved-projects-heading">
      <div className="flex items-center gap-2 mb-2">
        <History size={16} style={{ color: "#a78bfa" }} />
        <h2 id="saved-projects-heading" className="text-sm font-semibold" style={{ color: "#e2e8f0" }}>Saved projects</h2>
      </div>
      <p className="text-xs leading-relaxed mb-3" style={{ color: "#94a3b8" }}>
        Completed work is saved on this server. Failed or interrupted projects can be reviewed and resumed.
      </p>
      {failed && <p className="text-xs" style={{ color: "#fca5a5" }}>Could not load saved projects. Check the backend connection.</p>}
      {!projects && !failed && <p className="text-xs" style={{ color: "#64748b" }}>Loading projects…</p>}
      {projects?.length === 0 && <p className="text-xs" style={{ color: "#64748b" }}>New projects will appear here.</p>}
      {projects?.length > 0 && (
        <ul className="space-y-2">
          {projects.slice(0, visibleCount).map((project) => (
            <li key={project.job_id}>
              {project.status === "unavailable" ? (
                <div className="rounded-xl px-3 py-3 text-xs" style={{ backgroundColor: "#0d0d14", color: "#fca5a5" }}>
                  Project {project.job_id} could not be read.
                </div>
              ) : (
                <Link href={`/generate/${project.job_id}`} className="flex items-center gap-3 rounded-xl px-3 py-3"
                  style={{ backgroundColor: "#0d0d14", border: "1px solid #29243e" }}>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium" style={{ color: "#e2e8f0" }}>{project.title}</p>
                    <p className="text-xs mt-1" style={{ color: "#94a3b8" }}>
                      {STATUS[project.status] || project.status} · {project.completed_steps.length} saved steps
                    </p>
                  </div>
                  <ArrowRight size={16} className="shrink-0" style={{ color: "#a78bfa" }} />
                </Link>
              )}
            </li>
          ))}
        </ul>
      )}
      {projects?.length > visibleCount && (
        <button type="button" onClick={() => setVisibleCount((value) => value + 20)}
          className="text-xs mt-3 rounded-lg px-3 py-2" style={{ color: "#c4b5fd", border: "1px solid #35304c" }}>
          Show more projects ({projects.length - visibleCount} remaining)
        </button>
      )}
    </section>
  );
}
