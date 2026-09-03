import { useState } from "react";
import { Download } from "lucide-react";
import { LOGS } from "../data";

const DEVICE_KEYWORDS = ["battery","sensor","calibr","connection","temperature","humidity","heartbeat","operator","handover","environmental","noise","firmware"];
const LC: Record<string, string> = { ALERT:"#B3262E", WARNING:"#D99A27", SUCCESS:"#28734A", INFO:"#C49A4A" };

export default function SystemLogs({ theme }: { theme: "dark" | "light" }) {
  const dark  = theme === "dark";
  const [f, setF] = useState("ALL");

  const bg   = dark ? "bg-obsidian"    : "bg-[#F1EDE3]";
  const cBg  = dark ? "bg-gunmetal"    : "bg-white";
  const text = dark ? "text-ivory"     : "text-obsidian";
  const muted= dark ? "text-warm-grey" : "text-[#6F6A61]";
  const bdr  = dark ? "border-warm-grey/10" : "border-obsidian/8";

  const filtered =
    f === "ALERTS"  ? LOGS.filter(l => l.level === "ALERT") :
    f === "DEVICES" ? LOGS.filter(l => DEVICE_KEYWORDS.some(k => l.event.toLowerCase().includes(k))) :
    LOGS;

  function exportLogs() {
    const header = "TIMESTAMP,DEVICE,EVENT,LOCATION,STATUS\n";
    const rows   = filtered.map(l => `${l.timestamp},${l.device},"${l.event}",${l.location},${l.level}`).join("\n");
    const blob   = new Blob([header + rows], { type: "text/csv" });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement("a");
    a.href = url; a.download = `sentry-logs-${Date.now()}.csv`; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={`min-h-full ${bg}`}>
      <div className="p-4 md:p-6 space-y-4 pb-20 md:pb-6">

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div>
            <div className={`font-display text-[40px] md:text-[50px] tracking-widest leading-none ${text}`}>SYSTEM LOGS</div>
            <div className={`font-mono text-[9.5px] tracking-widest uppercase mt-1 ${muted}`}>Device activity, sensor health and security events.</div>
          </div>
        </div>

        {/* Summary tiles */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
          {[
            { l: "TOTAL ENTRIES",  v: LOGS.length,                                   c: "#C49A4A" },
            { l: "ALERTS",         v: LOGS.filter(l => l.level === "ALERT").length,   c: "#B3262E" },
            { l: "WARNINGS",       v: LOGS.filter(l => l.level === "WARNING").length, c: "#D99A27" },
            { l: "SUCCESSFUL OPS", v: LOGS.filter(l => l.level === "SUCCESS").length, c: "#28734A" },
          ].map(s => (
            <div key={s.l} className={`panel ${cBg} p-3`}>
              <div className={`font-mono text-[8px] tracking-widest ${muted} mb-1`}>{s.l}</div>
              <div className="font-display text-[32px] tracking-widest" style={{ color: s.c }}>{s.v}</div>
            </div>
          ))}
        </div>

        {/* Log panel — identical style to dashboard */}
        <div className={`panel ${cBg}`}>
          <div className="h-[1px] bg-brass/25" />
          <div className={`px-4 py-3 border-b ${bdr}`}>
            <div className={`font-heading text-[12px] tracking-[.14em] font-semibold uppercase ${text}`}>SYSTEM LOGS</div>
            <div className={`font-mono text-[9px] tracking-widest ${muted} mt-0.5`}>Monitor device connectivity, sensor health and system activity.</div>
          </div>
          {/* Filters + export */}
          <div className={`px-4 py-2.5 border-b ${bdr} flex flex-wrap items-center gap-2`}>
            <div className="flex gap-1 flex-wrap">
              {["ALL", "DEVICES", "ALERTS"].map(fi => (
                <button key={fi} onClick={() => setF(fi)}
                  className={`font-mono text-[8.5px] tracking-widest px-2 py-1 border transition-all
                    ${f === fi ? "bg-signal-red text-ivory border-signal-red" : `${bdr} ${muted} hover:border-brass`}`}>
                  {fi}
                </button>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <span className={`font-mono text-[8px] tracking-widest ${muted}`}>{filtered.length} ENTRIES</span>
              <button onClick={exportLogs}
                className={`flex items-center gap-1.5 px-2.5 py-1 border ${bdr} font-mono text-[8.5px] tracking-widest transition-all ${muted} hover:border-brass hover:text-brass`}>
                <Download className="w-3 h-3" /> EXPORT CSV
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className={`border-b ${bdr} ${dark ? "bg-charcoal/50" : "bg-[#F4F0E8]"}`}>
                  {["TIMESTAMP", "DEVICE", "EVENT", "LOCATION", "STATUS"].map(h => (
                    <th key={h} className={`px-4 py-2.5 font-mono text-[8.5px] tracking-[.2em] uppercase ${muted}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((l, i) => {
                  const c = LC[l.level] || "#A8A39A";
                  return (
                    <tr key={i} className={`border-b ${bdr} transition-colors hover:bg-signal-red/5`}
                      style={{ background: i % 2 === 0 ? (dark ? "rgba(29,32,35,.25)" : "rgba(244,240,234,.35)") : undefined }}>
                      <td className="px-4 py-2.5 font-mono text-[9.5px] text-brass whitespace-nowrap">{l.timestamp}</td>
                      <td className="px-4 py-2.5 font-mono text-[9.5px] text-brass whitespace-nowrap">{l.device}</td>
                      <td className={`px-4 py-2.5 font-mono text-[9.5px] ${text} max-w-[240px] truncate`}>{l.event}</td>
                      <td className={`px-4 py-2.5 font-mono text-[9.5px] ${muted} whitespace-nowrap`}>{l.location}</td>
                      <td className="px-4 py-2.5 whitespace-nowrap">
                        <span className="font-mono text-[8.5px] tracking-widest px-1.5 py-[2px]"
                          style={{ background: `${c}20`, color: c, border: `1px solid ${c}40` }}>
                          {l.level}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className={`px-4 py-2.5 border-t ${bdr} flex items-center justify-between`}>
            <div className={`font-mono text-[8.5px] tracking-widest ${muted}`}>SHOWING {filtered.length} OF {LOGS.length} ENTRIES</div>
            <div className="font-mono text-[8.5px] tracking-widest text-brass">LAST REFRESH: {new Date().toLocaleTimeString("en-IN", { hour12: false })}</div>
          </div>
        </div>

        <div className={`font-mono text-[8.5px] tracking-widest ${muted} text-center`}>
          SENTRY OS // AUDIT LOG — ALL ENTRIES CRYPTOGRAPHICALLY SIGNED // TAMPER-EVIDENT RECORD
        </div>
      </div>
    </div>
  );
}
