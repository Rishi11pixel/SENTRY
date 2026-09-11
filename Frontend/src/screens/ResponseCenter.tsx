import { useState } from "react";
import type { Incident } from "../data";
import ThreatIcon from "../components/ThreatIcon";

type IncStatus = "LIVE" | "RECOGNISED" | "RESOLVED";

const STATUS_MAP: Record<string, IncStatus> = {
  "RESPONSE DISPATCHED": "LIVE",
  "TEAM NOTIFIED":       "RECOGNISED",
  "RESOLVED":            "RESOLVED",
  "ACKNOWLEDGED":        "RECOGNISED",
};

const STATUS_COLOR: Record<IncStatus, string> = {
  LIVE:       "#B3262E",
  RECOGNISED: "#D99A27",
  RESOLVED:   "#20C878",
};

const GLOW: Record<IncStatus, string> = {
  LIVE:       "0 0 10px #B3262E99, 0 0 24px #B3262E44",
  RECOGNISED: "0 0 10px #D99A2799, 0 0 24px #D99A2744",
  RESOLVED:   "0 0 10px #20C87899, 0 0 24px #20C87844",
};

export default function ThreatHistory({ theme, incidents }: { theme: "dark" | "light"; incidents: Incident[] }) {
  const dark  = theme === "dark";
  const bg    = dark ? "bg-obsidian"   : "bg-[#F1EDE3]";
  const cBg   = dark ? "bg-gunmetal"   : "bg-white";
  const text  = dark ? "text-ivory"    : "text-obsidian";
  const muted = dark ? "text-warm-grey": "text-[#6F6A61]";
  const bdr   = dark ? "border-warm-grey/10" : "border-obsidian/8";

  const [filter, setFilter] = useState<"ALL" | IncStatus>("ALL");

  const tc = (t: string) => t.includes("EXPLOSIVE") || t.includes("NARCOTIC") ? "#B3262E" : "#D99A27";

  const items = incidents.map(inc => ({
    ...inc,
    derivedStatus: STATUS_MAP[inc.status] ?? "LIVE" as IncStatus,
  }));

  const shown = filter === "ALL" ? items : items.filter(i => i.derivedStatus === filter);

  const counts = {
    LIVE:       items.filter(i => i.derivedStatus === "LIVE").length,
    RECOGNISED: items.filter(i => i.derivedStatus === "RECOGNISED").length,
    RESOLVED:   items.filter(i => i.derivedStatus === "RESOLVED").length,
  };

  return (
    <div className={`min-h-full ${bg}`}>
      <div className="p-4 md:p-6 space-y-4 pb-20 md:pb-6">

        {/* Header */}
        <div>
          <div className={`font-display text-[40px] md:text-[50px] tracking-widest leading-none ${text}`}>THREAT HISTORY</div>
          <div className={`font-mono text-[9.5px] tracking-widest uppercase mt-1 ${muted}`}>Incident log and status tracking.</div>
          <div className="h-[1px] bg-signal-red/25 mt-4" />
        </div>

        {/* Summary — only ACTIVE + RESOLVED */}
        <div className="grid grid-cols-2 gap-3">
          {([
            { l: "ACTIVE INCIDENTS", v: counts.LIVE,     c: "#B3262E", glow: GLOW.LIVE     },
            { l: "RESOLVED TODAY",   v: counts.RESOLVED,  c: "#20C878", glow: GLOW.RESOLVED },
          ] as const).map(s => (
            <div key={s.l} className={`panel ${cBg} p-4 flex flex-col gap-1`}
              style={{ boxShadow: s.v > 0 ? s.glow : undefined }}>
              <div className={`font-mono text-[8px] tracking-widest ${muted}`}>{s.l}</div>
              <div className="font-display text-[38px] tracking-widest" style={{ color: s.c }}>{s.v}</div>
            </div>
          ))}
        </div>

        {/* Filter tabs */}
        <div className={`panel ${cBg} p-3 flex flex-wrap gap-1`}>
          {(["ALL", "LIVE", "RECOGNISED", "RESOLVED"] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`font-mono text-[8.5px] tracking-widest px-3 py-1.5 border transition-all
                ${filter === f ? "bg-signal-red text-ivory border-signal-red" : `${bdr} ${muted} hover:border-brass`}`}>
              {f}
            </button>
          ))}
        </div>

        {/* Incident cards */}
        <div className="space-y-3">
          {shown.map(inc => {
            const t = tc(inc.type);
            const st = inc.derivedStatus;
            const sc = STATUS_COLOR[st];
            const isLive = st === "LIVE";
            const iconStyle = { color: t };
            return (
              <div key={inc.id} className={`panel ${cBg} overflow-hidden`}
                style={{ borderLeft: `3px solid ${t}`, boxShadow: isLive ? GLOW.LIVE : undefined }}>
                <div className="h-[1px]" style={{ background: `${t}60` }} />
                <div className="p-5">
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    <div className="flex items-center gap-2">
                      <span className={`font-mono text-[9px] ${muted}`}>INCIDENT</span>
                      <span className="font-mono text-[10px] text-brass font-medium">#{inc.id}</span>
                    </div>
                    {/* Status badge with glow */}
                    <span
                      className="ml-auto font-mono text-[8.5px] px-2 py-[3px] tracking-widest"
                      style={{
                        background: `${sc}22`,
                        color: sc,
                        border: `1px solid ${sc}60`,
                        boxShadow: GLOW[st],
                      }}>
                      {st}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 mb-3" style={{ color: t }}>
                    <span className="flex-shrink-0" style={iconStyle}>
                      <ThreatIcon state={inc.type} size={58} />
                    </span>
                    <div className="font-display text-[28px] tracking-widest leading-none">{inc.type}</div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {[
                      { l: "LOCATION",   v: inc.location },
                      { l: "PLATFORM",   v: inc.platform },
                      { l: "CONFIDENCE", v: `${inc.confidence}%` },
                    ].map(r => (
                      <div key={r.l} className={`p-2 ${dark ? "bg-charcoal" : "bg-[#F4F0E8]"}`}>
                        <div className={`font-mono text-[7.5px] tracking-widest ${muted} mb-0.5`}>{r.l}</div>
                        <div className={`font-mono text-[11px] ${text}`}>{r.v}</div>
                      </div>
                    ))}
                  </div>

                  <div className={`mt-3 pt-2.5 border-t ${bdr}`}>
                    <span className={`font-mono text-[8.5px] ${muted}`}>DEVICE // </span>
                    <span className="font-mono text-[9.5px] text-brass">{inc.device}</span>
                  </div>
                </div>
              </div>
            );
          })}

          {shown.length === 0 && (
            <div className={`panel ${cBg} p-10 text-center`}>
              <div className={`font-mono text-[10px] tracking-widest ${muted}`}>NO INCIDENTS MATCH FILTER</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
