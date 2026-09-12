import { useState } from "react";
import { Search, Battery } from "lucide-react";
import type { Device } from "../data";

interface Props { theme: "dark" | "light"; devices: Device[]; onViewDevice: (id: string) => void; }

const SC: Record<string, string> = {
  ONLINE:  "#20C878",
  OFFLINE: "#A8A39A",
  ALERT:   "#B3262E",
  WARNING: "#D99A27",
};

function displayStatus(d: Device): Device["status"] {
  return d.lastResult === "ALCOHOL" ? "WARNING" : d.status;
}

function DeviceCard({ d, theme, onClick }: { d: Device; theme: "dark" | "light"; onClick: () => void }) {
  const dark = theme === "dark";
  const shownStatus = displayStatus(d);
  const sc   = SC[shownStatus];
  const muted= dark ? "text-warm-grey" : "text-[#6F6A61]";
  const text = dark ? "text-ivory"     : "text-obsidian";
  const bdr  = dark ? "border-warm-grey/10" : "border-obsidian/8";

  const battColor = d.battery > 50 ? "#20C878" : d.battery > 20 ? "#D99A27" : "#B3262E";

  return (
    <button onClick={onClick}
      className={`w-full text-left panel transition-all ${dark ? "bg-gunmetal hover:bg-charcoal" : "bg-white hover:bg-[#F4F0E8] panel-light"}`}>
      <div className="h-[2px]" style={{ background: sc }} />
      <div className="p-4">
        {/* ID + status badge */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className={`font-display text-[26px] tracking-widest leading-none ${text}`}>{d.id}</div>
            <div className={`font-mono text-[9px] tracking-widest ${muted} mt-0.5`}>{d.location}</div>
          </div>
          <span className="font-mono text-[8px] px-2 py-[3px] tracking-widest inline-flex items-center gap-1.5 mt-0.5"
            style={{ background: `${sc}22`, color: sc, border: `1px solid ${sc}40` }}>
            <span className={`w-1.5 h-1.5 rounded-full ${shownStatus === "ONLINE" ? "pulse-green" : shownStatus === "ALERT" ? "pulse-red" : shownStatus === "WARNING" ? "pulse-amber" : ""}`}
              style={{ background: sc }} />
            {shownStatus}
          </span>
        </div>

        {/* Battery only */}
        <div className={`flex items-center gap-2 p-2.5 ${dark ? "bg-charcoal" : "bg-[#F4F0E8]"}`}>
          <Battery className="w-3.5 h-3.5 flex-shrink-0" style={{ color: battColor }} />
          <span className={`font-mono text-[8.5px] tracking-widest uppercase ${muted}`}>BATTERY</span>
          <span className="font-mono text-[14px] ml-auto" style={{ color: d.battery > 0 ? battColor : "#A8A39A" }}>
            {d.battery > 0 ? `${d.battery}%` : "—"}
          </span>
        </div>
      </div>
    </button>
  );
}

export default function Devices({ theme, devices, onViewDevice }: Props) {
  const dark  = theme === "dark";
  const [q, setQ]   = useState("");
  const [sf, setSf] = useState("ALL");

  const bg   = dark ? "bg-obsidian"   : "bg-[#F1EDE3]";
  const cBg  = dark ? "bg-gunmetal"   : "bg-white";
  const text = dark ? "text-ivory"    : "text-obsidian";
  const muted= dark ? "text-warm-grey": "text-[#6F6A61]";
  const bdr  = dark ? "border-warm-grey/10" : "border-obsidian/8";

  const filtered = devices
    .filter(d => !q || d.id.toLowerCase().includes(q.toLowerCase()) || d.location.toLowerCase().includes(q.toLowerCase()))
    .filter(d => sf === "ALL" || (sf === "ONLINE" ? d.status !== "OFFLINE" : d.status === sf));

  return (
    <div className={`min-h-full ${bg}`}>
      <div className="p-4 md:p-6 space-y-4 pb-20 md:pb-6">

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div>
            <div className={`font-display text-[40px] md:text-[50px] tracking-widest leading-none ${text}`}>DEVICES</div>
            <div className={`font-mono text-[9.5px] tracking-widest uppercase mt-1 ${muted}`}>
                  {devices.length} registered units //&nbsp;
                  {devices.filter(d => d.status === "ONLINE").length} online //&nbsp;
                  {devices.filter(d => d.status === "ALERT" && d.lastResult !== "ALCOHOL").length} alert
            </div>
          </div>
          <div className={`font-mono text-[8.5px] tracking-widest px-2 py-1 ${cBg} border ${bdr}`}>
            <span className="text-brass">STATION //</span> NEW DELHI JN (NDLS)
          </div>
        </div>

        {/* Filters */}
        <div className={`panel ${cBg} p-4`}>
          <div className="flex flex-wrap gap-3 items-center">
            <div className="relative flex-1 min-w-[180px]">
              <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 ${muted}`} />
              <input type="text" placeholder="SEARCH DEVICES..." value={q}
                onChange={e => setQ(e.target.value)} className="input-sentry pl-9 text-[9.5px] py-2" />
            </div>
            <div className="flex gap-1">
              {["ALL", "ONLINE", "OFFLINE"].map(f => (
                <button key={f} onClick={() => setSf(f)}
                  className={`font-mono text-[8.5px] tracking-widest px-3 py-1.5 border transition-all
                    ${sf === f ? "bg-signal-red text-ivory border-signal-red" : `${bdr} ${muted} hover:border-brass`}`}>
                  {f}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {filtered.map(d => (
            <DeviceCard key={d.id} d={d} theme={theme} onClick={() => onViewDevice(d.id)} />
          ))}
        </div>

        {filtered.length === 0 && (
          <div className={`panel ${cBg} p-10 text-center`}>
            <div className={`font-mono text-[10px] tracking-widest ${muted}`}>NO DEVICES MATCH CURRENT FILTERS</div>
          </div>
        )}
      </div>
    </div>
  );
}
