import { useEffect, useState } from "react";
import { ArrowLeft, Battery, Signal, MapPin } from "lucide-react";
import { getDevicePredictions } from "../api";
import type { Device } from "../data";
import type { PredictionResult } from "../data";
import ThreatIcon from "../components/ThreatIcon";

interface Props { deviceId: string; theme: "dark" | "light"; devices: Device[]; onBack: () => void; onLiveTracking: () => void; }

const RC: Record<string, string> = {
  "EXPLOSIVE PROXY": "#B3262E",
  "NARCOTIC PROXY":  "#B3262E",
  "ALCOHOL":         "#D99A27",
};

function displayResult(result: string) {
  return result === "CAUTION" ? "WEATHER DRIFT" : result;
}

export default function DeviceInfo({ deviceId, theme, devices, onBack, onLiveTracking }: Props) {
  const dark   = theme === "dark";
  const device = devices.find(d => d.id === deviceId);
  const [livePredictions, setLivePredictions] = useState<PredictionResult[]>([]);

  useEffect(() => {
    if (!device) return;
    let active = true;
    const refresh = () => getDevicePredictions(device.id).then(next => {
      if (active) setLivePredictions(next);
    }).catch(() => undefined);
    refresh();
    const interval = window.setInterval(refresh, 2000);
    return () => { active = false; window.clearInterval(interval); };
  }, [device?.id]);

  const anomalies = livePredictions.map(item => ({ timestamp: item.timestamp, result: item.displayResult || item.prediction || "SAFE", confidence: item.confidence * 100 }));

  const bg   = dark ? "bg-obsidian"   : "bg-[#F1EDE3]";
  const cBg  = dark ? "bg-gunmetal"   : "bg-white";
  const text = dark ? "text-ivory"    : "text-obsidian";
  const muted= dark ? "text-warm-grey": "text-[#6F6A61]";
  const bdr  = dark ? "border-warm-grey/10" : "border-obsidian/8";

  if (!device) {
    return (
      <div className={`min-h-full ${bg} p-4 md:p-6`}>
        <div className={`panel ${cBg} p-10 text-center`}>
          <div className={`font-mono text-[10px] tracking-widest ${muted}`}>DEVICE NOT AVAILABLE</div>
        </div>
      </div>
    );
  }

  const sc = device.lastResult === "ALCOHOL"
    ? "#D99A27"
    : { ONLINE: "#20C878", OFFLINE: "#A8A39A", ALERT: "#B3262E", WARNING: "#D99A27" }[device.status];
  const shownStatus = device.lastResult === "ALCOHOL" ? "WARNING" : device.status;
  const battColor = device.battery > 50 ? "#20C878" : device.battery > 20 ? "#D99A27" : "#B3262E";

  return (
    <div className={`min-h-full ${bg}`}>
      <div className="p-4 md:p-6 space-y-4 pb-20 md:pb-6">

        {/* Back */}
        <button onClick={onBack} className={`flex items-center gap-2 transition-colors ${muted} hover:text-brass`}>
          <ArrowLeft className="w-3.5 h-3.5" />
          <span className="font-mono text-[9.5px] tracking-widest uppercase">BACK TO DEVICES</span>
        </button>

        {/* Header card */}
        <div className={`panel ${cBg} overflow-hidden`}>
          <div className="h-[2px]" style={{ background: sc }} />
          <div className="p-5">

            {/* Identity + location */}
            <div className="flex-1">
              <div className="flex flex-wrap items-start gap-3 mb-4">
                <div>
                  <div className={`font-display text-[44px] md:text-[52px] tracking-widest leading-none ${text}`}>
                    {device.id}
                  </div>
                  <div className={`font-mono text-[9px] tracking-widest ${muted} mt-1`}>
                    SENTRY DETECTION UNIT
                  </div>
                </div>
                <span className="font-mono text-[8.5px] tracking-widest px-2 py-1 inline-flex items-center gap-1.5 mt-1"
                  style={{ background: `${sc}22`, color: sc, border: `1px solid ${sc}40` }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: sc }} />
                  {shownStatus}
                </span>
              </div>

              <div className="flex items-center gap-2 mb-6">
                <MapPin className="w-3.5 h-3.5 text-brass" />
                <span className="font-mono text-[10px] tracking-widest text-brass">
                  {device.platform} // {device.location}
                </span>
              </div>

              {/* Device Status + Battery — only these two */}
              <div className="grid grid-cols-2 gap-3">
                {/* Device Status */}
                <div className={`p-4 ${dark ? "bg-charcoal" : "bg-[#F4F0E8]"}`}>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Signal className="w-3.5 h-3.5 flex-shrink-0" style={{ color: sc }} />
                    <span className={`font-mono text-[8px] tracking-widest uppercase ${muted}`}>DEVICE STATUS</span>
                  </div>
                  <div className="font-mono text-[16px] font-medium" style={{ color: sc }}>
                    {shownStatus}
                  </div>
                </div>

                {/* Battery */}
                <div className={`p-4 ${dark ? "bg-charcoal" : "bg-[#F4F0E8]"}`}>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Battery className="w-3.5 h-3.5 flex-shrink-0" style={{ color: battColor }} />
                    <span className={`font-mono text-[8px] tracking-widest uppercase ${muted}`}>BATTERY</span>
                  </div>
                  <div className="font-mono text-[16px] font-medium" style={{ color: device.battery > 0 ? battColor : "#A8A39A" }}>
                    {device.battery > 0 ? `${device.battery}%` : "—"}
                  </div>
                  {device.battery > 0 && (
                    <div className={`mt-2 h-1 w-full ${dark ? "bg-obsidian" : "bg-[#E4E0D7]"}`}>
                      <div className="h-full transition-all" style={{ width: `${device.battery}%`, background: battColor }} />
                    </div>
                  )}
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Recent anomalies — last 5 */}
        <div className={`panel ${cBg}`}>
          <div className="h-[1px] bg-signal-red/30" />
          <div className="p-4">
            <div className="font-mono text-[9px] tracking-[.2em] uppercase text-brass mb-3">RECENT ANOMALIES</div>
            {anomalies.length > 0 ? (
              <table className="w-full">
                <thead>
                  <tr className={`border-b ${bdr}`}>
                    {["TIME", "DETECTION TYPE", "CONFIDENCE"].map(h => (
                      <th key={h} className={`text-left pb-2.5 font-mono text-[8px] tracking-widest ${muted}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {anomalies.slice(0, 5).map((a, i) => {
                    const c = RC[a.result] || "#B3262E";
                    return (
                      <tr key={i} className={`border-b ${bdr} last:border-0`}>
                        <td className="py-2.5 font-mono text-[9.5px] text-brass whitespace-nowrap">{a.timestamp}</td>
                        <td className="py-2.5 pr-4">
                          <div className="flex items-center gap-2" style={{ color: c }}>
                            <ThreatIcon state={a.result} size={18} className="flex-shrink-0" />
                            <span className="font-mono text-[8.5px] px-1.5 py-[2px] whitespace-nowrap"
                              style={{ background: `${c}20`, color: c, border: `1px solid ${c}40` }}>
                              {displayResult(a.result)}
                            </span>
                          </div>
                        </td>
                        <td className="py-2.5 font-mono text-[11px] font-medium" style={{ color: c }}>
                          {a.confidence}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <div className={`py-6 text-center font-mono text-[9.5px] tracking-widest ${muted}`}>
                NO ANOMALIES RECORDED
              </div>
            )}
          </div>
        </div>

        <button onClick={onLiveTracking} className="btn-primary tracking-[.15em]">
          OPEN LIVE TRACKING
        </button>
      </div>
    </div>
  );
}
