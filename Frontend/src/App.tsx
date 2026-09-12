import { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";
import { acknowledgeIncident, getDashboardSummary, getDevices, getIncidents, getLogs, resolveIncident } from "./api";
import ThreatIcon from "./components/ThreatIcon";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import type { Screen } from "./components/Sidebar";
import Dashboard from "./screens/Dashboard";
import LiveTracking from "./screens/LiveTracking";
import DeviceInfo from "./screens/DeviceInfo";
import AnomalyAlert from "./screens/AnomalyAlert";
import ThreatHistory from "./screens/ResponseCenter";
import Devices from "./screens/Devices";
import SystemLogs from "./screens/SystemLogs";
import { truncateConfidencePercent, type Device, type Incident, type LogEntry } from "./data";

/* ── Toast ── */
interface Toast { id:number; msg:string; type:"alert"|"warning"|"info"|"success"; state:string; }
function ToastBar({ toasts, onDismiss }:{ toasts:Toast[]; onDismiss:(id:number)=>void }) {
  return (
    <div className="fixed top-4 right-4 z-[60] space-y-2 pointer-events-none">
      {toasts.map(t=>(
        <div key={t.id} className={`slide-in flex items-center gap-3 px-4 py-3 min-w-[280px] pointer-events-auto border shadow-xl ${
          t.type==="alert" ?"bg-signal-red border-signal-red/60 text-ivory"
          :t.type==="warning"?"bg-caution border-caution/60 text-obsidian"
          :t.type==="success"?"bg-safe/90 border-safe/50 text-ivory"
          :"bg-charcoal border-warm-grey/20 text-ivory"
        }`}>
          <ThreatIcon state={t.state} size={28} className="flex-shrink-0" />
          <span className="font-mono text-[10px] tracking-widest flex-1">{t.msg}</span>
          <button onClick={()=>onDismiss(t.id)} className="flex-shrink-0 opacity-70 hover:opacity-100 transition-opacity">
            <X className="w-3.5 h-3.5"/>
          </button>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [loggedIn, setLoggedIn]   = useState(false);
  const [screen, setScreen]       = useState<Screen>("dashboard");
  const [devId, setDevId]         = useState("SENTRY-032");
  const [mobileNav, setMobileNav] = useState(false);
  const [toasts, setToasts]       = useState<Toast[]>([]);
  const [toastId, setToastId]     = useState(0);
  const [devices, setDevices]     = useState<Device[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [logs, setLogs]           = useState<LogEntry[]>([]);
  const [summary, setSummary]     = useState({ connectedDevices: 0, online: 0, offline: 0, activeAlerts: 0 });
  const seenIncidentIds = useRef<Set<string> | null>(null);

  function addToast(msg:string, type:Toast["type"]="info", state="SAFE") {
    const id=toastId+1; setToastId(id);
    setToasts(t=>[...t,{id,msg,type,state}]);
    setTimeout(()=>setToasts(t=>t.filter(x=>x.id!==id)),4500);
  }

  useEffect(() => {
    if (!loggedIn) return;
    let active = true;
    const refresh = async () => {
      try {
        const [nextDevices, nextSummary, nextIncidents, nextLogs] = await Promise.all([
          getDevices(), getDashboardSummary(), getIncidents(), getLogs(),
        ]);
        if (!active) return;
        setDevices(nextDevices);
        setSummary(nextSummary);
        setIncidents(nextIncidents);
        setLogs(nextLogs);

        const currentIds = new Set(nextIncidents.map(incident => incident.id));
        if (seenIncidentIds.current) {
          nextIncidents
            .filter(incident => !seenIncidentIds.current?.has(incident.id) && incident.status !== "RESOLVED" && incident.type !== "ALCOHOL")
            .forEach(incident => {
              const isWarning = incident.type === "ALCOHOL";
              addToast(
                `${isWarning ? "WARNING" : "ANOMALY DETECTED"} — ${incident.device} / ${incident.location} — ${incident.type} ${truncateConfidencePercent(incident.confidence)}%`,
                isWarning ? "warning" : "alert",
                incident.type,
              );
            });
        }
        seenIncidentIds.current = currentIds;
      } catch {
        // Keep the last known state while the local backend is unavailable.
      }
    };
    refresh();
    const interval = window.setInterval(refresh, 2000);
    return () => { active = false; window.clearInterval(interval); };
  }, [loggedIn]);

  async function handleAcknowledge(id: string) {
    await acknowledgeIncident(id);
    setIncidents(await getIncidents());
  }

  async function handleResolve(id: string) {
    await resolveIncident(id);
    setIncidents(await getIncidents());
  }

  if(!loggedIn) {
    return <Login onLogin={()=>setLoggedIn(true)} />;
  }

  const activeAlertCount = incidents.filter(incident => incident.status !== "RESOLVED" && incident.type !== "ALCOHOL").length;
  const activeThreatIncident = incidents.find(incident => incident.status !== "RESOLVED" && incident.type !== "ALCOHOL");
  const displaySummary = { ...summary, activeAlerts: activeAlertCount };

  return (
    <div className="dark flex h-screen overflow-hidden bg-obsidian">
      <Sidebar
        active={screen}
        onNav={s=>{setScreen(s);setMobileNav(false);}}
        mobileOpen={mobileNav}
        onMobileToggle={()=>setMobileNav(o=>!o)}
        alertCount={activeAlertCount}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Topbar */}
        <div className="flex-shrink-0 flex items-center justify-between px-4 md:px-6 py-2.5 border-b bg-charcoal border-warm-grey/10">
          <div className="flex items-center gap-3 md:pl-0 pl-12">
            <div className="font-mono text-[9.5px] tracking-[.18em] uppercase text-warm-grey">
              <span className="text-brass">SENTRY</span> // CONTROL CENTER // NEW DELHI JN
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={()=>setScreen("anomalies")}
              className="flex items-center gap-1.5 hover:opacity-80 transition-opacity">
              <span className="w-1.5 h-1.5 rounded-full bg-signal-red blink"/>
              <span className="font-mono text-[8.5px] text-signal-red tracking-widest hidden sm:block">
                {activeAlertCount} ACTIVE ALERTS
              </span>
            </button>
            <div className="font-mono text-[8.5px] tracking-widest hidden sm:block text-warm-grey/50">v1.0.0</div>
          </div>
        </div>

        {/* Screen */}
        <div className="flex-1 overflow-y-auto">
          {screen==="dashboard"&&(
            <Dashboard theme="dark"
              devices={devices} incidents={incidents} logs={logs} summary={displaySummary}
              onViewDevice={id=>{setDevId(id);setScreen("device-info");}}
              onViewAnomaly={()=>setScreen("anomalies")}/>
          )}
          {screen==="live-tracking"&&<LiveTracking theme="dark" devices={devices}/>} 
          {screen==="device-info"&&(
            <DeviceInfo deviceId={devId} theme="dark" devices={devices}
              onBack={()=>setScreen("devices")}
              onLiveTracking={()=>setScreen("live-tracking")}/>
          )}
          {screen==="devices"&&<Devices theme="dark" devices={devices} onViewDevice={id=>{setDevId(id);setScreen("device-info");}}/>}
          {screen==="anomalies"&&(
            <AnomalyAlert theme="dark"
              incident={activeThreatIncident}
              onAcknowledge={async id=>{ await handleAcknowledge(id); addToast("Issue recognised — RAIL_ADM_001","success","SAFE"); }}
              onResolved={async id=>{ await handleResolve(id); addToast("Incident resolved — moved to Threat History","success","SAFE"); setScreen("threat-history"); }}/>
          )}
          {screen==="threat-history"&&<ThreatHistory theme="dark" incidents={incidents}/>} 
          {screen==="logs"&&<SystemLogs theme="dark" logs={logs}/>} 
        </div>
      </div>

      <ToastBar toasts={toasts} onDismiss={id=>setToasts(t=>t.filter(x=>x.id!==id))}/>
    </div>
  );
}
