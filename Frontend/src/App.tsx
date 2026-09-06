import { useState, useEffect } from "react";
import { X } from "lucide-react";
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

/* ── Toast ── */
interface Toast { id:number; msg:string; type:"alert"|"info"|"success"; state:string; }
function ToastBar({ toasts, onDismiss }:{ toasts:Toast[]; onDismiss:(id:number)=>void }) {
  return (
    <div className="fixed top-4 right-4 z-[60] space-y-2 pointer-events-none">
      {toasts.map(t=>(
        <div key={t.id} className={`slide-in flex items-center gap-3 px-4 py-3 min-w-[280px] pointer-events-auto border shadow-xl ${
          t.type==="alert" ?"bg-signal-red border-signal-red/60 text-ivory"
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

  function addToast(msg:string, type:Toast["type"]="info", state="SAFE") {
    const id=toastId+1; setToastId(id);
    setToasts(t=>[...t,{id,msg,type,state}]);
    setTimeout(()=>setToasts(t=>t.filter(x=>x.id!==id)),4500);
  }

  useEffect(()=>{
    if(!loggedIn)return;
    const id=setTimeout(()=>addToast("ANOMALY DETECTED — SENTRY-032 / ENTRY GATE 2 — EXPLOSIVE PROXY 94%","alert","EXPLOSIVE PROXY"),2000);
    return ()=>clearTimeout(id);
  },[loggedIn]);

  if(!loggedIn) {
    return <Login onLogin={()=>setLoggedIn(true)} />;
  }

  return (
    <div className="dark flex h-screen overflow-hidden bg-obsidian">
      <Sidebar
        active={screen}
        onNav={s=>{setScreen(s);setMobileNav(false);}}
        mobileOpen={mobileNav}
        onMobileToggle={()=>setMobileNav(o=>!o)}
        alertCount={1}
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
              <span className="font-mono text-[8.5px] text-signal-red tracking-widest hidden sm:block">3 ACTIVE ALERTS</span>
            </button>
            <div className="font-mono text-[8.5px] tracking-widest hidden sm:block text-warm-grey/50">v1.0.0</div>
          </div>
        </div>

        {/* Screen */}
        <div className="flex-1 overflow-y-auto">
          {screen==="dashboard"&&(
            <Dashboard theme="dark"
              onViewDevice={id=>{setDevId(id);setScreen("device-info");}}
              onViewAnomaly={()=>setScreen("anomalies")}/>
          )}
          {screen==="live-tracking"&&<LiveTracking theme="dark"/>}
          {screen==="device-info"&&(
            <DeviceInfo deviceId={devId} theme="dark"
              onBack={()=>setScreen("devices")}
              onLiveTracking={()=>setScreen("live-tracking")}/>
          )}
          {screen==="devices"&&<Devices theme="dark" onViewDevice={id=>{setDevId(id);setScreen("device-info");}}/>}
          {screen==="anomalies"&&(
            <AnomalyAlert theme="dark"
              onAcknowledge={()=>addToast("Issue recognised — RAIL_ADM_001","success","SAFE")}
              onResolved={()=>{ addToast("EXPLOSIVE PROXY resolved — moved to Threat History","success","EXPLOSIVE PROXY"); setScreen("threat-history"); }}/>
          )}
          {screen==="threat-history"&&<ThreatHistory theme="dark"/>}
          {screen==="logs"&&<SystemLogs theme="dark"/>}
        </div>
      </div>

      <ToastBar toasts={toasts} onDismiss={id=>setToasts(t=>t.filter(x=>x.id!==id))}/>
    </div>
  );
}
