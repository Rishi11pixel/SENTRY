import { Shield, LayoutDashboard, MapPin, HardDrive, AlertTriangle, History, ScrollText, Sun, Moon, Menu, X, ChevronRight } from "lucide-react";

export type Screen = "dashboard"|"live-tracking"|"devices"|"device-info"|"anomalies"|"threat-history"|"logs";

interface Props {
  active: Screen;
  onNav: (s:Screen)=>void;
  theme: "dark"|"light";
  onTheme: ()=>void;
  mobileOpen: boolean;
  onMobileToggle: ()=>void;
  alertCount: number;
}

const NAV = [
  { id:"dashboard"      as Screen, label:"DASHBOARD",      icon:LayoutDashboard, n:"01" },
  { id:"live-tracking"  as Screen, label:"LIVE TRACKING",  icon:MapPin,          n:"02" },
  { id:"devices"        as Screen, label:"DEVICES",         icon:HardDrive,       n:"03" },
  { id:"anomalies"      as Screen, label:"ANOMALIES",       icon:AlertTriangle,   n:"04" },
  { id:"logs"           as Screen, label:"SYSTEM LOGS",     icon:ScrollText,      n:"05" },
  { id:"threat-history" as Screen, label:"THREAT HISTORY",  icon:History,         n:"06" },
] as const;

function SidebarInner({ active, onNav, theme, onTheme, alertCount }: Omit<Props,"mobileOpen"|"onMobileToggle">) {
  const dark = theme==="dark";
  const bg   = dark?"bg-charcoal":"bg-white";
  const text = dark?"text-ivory":"text-obsidian";
  const muted= dark?"text-warm-grey":"text-[#6F6A61]";
  const bdr  = dark?"border-warm-grey/10":"border-obsidian/8";

  const isActive=(id:string)=> active===id||(id==="devices"&&active==="device-info");

  return (
    <div className={`h-full flex flex-col ${bg} border-r ${bdr} w-[220px] flex-shrink-0`}>
      {/* Logo */}
      <div className="px-4 pt-5 pb-4">
        <div className="flex items-center gap-2.5 mb-0.5">
          <div className="w-8 h-8 bg-signal-red flex items-center justify-center flex-shrink-0">
            <Shield className="w-4 h-4 text-ivory"/>
          </div>
          <div>
            <div className={`font-display text-[22px] tracking-widest leading-none ${text}`}>SENTRY</div>
            <div className={`font-mono text-[7.5px] tracking-[.15em] uppercase ${muted} mt-0.5`}>CONTROL CENTER</div>
          </div>
        </div>
        <div className="h-[1px] bg-signal-red/40 mt-3.5"/>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2.5 py-1.5 space-y-0.5 overflow-y-auto">
        {NAV.map(item=>{
          const Icon=item.icon;
          const act=isActive(item.id);
          const isAlert=item.id==="anomalies"&&alertCount>0;
          return (
            <button key={item.id} onClick={()=>onNav(item.id as Screen)}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2.5 text-left transition-all group relative border-l-2 ${
                act
                  ? dark?"bg-gunmetal border-l-signal-red":"bg-[#F1EDE3] border-l-signal-red"
                  : dark?"hover:bg-gunmetal/50 border-l-transparent":"hover:bg-[#F1EDE3]/60 border-l-transparent"
              }`}>
              <span className={`font-mono text-[8px] tracking-widest flex-shrink-0 ${act?"text-signal-red":muted}`}>{item.n}</span>
              <Icon className={`w-3.5 h-3.5 flex-shrink-0 ${act?"text-signal-red":muted} transition-colors`}/>
              <span className={`font-heading text-[11px] tracking-[.12em] font-medium flex-1 ${act?text:muted} transition-colors`}>{item.label}</span>
              {isAlert&&<span className="w-2 h-2 rounded-full bg-signal-red blink flex-shrink-0"/>}
              {act&&<ChevronRight className="w-3 h-3 text-signal-red flex-shrink-0"/>}
            </button>
          );
        })}
      </nav>

      {/* Status */}
      <div className={`px-4 py-2.5 border-t ${bdr}`}>
        <div className={`font-mono text-[8px] tracking-[.2em] uppercase ${muted} mb-1.5`}>SYSTEM STATUS</div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-safe pulse-green inline-block flex-shrink-0"/>
          <span className="font-mono text-[9px] text-safe tracking-widest">ALL SYSTEMS OPERATIONAL</span>
        </div>
      </div>

      {/* User */}
      <div className={`px-4 py-2.5 border-t ${bdr} ${dark?"bg-obsidian/40":"bg-[#F4F0E8]/60"}`}>
        <div className={`font-mono text-[8px] tracking-widest uppercase ${muted} mb-0.5`}>USER</div>
        <div className={`font-heading text-[11px] tracking-widest font-semibold ${text}`}>RAIL_ADM_001</div>
        <div className={`font-mono text-[8px] tracking-widest ${muted}`}>Security Div // NR-NDLS</div>
      </div>

      {/* Theme toggle */}
      <div className={`flex items-center justify-end px-4 py-2.5 border-t ${bdr}`}>
        <button onClick={onTheme}
          className={`flex items-center gap-1.5 px-2 py-1 border transition-all ${dark?"border-warm-grey/20 hover:border-brass":"border-obsidian/15 hover:border-brass"}`}>
          {dark
            ? <><Sun className="w-3 h-3 text-brass"/><span className="font-mono text-[8.5px] text-brass tracking-widest">LIGHT</span></>
            : <><Moon className="w-3 h-3 text-obsidian"/><span className="font-mono text-[8.5px] text-obsidian tracking-widest">DARK</span></>
          }
        </button>
      </div>
    </div>
  );
}

export default function Sidebar(props: Props) {
  const { mobileOpen, onMobileToggle, ...rest } = props;
  const dark = props.theme==="dark";

  return (
    <>
      {/* Desktop */}
      <div className="hidden md:flex h-full">
        <SidebarInner {...rest}/>
      </div>

      {/* Mobile hamburger */}
      <button className="md:hidden fixed top-3.5 left-3.5 z-50 w-9 h-9 bg-signal-red flex items-center justify-center" onClick={onMobileToggle}>
        {mobileOpen ? <X className="w-4 h-4 text-ivory"/> : <Menu className="w-4 h-4 text-ivory"/>}
      </button>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40">
          <div className="absolute inset-0 bg-obsidian/75" onClick={onMobileToggle}/>
          <div className="absolute left-0 top-0 bottom-0 z-50">
            <SidebarInner {...rest} onNav={s=>{rest.onNav(s);onMobileToggle();}}/>
          </div>
        </div>
      )}

      {/* Mobile bottom nav */}
      <div className={`md:hidden fixed bottom-0 left-0 right-0 z-40 flex border-t ${dark?"bg-charcoal border-warm-grey/10":"bg-white border-obsidian/10"}`}>
        {[
          {id:"dashboard",icon:LayoutDashboard,label:"HOME"},
          {id:"live-tracking",icon:MapPin,label:"LIVE"},
          {id:"devices",icon:HardDrive,label:"DEVICES"},
          {id:"anomalies",icon:AlertTriangle,label:"ALERTS"},
          {id:"logs",icon:ScrollText,label:"LOGS"},
        ].map(item=>{
          const Icon=item.icon;
          const act=props.active===item.id||(item.id==="devices"&&props.active==="device-info");
          const isAlert=item.id==="anomalies"&&props.alertCount>0;
          return (
            <button key={item.id} onClick={()=>props.onNav(item.id as Screen)}
              className={`flex-1 flex flex-col items-center gap-0.5 py-2.5 relative transition-colors ${act?"text-signal-red":dark?"text-warm-grey":"text-[#6F6A61]"}`}>
              {act&&<div className="absolute top-0 left-0 right-0 h-[2px] bg-signal-red"/>}
              <Icon className="w-4 h-4"/>
              <span className="font-mono text-[7px] tracking-widest">{item.label}</span>
              {isAlert&&<span className="absolute top-1.5 right-1/2 translate-x-3 w-1.5 h-1.5 rounded-full bg-signal-red blink"/>}
            </button>
          );
        })}
      </div>
    </>
  );
}
