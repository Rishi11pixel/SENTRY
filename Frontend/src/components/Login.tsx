import { useState } from "react";
import { Shield, Eye, EyeOff } from "lucide-react";

interface Props { onLogin: () => void; }

export default function Login({ onLogin }: Props) {
  const [showPass, setShowPass] = useState(false);
  const [railId, setRailId]   = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(false);
  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-obsidian">

      {/* ─── LEFT PANEL ─── */}
      <div className="relative flex-1 flex flex-col justify-between p-8 md:p-14 overflow-hidden bg-obsidian">
        {/* grid */}
        <div className="absolute inset-0 grid-bg opacity-70 pointer-events-none" />

        {/* SVG background — railway schematic */}
        <svg className="absolute inset-0 w-full h-full opacity-[0.055] pointer-events-none" viewBox="0 0 800 600" fill="none" preserveAspectRatio="xMidYMid meet">
          {/* Main rails */}
          <line x1="160" y1="0"   x2="60"  y2="600" stroke="#F1EDE3" strokeWidth="9"/>
          <line x1="440" y1="0"   x2="340" y2="600" stroke="#F1EDE3" strokeWidth="9"/>
          {/* Sleepers */}
          {Array.from({length:22},(_,i)=>i).map(i=>{
            const y=i*29; const x1=157-(i*4.5); const x2=440+(i*4.5);
            return <line key={i} x1={x1} y1={y} x2={x2} y2={y} stroke="#F1EDE3" strokeWidth="7"/>;
          })}
          {/* Platform edge */}
          <rect x="460" y="80" width="280" height="420" rx="0" stroke="#C49A4A" strokeWidth="2" fill="none"/>
          <line x1="460" y1="140" x2="740" y2="140" stroke="#C49A4A" strokeWidth="1" strokeDasharray="6 4"/>
          <line x1="460" y1="200" x2="740" y2="200" stroke="#C49A4A" strokeWidth="1" strokeDasharray="6 4"/>
          {/* Device outline */}
          <rect x="560" y="210" width="100" height="185" rx="10" stroke="#C49A4A" strokeWidth="2.5"/>
          <rect x="576" y="225" width="68"  height="55"  rx="2"  stroke="#C49A4A" strokeWidth="1.2"/>
          <circle cx="610" cy="345" r="22" stroke="#B3262E" strokeWidth="2.5"/>
          <circle cx="610" cy="345" r="10" stroke="#B3262E" strokeWidth="1.2" strokeDasharray="3 3"/>
          <line x1="560" y1="290" x2="660" y2="290" stroke="#C49A4A" strokeWidth="1" strokeDasharray="4 4"/>
          <text x="610" y="403" textAnchor="middle" fontSize="10" fill="#C49A4A" fontFamily="monospace" letterSpacing="2">SENTRY</text>
          {/* Signal markers */}
          <circle cx="80"  cy="180" r="12" stroke="#20C878" strokeWidth="2"/>
          <circle cx="80"  cy="230" r="12" stroke="#D99A27" strokeWidth="2"/>
          <circle cx="80"  cy="280" r="12" stroke="#B3262E" strokeWidth="2"/>
          <line x1="80" y1="120" x2="80" y2="160" stroke="#A8A39A" strokeWidth="2"/>
          <line x1="80" y1="300" x2="80" y2="380" stroke="#A8A39A" strokeWidth="2"/>
        </svg>

        {/* Logo */}
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-9 h-9 bg-signal-red flex items-center justify-center flex-shrink-0">
              <Shield className="w-5 h-5 text-ivory"/>
            </div>
            <div>
              <div className="font-display text-3xl text-ivory tracking-widest leading-none">SENTRY</div>
              <div className="font-mono text-[8.5px] text-warm-grey tracking-[.18em] uppercase leading-tight mt-0.5">SMART EXPLOSIVE & NARCOTIC TRACE RECOGNITION SYSTEM</div>
            </div>
          </div>
          <div className="h-[1px] bg-signal-red w-28 mt-4"/>
        </div>

        {/* Headline */}
        <div className="relative z-10">
          <div className="font-display text-[56px] md:text-[72px] text-ivory leading-[.92] tracking-wide mb-6">
            SECURE THE<br/><span className="text-signal-red">RAILWAY.</span><br/>DETECT THE<br/>UNSEEN.
          </div>
          <p className="font-body text-[13px] text-warm-grey max-w-[340px] leading-relaxed">
            Centralized monitoring and real-time intelligence for connected SENTRY detection devices across Indian railway networks.
          </p>
        </div>

        {/* Footer meta */}
        <div className="relative z-10">
          <div className="h-[1px] bg-brass/25 mb-4"/>
          <div className="font-mono text-[9.5px] text-warm-grey space-y-1 tracking-[.18em] uppercase">
            <div className="text-brass">MINISTRY OF RAILWAYS // SECURITY SYSTEMS DIVISION</div>
            <div>SECURE NETWORK // AES-256 // TLS 1.3</div>
            <div>SENTRY OS // CONTROL CENTER v1.0.0-PROD</div>
          </div>
        </div>
      </div>

      {/* ─── RIGHT PANEL ─── */}
      <div className="w-full md:w-[420px] flex flex-col justify-center p-8 md:p-12 relative bg-charcoal">
        {/* thin red top bar */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-signal-red"/>
        <div className="absolute top-4 right-5 font-mono text-[9px] tracking-widest uppercase text-warm-grey/50">v1.0.0</div>

        <div className="max-w-sm w-full mx-auto">
          <div className="mb-8">
            <div className="font-display text-[32px] tracking-widest leading-none mb-1 text-ivory">WELCOME TO SENTRY</div>
            <div className="font-mono text-[10px] tracking-[.18em] uppercase text-warm-grey">Railway Security Command Center</div>
            <div className="h-[1px] bg-signal-red w-14 mt-3"/>
          </div>

          <form onSubmit={e=>{e.preventDefault();onLogin();}} className="space-y-5">
            {/* Rail ID */}
            <div>
              <label className={`block font-mono text-[9.5px] tracking-[.22em] uppercase mb-2 text-brass`}>RAIL_ID</label>
              <input type="text" className="input-sentry" placeholder="e.g. RAIL_ADM_001"
                value={railId} onChange={e=>setRailId(e.target.value)}/>
            </div>
            {/* Password */}
            <div>
              <label className="block font-mono text-[9.5px] tracking-[.22em] uppercase mb-2 text-brass">PASSWORD</label>
              <div className="relative">
                <input type={showPass?"text":"password"} className="input-sentry pr-10"
                  placeholder="Enter password" value={password} onChange={e=>setPassword(e.target.value)}/>
                <button type="button" onClick={()=>setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors text-warm-grey hover:text-brass">
                  {showPass ? <EyeOff className="w-4 h-4"/> : <Eye className="w-4 h-4"/>}
                </button>
              </div>
            </div>
            {/* Remember */}
            <div className="flex items-center gap-3">
              <div onClick={()=>setRemember(!remember)}
                className={`w-4 h-4 border cursor-pointer flex items-center justify-center flex-shrink-0 transition-all ${remember?"bg-signal-red border-signal-red":"border-warm-grey/30"}`}>
                {remember && <div className="w-2 h-1.5 bg-ivory"/>}
              </div>
              <span onClick={()=>setRemember(!remember)}
                className="font-mono text-[10px] tracking-[.15em] uppercase cursor-pointer text-warm-grey">
                REMEMBER THIS DEVICE
              </span>
            </div>
            {/* Submit */}
            <button type="submit" className="btn-primary w-full tracking-[.16em]">LOGIN TO SENTRY</button>

          </form>

          <div className="mt-10 pt-5 border-t border-warm-grey/10 text-center space-y-1">
            <div className="font-mono text-[9px] tracking-[.2em] uppercase text-warm-grey/40">SECURE RAILWAY NETWORK</div>
            <div className="font-mono text-[9px] tracking-[.2em] uppercase text-signal-red">AUTHORIZED PERSONNEL ONLY</div>
          </div>
        </div>
      </div>
    </div>
  );
}
