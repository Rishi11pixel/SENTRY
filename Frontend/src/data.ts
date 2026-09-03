export type DeviceStatus = "ONLINE" | "OFFLINE" | "ALERT" | "WARNING";
export type ScanResult  = "SAFE" | "CAUTION" | "EXPLOSIVE PROXY" | "NARCOTIC PROXY";
export type LogLevel    = "ALERT" | "WARNING" | "SUCCESS" | "INFO";

export interface Device {
  id: string;
  location: string;
  platform: string;
  status: DeviceStatus;
  battery: number;
  signal: number;
  health: number;
  lastSync: string;
  operatingTime: string;
  lastResult: ScanResult;
  confidence: number;
  mapX: number; // percent positions on SVG viewbox 0-100
  mapY: number;
}

export interface LogEntry {
  timestamp: string;
  device: string;
  event: string;
  location: string;
  level: LogLevel;
}

export interface Incident {
  id: string;
  type: ScanResult;
  location: string;
  platform: string;
  confidence: number;
  status: "RESPONSE DISPATCHED" | "TEAM NOTIFIED" | "RESOLVED" | "ACKNOWLEDGED";
  team: string;
  time: string;
  device: string;
}

export interface ScanRecord {
  timestamp: string;
  result: ScanResult;
  confidence: number;
}

export const DEVICES: Device[] = [
  { id:"SENTRY-014", location:"Platform 3", platform:"Platform 3",     status:"ONLINE",   battery:91, signal:95, health:99, lastSync:"14:32:08", operatingTime:"7h 14m", lastResult:"SAFE",           confidence:98, mapX:38, mapY:32 },
  { id:"SENTRY-021", location:"Platform 5", platform:"Platform 5",     status:"ONLINE",   battery:78, signal:88, health:97, lastSync:"14:32:01", operatingTime:"5h 30m", lastResult:"SAFE",           confidence:96, mapX:62, mapY:52 },
  { id:"SENTRY-032", location:"Entry Gate 2", platform:"Platform 4",   status:"ALERT",    battery:84, signal:92, health:97, lastSync:"14:32:08", operatingTime:"6h 42m", lastResult:"EXPLOSIVE PROXY",confidence:94, mapX:18, mapY:72 },
  { id:"SENTRY-008", location:"Coaching Area", platform:"Coaching Bay", status:"ONLINE",  battery:65, signal:72, health:94, lastSync:"14:31:58", operatingTime:"8h 02m", lastResult:"SAFE",           confidence:99, mapX:82, mapY:58 },
  { id:"SENTRY-019", location:"Entry Gate 1", platform:"Main Entrance", status:"WARNING", battery:31, signal:85, health:91, lastSync:"14:29:44", operatingTime:"9h 11m", lastResult:"CAUTION",        confidence:72, mapX:18, mapY:38 },
  { id:"SENTRY-041", location:"Waiting Hall", platform:"Ground Floor",  status:"OFFLINE", battery:0,  signal:0,  health:0,  lastSync:"13:42:11", operatingTime:"—",      lastResult:"SAFE",           confidence:0,  mapX:62, mapY:22 },
];

export const LOGS: LogEntry[] = [
  { timestamp:"14:32:08", device:"ST-032", event:"Explosive proxy detected — Confidence 94%", location:"Entry Gate 2",   level:"ALERT"   },
  { timestamp:"14:31:52", device:"ST-014", event:"Sensor calibration completed",               location:"Platform 3",    level:"SUCCESS" },
  { timestamp:"14:29:14", device:"ST-021", event:"Battery level warning — 31%",                location:"Platform 5",    level:"WARNING" },
  { timestamp:"14:27:01", device:"ST-008", event:"Connection restored",                        location:"Coaching Area", level:"SUCCESS" },
  { timestamp:"14:25:43", device:"ST-019", event:"Temperature anomaly — 41.2°C",              location:"Entry Gate 1",  level:"WARNING" },
  { timestamp:"14:22:09", device:"ST-014", event:"Scan cycle completed — all clear",           location:"Platform 3",    level:"SUCCESS" },
  { timestamp:"14:18:31", device:"ST-041", event:"Device offline — heartbeat lost",            location:"Waiting Hall",  level:"ALERT"   },
  { timestamp:"14:11:04", device:"ST-021", event:"Narcotic proxy detected — Confidence 89%",  location:"Platform 5",    level:"ALERT"   },
  { timestamp:"14:07:22", device:"ST-032", event:"Environmental noise filtered",               location:"Entry Gate 2",  level:"INFO"    },
  { timestamp:"14:02:55", device:"ST-008", event:"Firmware updated successfully",              location:"Coaching Area", level:"SUCCESS" },
  { timestamp:"13:58:14", device:"ST-019", event:"Humidity sensor recalibrated",              location:"Entry Gate 1",  level:"SUCCESS" },
  { timestamp:"13:52:37", device:"ST-014", event:"Narcotic proxy detected — Confidence 82%", location:"Platform 3",    level:"ALERT"   },
  { timestamp:"13:44:02", device:"ST-008", event:"Scan cycle completed — all clear",           location:"Coaching Area", level:"SUCCESS" },
  { timestamp:"13:42:11", device:"ST-041", event:"Last heartbeat received",                   location:"Waiting Hall",  level:"INFO"    },
  { timestamp:"13:38:50", device:"ST-032", event:"Operator handover logged",                  location:"Entry Gate 2",  level:"INFO"    },
];

export const INCIDENTS: Incident[] = [
  { id:"ST032-143208", type:"EXPLOSIVE PROXY", location:"Entry Gate 2", platform:"Platform 4", confidence:94, status:"RESPONSE DISPATCHED", team:"BOMB SQUAD",       time:"14:32:08", device:"SENTRY-032" },
  { id:"ST021-141104", type:"NARCOTIC PROXY",  location:"Platform 5",   platform:"Platform 5", confidence:89, status:"TEAM NOTIFIED",       team:"ANTI-DRUG SQUAD", time:"14:11:04", device:"SENTRY-021" },
  { id:"ST014-135214", type:"NARCOTIC PROXY",  location:"Platform 3",   platform:"Platform 3", confidence:82, status:"RESOLVED",            team:"ANTI-DRUG SQUAD", time:"13:52:14", device:"SENTRY-014" },
];

export const SCAN_HISTORY: ScanRecord[] = [
  { timestamp:"14:32:08", result:"EXPLOSIVE PROXY", confidence:94 },
  { timestamp:"14:31:55", result:"SAFE",            confidence:96 },
  { timestamp:"14:31:42", result:"SAFE",            confidence:94 },
  { timestamp:"14:31:30", result:"CAUTION",         confidence:72 },
  { timestamp:"14:31:18", result:"SAFE",            confidence:98 },
  { timestamp:"14:31:06", result:"SAFE",            confidence:97 },
  { timestamp:"14:30:54", result:"SAFE",            confidence:99 },
  { timestamp:"14:30:41", result:"SAFE",            confidence:95 },
  { timestamp:"14:30:28", result:"SAFE",            confidence:93 },
  { timestamp:"14:30:15", result:"SAFE",            confidence:97 },
];

export const ANOMALY_CHART = [
  { t:"00:00",d:0 },{ t:"01:00",d:1 },{ t:"02:00",d:0 },{ t:"03:00",d:0 },
  { t:"04:00",d:0 },{ t:"05:00",d:1 },{ t:"06:00",d:2 },{ t:"07:00",d:1 },
  { t:"08:00",d:3 },{ t:"09:00",d:2 },{ t:"10:00",d:1 },{ t:"11:00",d:0 },
  { t:"12:00",d:1 },{ t:"13:00",d:3 },{ t:"14:00",d:2 },{ t:"15:00",d:0 },
];

export const HEALTH_CHART = [
  { t:"08:00",on:5,off:1,warn:0 },{ t:"09:00",on:5,off:1,warn:1 },
  { t:"10:00",on:5,off:1,warn:1 },{ t:"11:00",on:6,off:0,warn:0 },
  { t:"12:00",on:5,off:1,warn:1 },{ t:"13:00",on:4,off:1,warn:2 },
  { t:"14:00",on:4,off:1,warn:1 },
];

export const DETECTION_DIST = [
  { name:"SAFE",           value:847, color:"#28734A" },
  { name:"CAUTION",        value:43,  color:"#D99A27" },
  { name:"NARCOTIC PROXY", value:12,  color:"#C49A4A" },
  { name:"EXPLOSIVE PROXY",value:4,   color:"#B3262E" },
];
