export type DeviceStatus = "ONLINE" | "OFFLINE" | "ALERT" | "WARNING";
export type ScanResult  = "SAFE" | "CAUTION" | "ALCOHOL" | "EXPLOSIVE PROXY" | "NARCOTIC PROXY";
export type LogLevel    = "ALERT" | "WARNING" | "SUCCESS" | "INFO";

export interface SensorReading {
  device_id: string;
  timestamp: string | number;
  temperature: number;
  humidity: number;
  mq2: number;
  mq3: number;
  mq135: number;
  sen0567: number;
  battery?: number;
  signal?: number;
  source_status?: string;
  test_object?: string;
}

export interface PredictionResult {
  id: string;
  window_id: string;
  timestamp: string;
  device_id: string;
  status: "SAFE" | "ALERT" | "ERROR";
  prediction: string | null;
  displayResult?: ScanResult;
  confidence: number;
  probabilities: Record<string, number>;
  features: Record<string, number>;
}

export interface Device {
  id: string;
  location: string;
  platform: string;
  status: DeviceStatus;
  battery: number | null;
  signal: number | null;
  health: number | null;
  lastSync: string | null;
  operatingTime: string | null;
  lastResult: ScanResult;
  confidence: number;
  mapX: number; // percent positions on SVG viewbox 0-100
  mapY: number;
  latestReading?: SensorReading | null;
  latestPrediction?: PredictionResult | null;
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
  latestReading?: SensorReading | null;
}

export const DETECTION_DIST = [
  { name:"SAFE",           value:847, color:"#20C878" },
  { name:"CAUTION",        value:43,  color:"#D99A27" },
  { name:"NARCOTIC PROXY", value:12,  color:"#B3262E" },
  { name:"EXPLOSIVE PROXY",value:4,   color:"#B3262E" },
];
