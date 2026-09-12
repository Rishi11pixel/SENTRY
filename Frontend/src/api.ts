import type { Device, Incident, LogEntry, PredictionResult } from "./data";

const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`SENTRY API ${response.status}`);
  return response.json() as Promise<T>;
}

export type DashboardSummary = {
  connectedDevices: number;
  online: number;
  offline: number;
  activeAlerts: number;
  activeIncidents: Incident[];
};

export async function getDevices() {
  return (await get<{ devices: Device[] }>("/api/v1/devices")).devices;
}

export async function getDashboardSummary() {
  return get<DashboardSummary>("/api/v1/dashboard/summary");
}

export async function getIncidents() {
  return (await get<{ incidents: Incident[] }>("/api/v1/incidents")).incidents;
}

export async function getLogs() {
  return (await get<{ logs: LogEntry[] }>("/api/v1/logs?limit=100")).logs;
}

export async function getDevicePredictions(deviceId: string) {
  return (await get<{ predictions: PredictionResult[] }>(
    `/api/v1/devices/${encodeURIComponent(deviceId)}/predictions?limit=10`,
  )).predictions;
}

export async function acknowledgeIncident(id: string) {
  const response = await fetch(`${API_BASE}/api/v1/incidents/${encodeURIComponent(id)}/acknowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operator_id: "RAIL_ADM_001" }),
  });
  if (!response.ok) throw new Error(`SENTRY API ${response.status}`);
}

export async function resolveIncident(id: string) {
  const response = await fetch(`${API_BASE}/api/v1/incidents/${encodeURIComponent(id)}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operator_id: "RAIL_ADM_001" }),
  });
  if (!response.ok) throw new Error(`SENTRY API ${response.status}`);
}