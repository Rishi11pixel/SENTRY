#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ---------------- Configuration ----------------
const char* WIFI_SSID     = "Om 2.4g";
const char* WIFI_PASSWORD = "7291849609";
const char* BACKEND_HOST  = "192.168.1.66";
const char* DEVICE_ID     = "SENTRY-001";

const uint16_t BACKEND_PORT = 8000;

// ---------------- Pin Definitions ----------------
#define DHTPIN      27
#define DHTTYPE     DHT22

#define MQ2_PIN     35
#define MQ3_PIN     32
#define MQ135_PIN   34
#define SEN0567_PIN 33

#define RED_LED_PIN    13
#define GREEN_LED_PIN  14

/*
 * Buzzer GPIO is intentionally not defined here.
 * The current sketch has no buzzer pin and no hardware wiring information was supplied,
 * so we are not guessing a pin value. The buzzer remains disabled until a real GPIO
 * is identified for the device.
 */

// ---------------- Timing ----------------
const unsigned long GAS_READ_INTERVAL_MS = 100;
const unsigned long DHT_READ_INTERVAL_MS = 2000;
const unsigned long WIFI_RETRY_MS        = 5000;
const unsigned long HTTP_TIMEOUT_MS       = 1500;
const unsigned long STATUS_LOG_INTERVAL_MS = 1000;
const unsigned long GAS_WARMUP_MS = 60000; // MQ sensors need this before readings are trustworthy

// ---------------- Objects ----------------
DHT dht(DHTPIN, DHTTYPE);
WiFiClient wifiClient;
HTTPClient httpClient;

// ---------------- State ----------------
float humidity = 0.0f;
float temperature = 0.0f;
bool hasValidDhtReading = false;

// Diagnostic: tracks the outcome of the most recent DHT attempt for this cycle's log line
bool lastDhtAttemptOk = true;
int  lastDhtAttemptsUsed = 0;

unsigned long lastDhtReadTime = 0;
unsigned long lastHttpSuccessTime = 0;
unsigned long lastStatusLogTime = 0;

struct SensorSample {
  float temperature = 0.0f;
  float humidity = 0.0f;
  int mq2 = 0;
  int mq3 = 0;
  int mq135 = 0;
  float sen0567 = 0.0f;
};

QueueHandle_t sensorQueue;
String latestSystemStatus = "NON-THREAT";

String buildBackendUrl() {
  return String("http://") + BACKEND_HOST + ":" + String(BACKEND_PORT) +
         "/api/v1/devices/" + DEVICE_ID + "/readings";
}

String extractJsonStringValue(const String& json, const String& key) {
  String needle = "\"" + key + "\"";
  int keyIndex = json.indexOf(needle);
  if (keyIndex < 0) {
    return "";
  }

  int colonIndex = json.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) {
    return "";
  }

  int valueStart = json.indexOf('"', colonIndex + 1);
  if (valueStart < 0) {
    return "";
  }

  int valueEnd = json.indexOf('"', valueStart + 1);
  if (valueEnd < 0) {
    return "";
  }

  return json.substring(valueStart + 1, valueEnd);
}

void applyBackendSafetyState(const String& systemStatus) {
  bool isAlert = (systemStatus == "THREAT" || systemStatus == "ALERT");

  digitalWrite(RED_LED_PIN, isAlert ? HIGH : LOW);
  digitalWrite(GREEN_LED_PIN, isAlert ? LOW : HIGH);

#ifdef BUZZER_PIN
  digitalWrite(BUZZER_PIN, isAlert ? HIGH : LOW);
#endif

  if (isAlert) {
    Serial.println("SYSTEM ALERT: RED ON, GREEN OFF, BUZZER ON");
  } else {
    Serial.println("SYSTEM SAFE: GREEN ON, RED OFF, BUZZER OFF");
  }
}

bool updateSystemStatusFromBackendResponse(const String& responseBody) {
  String parsedStatus = extractJsonStringValue(responseBody, "status");
  if (parsedStatus.length() == 0) {
    return false;
  }

  if (parsedStatus == "THREAT" || parsedStatus == "ALERT") {
    latestSystemStatus = "THREAT";
    applyBackendSafetyState(latestSystemStatus);
    return true;
  }

  if (parsedStatus == "NON-THREAT" || parsedStatus == "SAFE") {
    latestSystemStatus = "NON-THREAT";
    applyBackendSafetyState(latestSystemStatus);
    return true;
  }

  return false;
}

int computeSignalPercent() {
  if (WiFi.status() != WL_CONNECTED) {
    return 0;
  }

  int rssi = WiFi.RSSI();
  return constrain(map(rssi, -100, -30, 0, 100), 0, 100);
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(250);
    Serial.print('.');
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("WiFi connected: ");
    Serial.println(WIFI_SSID);
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi connection failed.");
  }
}

// Single attempt only - the DHT22 needs >=2s between any two reads,
// so retrying rapidly inside this function was causing guaranteed failures.
// Pacing is handled by the caller (2s minimum between calls to this function).
bool readDhtOnce() {
  float newHumidity = dht.readHumidity();
  float newTemperature = dht.readTemperature();

  if (!isnan(newHumidity) && !isnan(newTemperature)) {
    humidity = newHumidity;
    temperature = newTemperature;
    hasValidDhtReading = true;
    lastDhtAttemptOk = true;
    lastDhtAttemptsUsed = 1;
    return true;
  }
  lastDhtAttemptOk = false;
  lastDhtAttemptsUsed = 1;
  return false;
}

void refreshDhtIfNeeded() {
  unsigned long now = millis();
  if (now - lastDhtReadTime < DHT_READ_INTERVAL_MS) {
    return;
  }

  lastDhtReadTime = now;
  readDhtOnce();
}

bool sendReading(const SensorSample& sample) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  String payload = "{";
  payload += "\"temperature\":" + String(sample.temperature, 2) + ",";
  payload += "\"humidity\":" + String(sample.humidity, 2) + ",";
  payload += "\"mq2\":" + String(sample.mq2) + ",";
  payload += "\"mq3\":" + String(sample.mq3) + ",";
  payload += "\"mq135\":" + String(sample.mq135) + ",";
  payload += "\"sen0567\":" + String(sample.sen0567, 4) + ",";
  payload += "\"signal\":" + String(computeSignalPercent());
  payload += "}";

  String url = buildBackendUrl();
  httpClient.begin(wifiClient, url);
  httpClient.setTimeout(HTTP_TIMEOUT_MS);
  httpClient.addHeader("Content-Type", "application/json");

  int httpCode = httpClient.POST(payload);
  bool ok = (httpCode >= 200 && httpCode < 300);

  if (ok) {
    lastHttpSuccessTime = millis();
    String responseBody = httpClient.getString();
    if (responseBody.length() > 0) {
      updateSystemStatusFromBackendResponse(responseBody);
    }
  }

  httpClient.end();
  return ok;
}

void sensorAcquisitionTask(void* parameter) {
  TickType_t lastWake = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(GAS_READ_INTERVAL_MS);

  while (true) {
    vTaskDelayUntil(&lastWake, period);

    SensorSample sample;
    sample.temperature = hasValidDhtReading ? temperature : 0.0f;
    sample.humidity = hasValidDhtReading ? humidity : 0.0f;
    sample.mq2 = analogRead(MQ2_PIN);
    sample.mq3 = analogRead(MQ3_PIN);
    sample.mq135 = analogRead(MQ135_PIN);
    sample.sen0567 = analogReadMilliVolts(SEN0567_PIN) / 1000.0f;

    if (xQueueSend(sensorQueue, &sample, 0) != pdTRUE) {
      SensorSample dropped;
      xQueueReceive(sensorQueue, &dropped, 0);
      xQueueSend(sensorQueue, &sample, 0);
    }

    // ---------- Sensor diagnostics only; no local classification ----------
    if (millis() - lastStatusLogTime >= STATUS_LOG_INTERVAL_MS) {
      lastStatusLogTime = millis();

      Serial.println("------------------------------------------");

      if (hasValidDhtReading) {
        Serial.print("DHT22: ");
        Serial.print(sample.temperature);
        Serial.print(" C  ");
        Serial.print(sample.humidity);
        Serial.println(" %");
      } else {
        Serial.println("DHT22: Reading...");
      }

      Serial.print("MQ-2:    "); Serial.println(sample.mq2);
      Serial.print("MQ-3:    "); Serial.println(sample.mq3);
      Serial.print("MQ-135:  "); Serial.println(sample.mq135);
      Serial.print("SEN0567: "); Serial.print(sample.sen0567, 4); Serial.println(" V");
      Serial.print("WiFi:    ");
      Serial.print(WiFi.status() == WL_CONNECTED ? "connected" : "disconnected");
      Serial.print("  Signal: "); Serial.print(computeSignalPercent()); Serial.println("%");
      Serial.print("HTTP:    ");
      Serial.println(lastHttpSuccessTime > 0 ? "success" : "pending");

      Serial.println("------------------------------------------");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);

#ifdef BUZZER_PIN
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
#endif

  Serial.println("=== SENTRY ESP32 Diagnostic Firmware ===");
  Serial.println("Booting...");

  dht.begin();
  analogSetPinAttenuation(SEN0567_PIN, ADC_11db);

  // No blocking "wait for first valid reading" here - matches the approach that
  // worked reliably before. The 60s warmup below gives the DHT22 plenty of time
  // to stabilize, and the main loop just retries every 2s from then on, same as
  // the original working sketch.

  // ---------- Gas sensor warmup ----------
  // MQ-series sensors give unstable/inflated readings until their heater
  // element reaches operating temperature. No LED/buzzer behavior is tied to
  // this - it's purely a delay so garbage readings aren't sent to the backend
  // during the first ~60s. Keep sensors in clean air during this time.
  Serial.print("Warming up gas sensors for ");
  Serial.print(GAS_WARMUP_MS / 1000);
  Serial.println("s. Keep sensors in clean air.");
  unsigned long warmupStart = millis();
  while (millis() - warmupStart < GAS_WARMUP_MS) {
    delay(250);
  }
  Serial.println("Gas sensor warmup complete.");

  lastDhtReadTime = millis();

  sensorQueue = xQueueCreate(64, sizeof(SensorSample));
  if (sensorQueue == NULL) {
    Serial.println("Failed to create sensor queue.");
    while (true) {
      delay(1000);
    }
  }

  xTaskCreatePinnedToCore(
    sensorAcquisitionTask,
    "sensorAcq",
    4096,
    NULL,
    0,
    NULL,
    0
  );

  Serial.print("Device ID: ");
  Serial.println(DEVICE_ID);
  Serial.print("Backend: http://");
  Serial.print(BACKEND_HOST);
  Serial.print(":");
  Serial.print(BACKEND_PORT);
  Serial.print("/api/v1/devices/");
  Serial.print(DEVICE_ID);
  Serial.println("/readings");

  connectWifi();
  WiFi.setSleep(false);
  applyBackendSafetyState(latestSystemStatus);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
    WiFi.setSleep(false);
  }

  refreshDhtIfNeeded();

  SensorSample sample;
  if (xQueueReceive(sensorQueue, &sample, 0) == pdTRUE) {
    sendReading(sample);
  }

  delay(10);
}
