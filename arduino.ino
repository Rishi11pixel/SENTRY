#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ---------------- Configuration ----------------
const char* WIFI_SSID     = "Om 2.4g";
const char* WIFI_PASSWORD = "7291849609";
const char* BACKEND_HOST  = "192.168.1.67";
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

// ---------------- Timing ----------------
const unsigned long GAS_READ_INTERVAL_MS = 100;
const unsigned long DHT_READ_INTERVAL_MS = 2000;
const unsigned long WIFI_RETRY_MS        = 5000;
const unsigned long HTTP_TIMEOUT_MS       = 1500;
const unsigned long STATUS_LOG_INTERVAL_MS = 1000;

// ---------------- Objects ----------------
DHT dht(DHTPIN, DHTTYPE);
WiFiClient wifiClient;
HTTPClient httpClient;

// ---------------- State ----------------
float humidity = 0.0f;
float temperature = 0.0f;
bool hasValidDhtReading = false;

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

String buildBackendUrl() {
  return String("http://") + BACKEND_HOST + ":" + String(BACKEND_PORT) +
         "/api/v1/devices/" + DEVICE_ID + "/readings";
}

int computeSignalPercent() {
  if (WiFi.status() != WL_CONNECTED) {
    return 0;
  }

  int rssi = WiFi.RSSI();
  return constrain(map(rssi, -100, -30, 0, 100), 0, 100);
}

void setConnectionStatusLeds() {
  if (WiFi.status() != WL_CONNECTED) {
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
    return;
  }

  if (millis() - lastHttpSuccessTime < 5000) {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
  } else {
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
  }
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

void readDhtOnce() {
  float newHumidity = dht.readHumidity();
  float newTemperature = dht.readTemperature();

  if (isnan(newHumidity) || isnan(newTemperature)) {
    Serial.println("DHT22 read failed; keeping last valid values.");
    return;
  }

  humidity = newHumidity;
  temperature = newTemperature;
  hasValidDhtReading = true;
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
    Serial.println("WiFi not connected; skipping HTTP POST.");
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

  Serial.print("HTTP ");
  Serial.print(httpCode);
  Serial.print(" | ");
  Serial.println(url);

  if (ok) {
    lastHttpSuccessTime = millis();
    Serial.println("Backend accepted reading.");
  } else {
    Serial.print("HTTP POST failed; continuing sensor loop. SEN0567 voltage=");
    Serial.println(sample.sen0567, 4);
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

    if (millis() - lastStatusLogTime >= STATUS_LOG_INTERVAL_MS) {
      lastStatusLogTime = millis();
      Serial.print("Temp: ");
      Serial.print(sample.temperature);
      Serial.print("C  Hum: ");
      Serial.print(sample.humidity);
      Serial.print("%  MQ2: ");
      Serial.print(sample.mq2);
      Serial.print("  MQ3: ");
      Serial.print(sample.mq3);
      Serial.print("  MQ135: ");
      Serial.print(sample.mq135);
      Serial.print("  SEN0567: ");
      Serial.print(sample.sen0567, 4);
      Serial.print(" V");
      Serial.print("  Signal: ");
      Serial.print(computeSignalPercent());
      Serial.println("%");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);

  dht.begin();
  analogSetPinAttenuation(SEN0567_PIN, ADC_11db);

  readDhtOnce();
  while (!hasValidDhtReading) {
    readDhtOnce();
    delay(250);
  }
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
    1
  );

  Serial.println("=== SENTRY ESP32 Final Firmware ===");
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
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }

  if (millis() - lastDhtReadTime >= DHT_READ_INTERVAL_MS) {
    refreshDhtIfNeeded();
  }

  SensorSample sample;
  if (xQueueReceive(sensorQueue, &sample, 0) == pdTRUE) {
    sendReading(sample);
  }

  setConnectionStatusLeds();
  delay(10);
}