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

#define RED_LED_PIN    13
#define GREEN_LED_PIN  14

/*
 * Buzzer GPIO is intentionally not defined.
 * Add BUZZER_PIN only when the actual GPIO is confirmed.
 */

// ---------------- Timing ----------------
const unsigned long GAS_READ_INTERVAL_MS   = 100;
const unsigned long DHT_READ_INTERVAL_MS   = 2000;
const unsigned long HTTP_TIMEOUT_MS        = 1500;
const unsigned long STATUS_LOG_INTERVAL_MS = 1000;
const unsigned long GAS_WARMUP_MS          = 60000;
const unsigned int CALIBRATION_READINGS    = 30;

// ---------------- Objects ----------------
DHT dht(DHTPIN, DHTTYPE);
WiFiClient wifiClient;
HTTPClient httpClient;

// ---------------- State ----------------
float humidity = 0.0f;
float temperature = 0.0f;
bool hasValidDhtReading = false;
bool sensorPipelineReady = false;

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
};

QueueHandle_t sensorQueue;

String latestSystemStatus = "NON-THREAT";


// ============================================================
// SENSOR CALIBRATION
// ============================================================

void runSensorCalibration() {

  Serial.print(
    "Calibrating sensors with "
  );

  Serial.print(
    CALIBRATION_READINGS
  );

  Serial.println(
    " initial readings."
  );

  for (
    unsigned int readingNumber = 0;
    readingNumber < CALIBRATION_READINGS;
    readingNumber++
  ) {

    analogRead(MQ2_PIN);
    analogRead(MQ3_PIN);
    analogRead(MQ135_PIN);

    delay(GAS_READ_INTERVAL_MS);
  }

  sensorPipelineReady = true;

  Serial.println(
    "Sensor calibration complete."
  );
}


// ============================================================
// BACKEND URL
// ============================================================

String buildBackendUrl() {

  return String("http://") +
         BACKEND_HOST +
         ":" +
         String(BACKEND_PORT) +
         "/api/v1/devices/" +
         DEVICE_ID +
         "/readings";
}


// ============================================================
// JSON STRING EXTRACTION
// ============================================================

String extractJsonStringValue(
  const String& json,
  const String& key
) {

  String needle = "\"" + key + "\"";

  int keyIndex = json.indexOf(needle);

  if (keyIndex < 0) {
    return "";
  }

  int colonIndex =
    json.indexOf(
      ':',
      keyIndex + needle.length()
    );

  if (colonIndex < 0) {
    return "";
  }

  int valueStart =
    json.indexOf(
      '"',
      colonIndex + 1
    );

  if (valueStart < 0) {
    return "";
  }

  int valueEnd =
    json.indexOf(
      '"',
      valueStart + 1
    );

  if (valueEnd < 0) {
    return "";
  }

  return json.substring(
    valueStart + 1,
    valueEnd
  );
}


// ============================================================
// LED / SAFETY STATE
// ============================================================

void applyBackendSafetyState(
  const String& systemStatus
) {

  bool isAlert =
    (
      systemStatus == "THREAT" ||
      systemStatus == "ALERT"
    );

  digitalWrite(
    RED_LED_PIN,
    isAlert ? HIGH : LOW
  );

  digitalWrite(
    GREEN_LED_PIN,
    isAlert ? LOW : HIGH
  );

#ifdef BUZZER_PIN

  digitalWrite(
    BUZZER_PIN,
    isAlert ? HIGH : LOW
  );

#endif

  if (isAlert) {

    Serial.println(
      "SYSTEM ALERT: RED ON, GREEN OFF, BUZZER ON"
    );

  } else {

    Serial.println(
      "SYSTEM SAFE: GREEN ON, RED OFF, BUZZER OFF"
    );
  }
}


// ============================================================
// UPDATE STATUS FROM BACKEND
// ============================================================

bool updateSystemStatusFromBackendResponse(
  const String& responseBody
) {

  String parsedStatus =
    extractJsonStringValue(
      responseBody,
      "status"
    );

  if (parsedStatus.length() == 0) {
    return false;
  }

  if (
    parsedStatus == "THREAT" ||
    parsedStatus == "ALERT"
  ) {

    latestSystemStatus = "THREAT";

    applyBackendSafetyState(
      latestSystemStatus
    );

    return true;
  }

  if (
    parsedStatus == "NON-THREAT" ||
    parsedStatus == "SAFE"
  ) {

    latestSystemStatus = "NON-THREAT";

    applyBackendSafetyState(
      latestSystemStatus
    );

    return true;
  }

  return false;
}


// ============================================================
// WIFI SIGNAL
// ============================================================

int computeSignalPercent() {

  if (WiFi.status() != WL_CONNECTED) {
    return 0;
  }

  int rssi = WiFi.RSSI();

  return constrain(
    map(
      rssi,
      -100,
      -30,
      0,
      100
    ),
    0,
    100
  );
}


// ============================================================
// WIFI CONNECTION
// ============================================================

void connectWifi() {

  WiFi.mode(WIFI_STA);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  Serial.print(
    "Connecting to WiFi"
  );

  unsigned long start = millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - start < 15000
  ) {

    delay(250);

    Serial.print('.');
  }

  if (
    WiFi.status() == WL_CONNECTED
  ) {

    Serial.println();

    Serial.print(
      "WiFi connected: "
    );

    Serial.println(
      WIFI_SSID
    );

    Serial.print(
      "IP: "
    );

    Serial.println(
      WiFi.localIP()
    );

  } else {

    Serial.println();

    Serial.println(
      "WiFi connection failed."
    );
  }
}


// ============================================================
// DHT22
// ============================================================

bool readDhtOnce() {

  float newHumidity =
    dht.readHumidity();

  float newTemperature =
    dht.readTemperature();

  if (
    !isnan(newHumidity) &&
    !isnan(newTemperature)
  ) {

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


// ============================================================
// DHT REFRESH
// ============================================================

void refreshDhtIfNeeded() {

  unsigned long now = millis();

  if (
    now - lastDhtReadTime <
    DHT_READ_INTERVAL_MS
  ) {

    return;
  }

  lastDhtReadTime = now;

  readDhtOnce();
}


// ============================================================
// SEND READING TO BACKEND
// ============================================================

bool sendReading(
  const SensorSample& sample
) {

  if (!sensorPipelineReady) {
    return false;
  }

  if (
    WiFi.status() != WL_CONNECTED
  ) {

    return false;
  }

  String payload = "{";

  payload +=
    "\"temperature\":" +
    String(
      sample.temperature,
      2
    ) +
    ",";

  payload +=
    "\"humidity\":" +
    String(
      sample.humidity,
      2
    ) +
    ",";

  payload +=
    "\"mq2\":" +
    String(
      sample.mq2
    ) +
    ",";

  payload +=
    "\"mq3\":" +
    String(
      sample.mq3
    ) +
    ",";

  payload +=
    "\"mq135\":" +
    String(
      sample.mq135
    ) +
    ",";

  payload +=
    "\"signal\":" +
    String(
      computeSignalPercent()
    );

  payload +=
    ",\"source_status\":\"READY\"";

  payload += "}";


  String url =
    buildBackendUrl();


  httpClient.begin(
    wifiClient,
    url
  );

  httpClient.setTimeout(
    HTTP_TIMEOUT_MS
  );

  httpClient.addHeader(
    "Content-Type",
    "application/json"
  );


  int httpCode =
    httpClient.POST(
      payload
    );

  bool ok =
    (
      httpCode >= 200 &&
      httpCode < 300
    );


  if (ok) {

    lastHttpSuccessTime =
      millis();

    String responseBody =
      httpClient.getString();

    if (
      responseBody.length() > 0
    ) {

      updateSystemStatusFromBackendResponse(
        responseBody
      );
    }
  }


  httpClient.end();

  return ok;
}


// ============================================================
// SENSOR ACQUISITION TASK
// ============================================================

void sensorAcquisitionTask(
  void* parameter
) {

  TickType_t lastWake =
    xTaskGetTickCount();

  const TickType_t period =
    pdMS_TO_TICKS(
      GAS_READ_INTERVAL_MS
    );


  while (true) {

    vTaskDelayUntil(
      &lastWake,
      period
    );


    SensorSample sample;


    // --------------------------------------------------------
    // DHT22
    // --------------------------------------------------------

    sample.temperature =
      hasValidDhtReading
      ? temperature
      : 0.0f;

    sample.humidity =
      hasValidDhtReading
      ? humidity
      : 0.0f;


    // --------------------------------------------------------
    // MQ SENSORS
    // --------------------------------------------------------

    sample.mq2 =
      analogRead(
        MQ2_PIN
      );

    sample.mq3 =
      analogRead(
        MQ3_PIN
      );

    sample.mq135 =
      analogRead(
        MQ135_PIN
      );


    // --------------------------------------------------------
    // QUEUE
    // --------------------------------------------------------

    if (
      xQueueSend(
        sensorQueue,
        &sample,
        0
      ) != pdTRUE
    ) {

      SensorSample dropped;

      xQueueReceive(
        sensorQueue,
        &dropped,
        0
      );

      xQueueSend(
        sensorQueue,
        &sample,
        0
      );
    }


    // --------------------------------------------------------
    // SENSOR DIAGNOSTICS
    // --------------------------------------------------------

    if (
      millis() - lastStatusLogTime >=
      STATUS_LOG_INTERVAL_MS
    ) {

      lastStatusLogTime =
        millis();


      Serial.println(
        "------------------------------------------"
      );


      if (hasValidDhtReading) {

        Serial.print(
          "DHT22: "
        );

        Serial.print(
          sample.temperature
        );

        Serial.print(
          " C  "
        );

        Serial.print(
          sample.humidity
        );

        Serial.println(
          " %"
        );

      } else {

        Serial.println(
          "DHT22: Reading..."
        );
      }


      Serial.print(
        "MQ-2:    "
      );

      Serial.println(
        sample.mq2
      );


      Serial.print(
        "MQ-3:    "
      );

      Serial.println(
        sample.mq3
      );


      Serial.print(
        "MQ-135:  "
      );

      Serial.println(
        sample.mq135
      );


      Serial.print(
        "WiFi:    "
      );

      Serial.print(
        WiFi.status() == WL_CONNECTED
        ? "connected"
        : "disconnected"
      );

      Serial.print(
        "  Signal: "
      );

      Serial.print(
        computeSignalPercent()
      );

      Serial.println(
        "%"
      );


      Serial.print(
        "HTTP:    "
      );

      Serial.println(
        lastHttpSuccessTime > 0
        ? "success"
        : "pending"
      );


      Serial.println(
        "------------------------------------------"
      );
    }
  }
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(
    115200
  );

  delay(1000);


  // ----------------------------------------------------------
  // LED
  // ----------------------------------------------------------

  pinMode(
    RED_LED_PIN,
    OUTPUT
  );

  pinMode(
    GREEN_LED_PIN,
    OUTPUT
  );

  digitalWrite(
    RED_LED_PIN,
    LOW
  );

  digitalWrite(
    GREEN_LED_PIN,
    HIGH
  );


#ifdef BUZZER_PIN

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  digitalWrite(
    BUZZER_PIN,
    LOW
  );

#endif


  // ----------------------------------------------------------
  // BOOT
  // ----------------------------------------------------------

  Serial.println(
    "=== SENTRY ESP32 Firmware ==="
  );

  Serial.println(
    "Booting..."
  );


  // ----------------------------------------------------------
  // DHT
  // ----------------------------------------------------------

  dht.begin();


  // ----------------------------------------------------------
  // ADC ATTENUATION
  // ----------------------------------------------------------

  analogSetPinAttenuation(
    MQ2_PIN,
    ADC_11db
  );

  analogSetPinAttenuation(
    MQ3_PIN,
    ADC_11db
  );

  analogSetPinAttenuation(
    MQ135_PIN,
    ADC_11db
  );


  // ----------------------------------------------------------
  // GAS SENSOR WARMUP
  // ----------------------------------------------------------

  Serial.print(
    "Warming up gas sensors for "
  );

  Serial.print(
    GAS_WARMUP_MS / 1000
  );

  Serial.println(
    "s. Keep sensors in clean air."
  );


  unsigned long warmupStart =
    millis();


  while (
    millis() - warmupStart <
    GAS_WARMUP_MS
  ) {

    delay(250);
  }


  Serial.println(
    "Gas sensor warmup complete."
  );


  // ----------------------------------------------------------
  // SENSOR CALIBRATION
  // ----------------------------------------------------------

  runSensorCalibration();


  // ----------------------------------------------------------
  // DHT TIMING
  // ----------------------------------------------------------

  lastDhtReadTime =
    millis();


  // ----------------------------------------------------------
  // QUEUE
  // ----------------------------------------------------------

  sensorQueue =
    xQueueCreate(
      64,
      sizeof(SensorSample)
    );


  if (
    sensorQueue == NULL
  ) {

    Serial.println(
      "Failed to create sensor queue."
    );

    while (true) {
      delay(1000);
    }
  }


  // ----------------------------------------------------------
  // SENSOR TASK
  // ----------------------------------------------------------

  xTaskCreatePinnedToCore(
    sensorAcquisitionTask,
    "sensorAcq",
    4096,
    NULL,
    0,
    NULL,
    0
  );


  // ----------------------------------------------------------
  // BACKEND INFO
  // ----------------------------------------------------------

  Serial.print(
    "Device ID: "
  );

  Serial.println(
    DEVICE_ID
  );


  Serial.print(
    "Backend: http://"
  );

  Serial.print(
    BACKEND_HOST
  );

  Serial.print(
    ":"
  );

  Serial.print(
    BACKEND_PORT
  );

  Serial.print(
    "/api/v1/devices/"
  );

  Serial.print(
    DEVICE_ID
  );

  Serial.println(
    "/readings"
  );


  // ----------------------------------------------------------
  // WIFI
  // ----------------------------------------------------------

  connectWifi();

  WiFi.setSleep(false);


  // ----------------------------------------------------------
  // INITIAL STATE
  // ----------------------------------------------------------

  applyBackendSafetyState(
    latestSystemStatus
  );
}


// ============================================================
// LOOP
// ============================================================

void loop() {

  // ----------------------------------------------------------
  // WIFI RECONNECT
  // ----------------------------------------------------------

  if (
    WiFi.status() != WL_CONNECTED
  ) {

    connectWifi();

    WiFi.setSleep(false);
  }


  // ----------------------------------------------------------
  // DHT
  // ----------------------------------------------------------

  refreshDhtIfNeeded();


  // ----------------------------------------------------------
  // SEND SENSOR DATA
  // ----------------------------------------------------------

  SensorSample sample;


  if (
    xQueueReceive(
      sensorQueue,
      &sample,
      0
    ) == pdTRUE
  ) {

    sendReading(
      sample
    );
  }


  delay(10);
}