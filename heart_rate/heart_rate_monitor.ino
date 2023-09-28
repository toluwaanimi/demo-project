#include "Arduino.h"
#include <60ghzbreathheart.h>
#include <SoftwareSerial.h>

#define RX_Pin 2
#define TX_Pin 3

SoftwareSerial connectionSerial = SoftwareSerial(RX_Pin, TX_Pin);
BreathHeart_60GHz radar = BreathHeart_60GHz(&connectionSerial);

void setup() {
    initSerials();
    radar.ModeSelect_fuc(1);
    Serial.println("Radar sensor active!");
}

void loop() {
    radar.Breath_Heart();
    processSensorData();
    delay(200);  // Delay to prevent data flooding
}

void initSerials() {
    Serial.begin(115200);
    connectionSerial.begin(115200);
    while(!Serial);
}

void processSensorData() {
    switch(radar.sensor_report) {
        case HEARTRATEVAL:
            displayData("Heart rate", radar.heart_rate);
            break;
        case HEARTRATEWAVE:
            displayWaveData("Heart rate", radar.heart_point_1, radar.heart_point_2, radar.heart_point_3, radar.heart_point_4, radar.heart_point_5);
            break;
        case BREATHVAL:
            displayData("Breath rate", radar.breath_rate);
            break;
        case BREATHWAVE:
            displayWaveData("Breath rate", radar.breath_point_1, radar.breath_point_2, radar.breath_point_3, radar.breath_point_4, radar.breath_point_5);
            break;
        default:
            displaySensorStatus();
            break;
    }
}

// Display the sensor status messages
void displaySensorStatus() {
    static const char* messages[] = {
            "Sensor detects current breath rate is normal.",
            "Sensor detects current breath rate is too fast.",
            "Sensor detects current breath rate is too slow.",
            "There is no breathing information yet, please wait..."
    };

    if (radar.sensor_report >= BREATHNOR && radar.sensor_report <= BREATHNONE) {
        displayMessage(messages[radar.sensor_report - BREATHNOR]);
    }
}

// Standardized data display for sensor value
void displayData(const char* type, int value) {
    Serial.print(type);
    Serial.print(" value: ");
    Serial.print(value, DEC);
    if (strcmp(type, "Heart rate") == 0) {
        Serial.print(" bpm");
    } else if (strcmp(type, "Breath rate") == 0) {
        Serial.print(" brpm");
    }

     // Send the data to Raspberry Pi over the SoftwareSerial connection
    connectionSerial.print(type);
    connectionSerial.print(": ");
    connectionSerial.print(value);
    connectionSerial.println();
    Serial.println();
    Serial.println("----------------------------");
}


// Standardized data display for waveform data
void displayWaveData(const char* type, int p1, int p2, int p3, int p4, int p5) {
    Serial.print(type);
    Serial.print(" waveform(Sine wave) -- point 1: ");
    Serial.print(p1);
    Serial.print(", point 2: ");
    Serial.print(p2);
    Serial.print(", point 3: ");
    Serial.print(p3);
    Serial.print(", point 4: ");
    Serial.print(p4);
    Serial.print(", point 5: ");
    Serial.println(p5);
    Serial.println("----------------------------");
}

void displayMessage(const char *message) {
    Serial.println(message);
    Serial.println("----------------------------");
}
