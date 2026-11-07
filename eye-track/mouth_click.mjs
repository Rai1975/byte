import {
  SmartSpectraSDK,
  CameraSelection,
  faceMetrics,
  decodeMetrics,
} from "@smartspectra/node-sdk";

// --- Configuration ---
const API_KEY = process.env.PRESAGE_API_KEY || "YOUR_API_KEY";
const COOLDOWN_MS = 400; // Minimum time between clicks

// --- State ---
let lastClickTime = 0;
let lastBlinkState = false;

// --- Mouse click helper using child_process to call Python/pynput ---
import { execSync } from "node:child_process";

function click(button) {
  const now = Date.now();
  if (now - lastClickTime < COOLDOWN_MS) return;
  lastClickTime = now;

  console.log(`${button.toUpperCase()} CLICK`);
  try {
    execSync(`python -c "from pynput.mouse import Button,Controller;Controller().click(Button.${button})"`, { stdio: "ignore" });
  } catch (e) {
    console.error("Click failed:", e.message);
  }
}

// --- SDK setup ---
const sdk = new SmartSpectraSDK({
  apiKey: API_KEY,
  requestedMetrics: [...faceMetrics],
});

sdk.on("metrics", (buf, ts) => {
  const data = decodeMetrics(buf);

  // --- Blink-based click ---
  // Detect a completed blink (was closed, now open) to trigger a click.
  if (data.blinking !== undefined) {
    const isBlinking = Boolean(data.blinking);
    if (lastBlinkState && !isBlinking) {
      // Blink just ended — fire left click
      click("left");
    }
    lastBlinkState = isBlinking;
  }

  // --- Expression-based click ---
  // If the SDK provides expression labels, you can map them here.
  // Uncomment and adjust once you see what expressions the SDK returns:
  //
  // if (data.expressions) {
  //   console.log("Expressions:", data.expressions);
  //   // Example: map a specific expression to right-click
  //   // if (data.expressions.jawClench > 0.5) click("right");
  // }

  // Uncomment to inspect raw decoded data during development:
  // console.log("Raw metrics:", JSON.stringify(data, null, 2));
});

sdk.on("processingStatus", (status) => {
  console.log("Status:", status);
});

sdk.on("validationStatus", (codes) => {
  // Hints like "hold still", "move closer", etc.
  if (codes.length > 0) console.log("Validation:", codes);
});

sdk.on("error", (code, message, retryable) => {
  console.error(`Error [${code}]: ${message} (retryable: ${retryable})`);
});

// --- Start ---
console.log("Starting Presage SmartSpectra...");
console.log("Blink to left-click. Press Ctrl+C to quit.");
console.log(
  "Tip: uncomment the raw metrics log line to see what data is available."
);

sdk.useCamera(CameraSelection.default);
sdk.start();
