import { TarteelProvider } from "./core/recognition/providers/TarteelProvider";
import type { RecognitionResult } from "./core/recognition/RecitationRecognizer";

// Define the global window interface
declare global {
  interface Window {
    initEngine: (assetsUrl: string) => Promise<void>;
    processAudioChunk: (audioBase64: string) => void;
    endAudioStream: () => void;
    resetAudioStream: () => void;
    startTaraweeh: (surah: number, ayah: number) => void;
    stopTaraweeh: () => void;
    flutter_inappwebview?: {
      callHandler: (handlerName: string, ...args: any[]) => void;
    };
  }
}

let provider: TarteelProvider | null = null;

/**
 * Sends an event to Flutter via the InAppWebView callHandler.
 */
function sendToFlutter(eventName: string, data: any) {
  if (window.flutter_inappwebview?.callHandler) {
    window.flutter_inappwebview.callHandler('onEngineEvent', { type: eventName, ...data });
  } else {
    console.log(`[Engine -> Flutter] ${eventName}:`, data);
  }
}

import * as ort from "onnxruntime-web/wasm";

window.initEngine = async (assetsUrl: string) => {
  try {
    sendToFlutter("status", { message: "Initializing engine..." });
    
    // We implement a basic fetch-based AssetLoader for the WebView
    const fetchAsset = async (path: string) => {
      const response = await fetch(`${assetsUrl}/${path}`);
      if (!response.ok) throw new Error(`Failed to load asset: ${path}`);
      return await response.arrayBuffer();
    };

    // Configure ONNX Runtime to load WASM from our assetsUrl
    ort.env.wasm.wasmPaths = `${assetsUrl}/wasm/`;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    sendToFlutter("status", { message: "Loading Quran/vocab/model assets..." });

    provider = new TarteelProvider({
      loadVocab: async () => fetchAsset("phoneme_vocab.json").then(buf => JSON.parse(new TextDecoder().decode(buf))),
      loadQuranData: async () => fetchAsset("quran_phonemes.json").then(buf => JSON.parse(new TextDecoder().decode(buf))),
      loadModel: async () => fetchAsset("fastconformer_phoneme_q8.onnx"),
      getOrtSessionClass: () => ort.InferenceSession,
      getOrtTensorClass: () => ort.Tensor,
    });

    provider.onMessage((msg) => {
      if (
        msg.type === "verse_match" ||
        msg.type === "word_progress" ||
        msg.type === "word_correction" ||
        msg.type === "raw_transcript" ||
        msg.type === "verse_candidate" ||
        msg.type === "final_sequence"
      ) {
        sendToFlutter(msg.type, msg);
      } else if (msg.type === "loading" || msg.type === "loading_status") {
        sendToFlutter("status", msg);
      }
    });

    sendToFlutter("status", { message: "Creating on-device recognition session..." });
    await provider.initialize();
    sendToFlutter("status", { message: "On-device recognition session ready." });

    // Subscribe to events
    provider.onResult((_result: RecognitionResult) => {
      // The richer tracker protocol is forwarded through onMessage below.
    });

    sendToFlutter("ready", { status: "success" });
  } catch (error: any) {
    console.error("Initialization error:", error);
    sendToFlutter("error", { message: error.message || String(error) });
  }
};

window.processAudioChunk = (audioBase64: string) => {
  if (!provider) {
    sendToFlutter("status", { message: "Audio received before engine provider was ready." });
    return;
  }
  try {
    // Convert Base64 back to Float32Array
    const binaryString = atob(audioBase64);
    const len = binaryString.length;
    // Since each sample is 4 bytes (Float32), length of float array = len / 4
    if (len % 4 !== 0) {
      throw new Error(`Invalid audio chunk length: ${len}`);
    }
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const floatData = new Float32Array(bytes.buffer);

    provider.processAudioChunk(floatData).catch((error: any) => {
      console.error("Audio processing error:", error);
      sendToFlutter("error", { message: error?.message || String(error) });
    });
  } catch (error: any) {
    console.error("Audio chunk decode error:", error);
    sendToFlutter("error", { message: error?.message || String(error) });
  }
};

window.endAudioStream = () => {
  if (!provider) return;
  provider.endAudioStream();
};

window.resetAudioStream = () => {
  if (!provider) return;
  provider.resetAudioStream();
};

window.startTaraweeh = (surah: number, ayah: number) => {
  if (!provider) {
    sendToFlutter("status", { message: "Taraweeh lock requested before engine provider was ready." });
    return;
  }
  provider.startTaraweeh(surah, ayah);
  sendToFlutter("status", {
    message: `Standalone Taraweeh lock active for ${surah}:${ayah}.`,
    tracking_mode: "TARAWIH_ENGINE_LOCK",
    search_window: `Surah ${surah} only`,
  });
};

window.stopTaraweeh = () => {
  if (!provider) return;
  provider.stopTaraweeh();
  sendToFlutter("status", { message: "Standalone Taraweeh lock cleared." });
};
