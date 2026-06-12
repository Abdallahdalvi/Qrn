import { TarteelProvider } from "./core/recognition/providers/TarteelProvider";
import type { RecitationResult } from "./core/recognition/RecitationRecognizer";

// Define the global window interface
declare global {
  interface Window {
    initEngine: (assetsUrl: string) => Promise<void>;
    processAudioChunk: (audioBase64: string) => void;
    endAudioStream: () => void;
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

import * as ort from "onnxruntime-web";

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

    provider = new TarteelProvider({
      loadVocab: async () => fetchAsset("phoneme_vocab.json").then(buf => JSON.parse(new TextDecoder().decode(buf))),
      loadQuranData: async () => fetchAsset("quran_phonemes.json").then(buf => JSON.parse(new TextDecoder().decode(buf))),
      loadModel: async () => fetchAsset("fastconformer_phoneme_q8.onnx"),
      getOrtSessionClass: () => ort.InferenceSession,
      getOrtTensorClass: () => ort.Tensor,
    });

    await provider.initialize();

    // Subscribe to events
    provider.onResult((result: RecitationResult) => {
      sendToFlutter("verse_match", result);
    });

    // We can also tap into the underlying engine's word_progress
    if ((provider as any).engine) {
      (provider as any).engine.onMessage((msg: any) => {
        if (msg.type === "word_progress") {
          sendToFlutter("word_progress", msg);
        } else if (msg.type === "loading" || msg.type === "loading_status") {
          sendToFlutter("status", msg);
        }
      });
    }

    sendToFlutter("ready", { status: "success" });
  } catch (error: any) {
    console.error("Initialization error:", error);
    sendToFlutter("error", { message: error.message || String(error) });
  }
};

window.processAudioChunk = (audioBase64: string) => {
  if (!provider) return;
  // Convert Base64 back to Float32Array
  const binaryString = atob(audioBase64);
  const len = binaryString.length;
  // Since each sample is 4 bytes (Float32), length of float array = len / 4
  if (len % 4 !== 0) {
    console.error("Invalid audio chunk length");
    return;
  }
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const floatData = new Float32Array(bytes.buffer);
  
  provider.processAudioChunk(floatData);
};

window.endAudioStream = () => {
  if (!provider) return;
  provider.endAudioStream();
};
