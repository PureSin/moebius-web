/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL the .onnx model files are fetched from. Default "/models" (dev symlink).
   *  For deployment, set to e.g. "https://huggingface.co/<user>/Moebius-ONNX/resolve/main". */
  readonly VITE_MODEL_BASE?: string;
  /** Base URL for the onnxruntime-web wasm/glue assets. Default "/ort/". */
  readonly VITE_ORT_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
