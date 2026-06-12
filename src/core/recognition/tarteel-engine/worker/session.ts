import type { InferenceSession, Tensor } from "onnxruntime-common";

let session: InferenceSession | null = null;
let TensorClass: typeof Tensor | null = null;

export function setSession(ortSession: InferenceSession, ortTensor: typeof Tensor): void {
  session = ortSession;
  TensorClass = ortTensor;
}

export async function runInference(
  melFeatures: Float32Array,
  numMels: number,
  timeFrames: number,
): Promise<{ logprobs: Float32Array; timeSteps: number; vocabSize: number }> {
  if (!session || !TensorClass) throw new Error("Session or TensorClass not initialized");

  const inputTensor = new TensorClass("float32", melFeatures, [
    1,
    numMels,
    timeFrames,
  ]);
  const lengthTensor = new TensorClass(
    "int64",
    BigInt64Array.from([BigInt(timeFrames)]),
    [1],
  );

  const inputNames = session.inputNames;
  const feeds: Record<string, Tensor> = {
    [inputNames[0]]: inputTensor,
    [inputNames[1]]: lengthTensor,
  };

  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [_batch, timeSteps, vocabSize] = outputTensor.dims as number[];

  return {
    logprobs: outputTensor.data as Float32Array,
    timeSteps,
    vocabSize,
  };
}
