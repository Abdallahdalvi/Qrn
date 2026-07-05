export const env = {
  backends: {
    onnx: {},
  },
};

export class Tensor {
  constructor() {
    throw new Error('onnxruntime-node is not available in the browser bundle.');
  }
}

export class InferenceSession {
  static async create() {
    throw new Error('onnxruntime-node is not available in the browser bundle.');
  }
}

export default {
  env,
  Tensor,
  InferenceSession,
};
