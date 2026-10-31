// AIワーカーとのやりとり（読み込み・実行をそれぞれ Promise にする）

export class AiWorker {
  constructor() {
    this.worker = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = ({ data: m }) => {
      const p = this.pending.get(m.id);
      if (!p) return;
      if (m.type === 'progress') { p.onProgress?.(m.got, m.total); return; }
      this.pending.delete(m.id);
      if (m.type === 'error') p.rej(new Error(m.message));
      else p.res(m);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.rej(new Error(e.message || 'AIの処理に失敗しました'));
      this.pending.clear();
    };
  }

  #call(msg, transfer = [], onProgress) {
    return new Promise((res, rej) => {
      const id = this.nextId++;
      this.pending.set(id, { res, rej, onProgress });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  /** モデルを読み込む。返り値 { backend, gpuLimit } */
  load(name, urls, { preferWasm = false, totalHint = 0, onProgress, debug } = {}) {
    return this.#call({ type: 'load', name, urls, preferWasm, totalHint, debug }, [], onProgress);
  }

  /** 実行する。feeds: { 名前: { data: Float32Array, dims } }。返り値 { 出力名: { data, dims } } */
  async run(name, feeds) {
    const transfer = Object.values(feeds).map((f) => f.data.buffer);
    return (await this.#call({ type: 'run', name, feeds }, transfer)).outputs;
  }

  terminate() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.rej(new Error('中止しました'));
    this.pending.clear();
  }
}
