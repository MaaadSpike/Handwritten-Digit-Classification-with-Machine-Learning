// Plain-JavaScript inference for the exported CNNs (no ML framework).
// Layer list format is produced by digits/export_web.py (BatchNorm already folded into conv).
(function (root) {
  "use strict";

  function b64ToBytes(b64) {
    if (typeof atob === "function") {
      const s = atob(b64);
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64")); // Node
  }

  function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeF16(b64) {
    const bytes = b64ToBytes(b64);
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
  }

  function decodeF32(b64) {
    const bytes = b64ToBytes(b64);
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }

  function conv3x3Relu(x, c, h, w, W, B, cout) {
    const out = new Float32Array(cout * h * w);
    for (let o = 0; o < cout; o++) {
      const ob = o * h * w;
      for (let i = 0; i < h * w; i++) out[ob + i] = B[o];
      for (let ci = 0; ci < c; ci++) {
        const xb = ci * h * w;
        const wb = (o * c + ci) * 9;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const k = W[wb + ky * 3 + kx];
            if (k === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            const y0 = Math.max(0, -dy), y1 = Math.min(h, h - dy);
            const x0 = Math.max(0, -dx), x1 = Math.min(w, w - dx);
            for (let yy = y0; yy < y1; yy++) {
              const orow = ob + yy * w;
              const irow = xb + (yy + dy) * w + dx;
              for (let xx = x0; xx < x1; xx++) out[orow + xx] += k * x[irow + xx];
            }
          }
        }
      }
      for (let i = 0; i < h * w; i++) if (out[ob + i] < 0) out[ob + i] = 0;
    }
    return out;
  }

  function maxpool2(x, c, h, w) {
    const h2 = h >> 1, w2 = w >> 1;
    const out = new Float32Array(c * h2 * w2);
    for (let ci = 0; ci < c; ci++) {
      for (let y = 0; y < h2; y++) {
        for (let xx = 0; xx < w2; xx++) {
          const b = ci * h * w + 2 * y * w + 2 * xx;
          out[ci * h2 * w2 + y * w2 + xx] = Math.max(x[b], x[b + 1], x[b + w], x[b + w + 1]);
        }
      }
    }
    return out;
  }

  function linear(x, W, B, din, dout, relu) {
    const out = new Float32Array(dout);
    for (let o = 0; o < dout; o++) {
      let s = B[o];
      const wb = o * din;
      for (let i = 0; i < din; i++) s += W[wb + i] * x[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  function load(layers) {
    return layers.map(function (L) {
      const l = Object.assign({}, L);
      if (L.w) l.W = decodeF16(L.w);
      if (L.b) l.B = decodeF32(L.b);
      return l;
    });
  }

  // pixels: Float32Array(784) in [0, 1], row-major 28x28. Returns logits (Float32Array(10)).
  function forward(net, pixels) {
    let x = Float32Array.from(pixels);
    let c = 1, h = 28, w = 28;
    for (const L of net) {
      if (L.type === "norm") {
        for (let i = 0; i < x.length; i++) x[i] = (x[i] - L.mean) / L.std;
      } else if (L.type === "conv3x3_relu") {
        x = conv3x3Relu(x, c, h, w, L.W, L.B, L.cout);
        c = L.cout;
      } else if (L.type === "maxpool2") {
        x = maxpool2(x, c, h, w);
        h >>= 1; w >>= 1;
      } else if (L.type === "linear") {
        x = linear(x, L.W, L.B, L.din, L.dout, L.relu);
      }
    }
    return x;
  }

  function softmax(z) {
    const m = Math.max.apply(null, Array.from(z));
    const e = Array.from(z, function (v) { return Math.exp(v - m); });
    const s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }

  const api = { load: load, forward: forward, softmax: softmax, b64ToBytes: b64ToBytes };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DigitNet = api;
})(typeof window !== "undefined" ? window : this);
