"use strict";
/**
 * node-mlx's JavaScript core layer, vendored from `@frost-beta/mlx` 0.4.0.
 *
 * Upstream file: `node_modules/@frost-beta/mlx/dist/core.js`
 * Upstream commit: frost-beta/node-mlx @ 4bf8b1d6de32ae5ec402525bff1f041d73618275
 * Licence: MIT (see ./LICENSE)
 *
 * ONE line differs from upstream — the `require` that loads the native addon. Upstream hardcodes
 * `../build/Release/node_mlx.node`, which assumes the addon was built in place by an install
 * script. We resolve it from the platform package instead. See ./PATCHES.md.
 *
 * Because that resolution is indirect (`require(resolver())` rather than a literal path), Node's
 * `require` cache is still keyed by the resolved absolute path, so loading the addon twice from two
 * different specifiers still yields one instance.
 *
 * Only this file is vendored, not upstream's whole `dist/`: `dist/nn/`, `dist/optimizers/` and
 * `dist/utils.js` are ~400 KB that the runtime never imports. See ./README.md.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.mx = exports.core = void 0;
// PATCHED: was require("../build/Release/node_mlx.node")
const node_mlx_node_1 = __importDefault(
  require(require("./native-binding.cjs").resolveNativeBinding()),
);
exports.core = node_mlx_node_1.default;
exports.mx = node_mlx_node_1.default;
// Helper for creating complex number.
node_mlx_node_1.default.Complex = (re, im) => {
    return { re, im: im ?? 0 };
};
// Implementation of the StreamContext.
node_mlx_node_1.default.stream = function stream(s) {
    const old = node_mlx_node_1.default.defaultStream(node_mlx_node_1.default.defaultDevice());
    const target = node_mlx_node_1.default.toStream(s);
    node_mlx_node_1.default.setDefaultDevice(target.device);
    node_mlx_node_1.default.setDefaultStream(target);
    return {
        [Symbol.dispose]() {
            node_mlx_node_1.default.setDefaultDevice(old.device);
            node_mlx_node_1.default.setDefaultStream(old);
        },
    };
};