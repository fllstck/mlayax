# Third-party licence texts

These are the **full texts** of the licences covering the components we redistribute. `../NOTICE`
summarises which component is under which licence; these files are the authoritative copies.

`scripts/sync-licenses.mjs` copies this directory into `packages/*/licenses/`, and `npm run verify`
fails if a copy has drifted. Nothing here is generated or edited by hand — the texts are vendored
verbatim, because the build must not depend on network access to produce a compliant tarball.

| file | covers | where it came from |
|---|---|---|
| `MLX-MIT.txt` | `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib` | `deps/mlx/LICENSE` in the `ml-explore/mlx` source at tag `v0.32.3`. Byte-identical to the `LICENSE` in the `mlx` 0.32.3 and `mlx_metal` 0.32.3 Python wheels, which is what we actually link. |
| `node-mlx-MIT.txt` | `node_mlx.node`, `vendor/node-mlx/` | `LICENSE` in `frost-beta/node-mlx` at commit `4bf8b1d6de32ae5ec402525bff1f041d73618275`. |
| `Laya-Apache-2.0.txt` | the Laya model weights (not redistributed — downloaded at first use) | The canonical Apache-2.0 text from <https://www.apache.org/licenses/LICENSE-2.0.txt>. The upstream copy lives in the gated Hugging Face repository <https://huggingface.co/Convai-Innovations/laya>, which requires a token, so the licence is reproduced from the canonical source rather than fetched at build time. |

`MLX-MIT.txt` and `node-mlx-MIT.txt` differ by exactly one line: node-mlx adds
`Copyright © 2024 zcbenz` alongside Apple's notice. Keep them as two files — collapsing them would
misrepresent node-mlx's authorship.