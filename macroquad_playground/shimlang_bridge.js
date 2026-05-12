// Miniquad plugin that lets the host page hand Shimlang source code to the
// running WASM. The page sets window.__shimlang_pending_source to a string
// and window.__shimlang_source_ready to true; the WASM polls each frame.
//
// Loaded after mq_js_bundle.js and before load("...wasm") in the host page.

(function () {
    let pending_encoded = null;

    const shimlang_bridge_plugin = {
        register_plugin: function (importObject) {
            importObject.env.shimlang_source_len = function () {
                if (!window.__shimlang_source_ready) {
                    return -1;
                }
                const src = window.__shimlang_pending_source;
                if (typeof src !== "string") {
                    return -1;
                }
                pending_encoded = new TextEncoder().encode(src);
                return pending_encoded.length;
            };

            importObject.env.shimlang_take_source = function (ptr) {
                if (pending_encoded === null) {
                    return;
                }
                const mem = new Uint8Array(wasm_memory.buffer);
                mem.set(pending_encoded, ptr);
                window.__shimlang_source_ready = false;
                pending_encoded = null;
            };
        },
    };

    if (typeof miniquad_add_plugin === "function") {
        miniquad_add_plugin(shimlang_bridge_plugin);
    } else {
        console.error("shimlang_bridge.js loaded before mq_js_bundle.js — miniquad_add_plugin is undefined");
    }
})();
