const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, 'assets', 'web');

// Ensure output directory exists
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// Write an enhanced HTML wrapper with console redirects, error traps, dynamic
// path resolution, and retry-based startup. The engine bundle is inlined so
// Android WebView cannot race or fail a separate engine.js asset request.
function createHtmlContent(engineScript) {
  const safeEngineScript = engineScript.replace(/<\/script/gi, '<\\/script');
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tarteel Engine Headless</title>
  <script>
    window.onerror = function(message, source, lineno, colno, error) {
      var errMsg = 'HTML Error: ' + message + ' at ' + source + ':' + lineno;
      if (window.flutter_inappwebview && window.flutter_inappwebview.callHandler) {
        window.flutter_inappwebview.callHandler('onEngineEvent', {
          type: 'error', 
          message: errMsg
        });
      } else {
        console.error(errMsg);
      }
    };
  </script>
</head>
<body>
  <script>
${safeEngineScript}
  </script>
  <script>
    function getAssetsUrl() {
      var href = window.location.href;
      return href.substring(0, href.lastIndexOf('/'));
    }

    var initAttempts = 0;
    function tryInit() {
      if (window._engineInitialized) return;
      if (window.initEngine) {
        window._engineInitialized = true;
        var assetsUrl = getAssetsUrl();
        console.log("Initializing engine with assetsUrl: " + assetsUrl);
        window.initEngine(assetsUrl);
      } else {
        initAttempts++;
        if (initAttempts <= 120) {
          setTimeout(tryInit, 500);
        } else {
          console.error("window.initEngine is undefined after waiting for engine bundle startup.");
        }
      }
    }

    window.addEventListener("flutterInAppWebViewPlatformReady", function(event) {
      console.log("flutterInAppWebViewPlatformReady received");
      tryInit();
    });

    // Fallback if platform ready already fired or we are debugging in browser
    setTimeout(function() {
      console.log("Timeout fallback checking engine initialization");
      tryInit();
    }, 2000);
  </script>
</body>
</html>`;
}

const hfAudioPlugin = {
  name: 'hf-audio-plugin',
  setup(build) {
    build.onResolve({ filter: /^@huggingface\/transformers\/src\/utils\/audio\.js$/ }, args => {
      return { path: path.resolve(__dirname, 'node_modules', '@huggingface', 'transformers', 'src', 'utils', 'audio.js') }
    })
  },
};

const nodePolyfillPlugin = {
  name: 'node-polyfill-plugin',
  setup(build) {
    const emptyStubPath = path.resolve(__dirname, 'src', 'stubs', 'empty.ts');
    const onnxNodeStubPath = path.resolve(__dirname, 'src', 'stubs', 'onnxruntime-node.ts');
    build.onResolve({ filter: /^onnxruntime-node$/ }, () => {
      return { path: onnxNodeStubPath };
    });
    // Stub native Node modules (and their subpaths/promises) to prevent runtime dynamic require exceptions in the browser.
    build.onResolve({ filter: /^(node:)?(fs|path|url|crypto|stream|util)(\/.*)?$/ }, args => {
      return { path: emptyStubPath };
    });
  },
};

esbuild.build({
  entryPoints: ['src/web-engine.ts'],
  bundle: true,
  outfile: path.join(outDir, 'engine.js'),
  minify: true,
  sourcemap: false,
  target: ['chrome100', 'safari15'],
  format: 'iife',
  globalName: 'TarteelApp',
  platform: 'browser',
  plugins: [hfAudioPlugin, nodePolyfillPlugin],
}).then(() => {
  const engineScript = fs.readFileSync(path.join(outDir, 'engine.js'), 'utf8');
  fs.writeFileSync(path.join(outDir, 'index.html'), createHtmlContent(engineScript));
  console.log('Build completed successfully.');
}).catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
