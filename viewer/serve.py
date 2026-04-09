"""
Tiny static file server for the feature viewer.

Run from anywhere:
    python viewer/serve.py

Then open http://localhost:8765/viewer/ in your browser.

Serves from the repo root so the viewer can reach ../features/ via relative
URLs. Default port 8765 — override with the PORT env var.
"""

import http.server
import os
import socketserver
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get('PORT', '8765'))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that disables caching so refreshes always
    pick up newly-written features.
    """
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        # glTF 2.0 uses image/webp natively — make sure the right mimetype is served
        super().end_headers()

    # Add .glb and .webp mimetypes explicitly (some Python builds miss them)
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.glb': 'model/gltf-binary',
        '.gltf': 'model/gltf+json',
        '.webp': 'image/webp',
        '.json': 'application/json',
    }


def main():
    os.chdir(REPO_ROOT)
    handler = NoCacheHandler
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Serving {REPO_ROOT}")
        print(f"Open: http://localhost:{PORT}/viewer/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
