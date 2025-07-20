
import http.server
import socketserver

def start_dashboard():
    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler
    print("[🧠] Launching Shadow Dashboard at http://localhost:8080")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
