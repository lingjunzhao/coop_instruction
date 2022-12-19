import http.server
from socketserver import ThreadingMixIn
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
import os
import time


class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        SimpleHTTPRequestHandler.end_headers(self)

    def do_POST(self):
        output_file = "uploads/" + os.path.basename(self.path)
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        response = check_and_rename(output_file, body)
        if response:
            self.send_response(200)
        else:
            self.send_response(404)
        self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def check_and_rename(file, body):
    original_file = file
    version = 1
    max_versions = 100

    while version <= max_versions:
        file = original_file
        split = file.split(".")
        part1 = split[0] + "_v" + str(version)
        file = ".".join([part1, split[1]])

        if not os.path.isfile(file):
            # save here
            with open(file, "w") as f_out:
                f_out.write(body.decode("utf-8"))
            break

        version += 1

    timeout = time.time() + 60 * 3
    while True:
        with open(file, "r") as f_in:
            if "mistakes" in f_in.read():
                return True
            if time.time() > timeout:
                return False


if __name__ == '__main__':
    server = ThreadedHTTPServer(('localhost', 8888), MyHTTPRequestHandler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
