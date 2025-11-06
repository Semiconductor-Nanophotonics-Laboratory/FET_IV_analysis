import socket

class Keithley2600:
    def __init__(self, ip, port=5025, timeout=5):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)

    def send(self, cmd):
        if not self.sock:
            raise RuntimeError("Not connected.")
        self.sock.sendall((cmd + '\n').encode())

    def query(self, cmd):
        self.send(cmd)
        return self.sock.recv(4096).decode().strip()

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None