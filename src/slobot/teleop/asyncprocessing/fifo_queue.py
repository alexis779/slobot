"""FIFO Queue implementation using Linux named pipes for IPC."""

import os
import struct
import select
from typing import Any, Optional

from slobot.configuration import Configuration


class FifoQueue:
    """A FIFO queue wrapper using Linux named pipes for inter-process communication.
    
    Messages are binary-formatted with a fixed header containing length, type, deadline and step.
    The deadline is the timestamp by which all downstream processing must complete.
    Supports polling for the latest message while dropping stale ones.
    """

    # Queue names
    QUEUE_LEADER_READ = 'leader_read'
    QUEUE_FOLLOWER_CONTROL = 'follower_control'
    QUEUE_WEBCAM_CAPTURE = 'webcam_capture'
    QUEUE_SIM_STEP = 'sim_step'

    # Message header: [msg_length: u32][msg_type: u8][deadline: f64][step: u32]
    HEADER_FORMAT = '<IBdI'  # little-endian: uint32, uint8, float64, uint32
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    # Message types
    MSG_EMPTY = 0        # Empty tick (no payload)
    MSG_QPOS = 1         # N-DOF float array
    MSG_QPOS_RGB = 2     # N-DOF float array + RGB array
    MSG_BGR = 3          # BGR array
    MSG_POISON_PILL = 255
    
    # QPOS format: 6 doubles
    QPOS_FORMAT = '<6d'
    QPOS_SIZE = struct.calcsize(QPOS_FORMAT)  # 48 bytes
    
    LOGGER = Configuration.logger(__name__)

    def __init__(self, name: str):
        """Initialize a FIFO queue.
        
        Args:
            name: The name of the queue (used in the file path)
        """
        self.name = name
        self.path = f"/tmp/slobot/fifo/{name}.fifo"
        self.fd: Optional[int] = None
        self._read_buffer = b''        

    def open_write(self):
        """Open the FIFO for writing (blocking until a reader connects)."""
        self.ensure_exists()
        self.LOGGER.info(f"Opening FIFO {self.name} for writing")
        self.fd = os.open(self.path, os.O_WRONLY)
        self.LOGGER.info(f"FIFO {self.name} opened for writing")

    def open_read(self):
        """Open the FIFO for reading (draining it before hand if not empty)."""
        self.ensure_exists()
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.drain()

    def ensure_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            os.mkfifo(self.path)

    def drain(self):
        """Drain the FIFO by reading all available data."""
        drained_bytes = 0
        try:
            while True:
                chunk = os.read(self.fd, 65536) # read in non blocking mode
                if not chunk:
                    break
                else:
                    drained_bytes += len(chunk)
        except BlockingIOError:
            # read would be blocking, so FIFO is currently empty
            pass
        if drained_bytes > 0:
            self.LOGGER.warning(f"Drained {drained_bytes} bytes from FIFO {self.name}")

    def close(self):
        """Close the FIFO file descriptor."""
        os.close(self.fd)

    def write(self, msg_type: int, result_payload: Any, deadline: float, step: int):
        """Write a message to the FIFO.
        
        Args:
            msg_type: The message type (MSG_EMPTY, MSG_QPOS, MSG_RGB, etc.)
            payload: The raw payload bytes
            deadline: The deadline by which downstream processing must complete
        """
        # Serialize the payload into raw bytes
        payload = self.to_bytes(msg_type, result_payload)

        msg_len = self.HEADER_SIZE + len(payload)
        header = struct.pack(self.HEADER_FORMAT, msg_len, msg_type, deadline, step)
        try:
            os.write(self.fd, header + payload)
        except BrokenPipeError as bpe:
            self.LOGGER.error(f"Reader disconnected on FIFO {self.name}: {bpe}")
            os.close(self.fd)
            self.open_write()

            # retry once
            os.write(self.fd, header + payload)

    def write_empty(self, deadline: float, step: int):
        """Write an empty tick message."""
        self.write(self.MSG_EMPTY, b'', deadline, step)

    def write_qpos(self, qpos: list[float], deadline: float):
        """Write an N-DOF position array.
        
        Args:
            qpos: List of 6 joint positions in radians
            deadline: The deadline for downstream processing
        """
        payload = struct.pack(self.QPOS_FORMAT, *qpos)
        self.write(self.MSG_QPOS, payload, deadline)

    def send_poison_pill(self):
        """Send a poison pill message to signal graceful shutdown."""
        self.write(self.MSG_POISON_PILL, b'', 0.0, 0)

    def poll_next(self) -> Optional[tuple[int, float, bytes]]:
        """Poll for the next message without dropping any.
        
        Returns messages in FIFO order. Use this for workers that need all messages
        (e.g., metrics logging).
        
        Returns:
            Tuple of (msg_type, deadline, payload) or None if no message available
        """
        # Check if we already have a complete message in the buffer
        if len(self._read_buffer) >= self.HEADER_SIZE:
            msg_len, msg_type, deadline, step = struct.unpack(
                self.HEADER_FORMAT, 
                self._read_buffer[:self.HEADER_SIZE]
            )
            if len(self._read_buffer) >= msg_len:
                payload = self._read_buffer[self.HEADER_SIZE:msg_len]
                self._read_buffer = self._read_buffer[msg_len:]
                return (msg_type, deadline, payload, step)
        
        # Wait for data to be available
        select.select([self.fd], [], [])
        
        # Read available data into buffer
        try:
            while True:
                chunk = os.read(self.fd, 65536)
                if not chunk:
                    break
                self._read_buffer += chunk
        except BlockingIOError:
            pass  # No more data available
        
        # Try to parse one complete message
        if len(self._read_buffer) >= self.HEADER_SIZE:
            msg_len, msg_type, deadline, step = struct.unpack(
                self.HEADER_FORMAT, 
                self._read_buffer[:self.HEADER_SIZE]
            )
            if len(self._read_buffer) >= msg_len:
                payload = self._read_buffer[self.HEADER_SIZE:msg_len]
                self._read_buffer = self._read_buffer[msg_len:]
                return (msg_type, deadline, payload, step)
        
        return None

    def poll_latest(self, timeout: Optional[float] = None) -> Optional[tuple[int, float, bytes]]:
        """Poll for the latest non-stale message.
        
        Reads all available messages, drops stale ones (deadline already passed),
        and returns the most recent message that still has time remaining.
        
        Args:
            timeout: Optional timeout in seconds for blocking wait (None = block forever)
        
        Returns:
            Tuple of (msg_type, deadline, payload) or None if no message available
        """
        import time
        
        # Wait for data to be available
        if timeout is not None or timeout == 0:
            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                return None
        else:
            # Blocking wait
            select.select([self.fd], [], [])
        
        # Read all available data into buffer
        try:
            while True:
                chunk = os.read(self.fd, 65536)
                if not chunk:
                    break
                self._read_buffer += chunk
        except BlockingIOError:
            pass  # No more data available
        
        # Parse all complete messages, keep only the latest non-stale one
        latest_msg: Optional[tuple[int, float, bytes]] = None
        current_time = time.time()
        
        while len(self._read_buffer) >= self.HEADER_SIZE:
            # Peek at header
            msg_len, msg_type, deadline, step = struct.unpack(
                self.HEADER_FORMAT, 
                self._read_buffer[:self.HEADER_SIZE]
            )
            
            # Check if we have the complete message
            if len(self._read_buffer) < msg_len:
                break  # Incomplete message, wait for more data
            
            # Extract payload
            payload = self._read_buffer[self.HEADER_SIZE:msg_len]

            # Deserialize the raw bytes into an object
            payload = self.from_bytes(msg_type, payload)

            self._read_buffer = self._read_buffer[msg_len:]
            
            # Poison pill is always returned immediately
            if msg_type == self.MSG_POISON_PILL:
                return (msg_type, deadline, step, payload)
            
            # Keep only messages with deadline still in the future (or keep latest if all expired)
            if deadline > current_time or latest_msg is None:
                latest_msg = (msg_type, deadline, step, payload)
        
        return latest_msg

    @staticmethod
    def to_bytes(msg_type: int, result_payload: Any) -> bytes:
        """Convert a message type and payload to bytes."""
        match msg_type:
            case FifoQueue.MSG_EMPTY:
                return b''
            case FifoQueue.MSG_QPOS:
                return FifoQueue.pack_qpos(result_payload)
            case FifoQueue.MSG_POISON_PILL:
                return b''
            case _:
                raise ValueError(f"Unknown message type: {msg_type}")

    @staticmethod
    def from_bytes(msg_type: int, payload: bytes) -> Any:
        """Convert bytes to a message type and payload."""
        match msg_type:
            case FifoQueue.MSG_EMPTY:
                return None
            case FifoQueue.MSG_QPOS:
                return FifoQueue.parse_qpos(payload)
            case FifoQueue.MSG_POISON_PILL:
                return None
            case _:
                raise ValueError(f"Unknown message type: {msg_type}")

    @staticmethod
    def pack_qpos(qpos: list[float]) -> bytes:
        """Pack a qpos list into bytes."""
        return struct.pack(FifoQueue.QPOS_FORMAT, *qpos)

    @staticmethod
    def parse_qpos(payload: bytes) -> list[float]:
        """Parse a qpos payload into a list of floats."""
        return list(struct.unpack(FifoQueue.QPOS_FORMAT, payload))

    def cleanup(self):
        """Remove the FIFO file."""
        self.close()
        if os.path.exists(self.path):
            os.remove(self.path)
