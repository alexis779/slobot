"""Shared memory block for zero-copy image transfer between processes."""

import numpy as np
import multiprocessing.shared_memory as shm
import struct
from slobot.configuration import Configuration

class SharedMemoryBlock:
    """A single shared memory block with state-based locking for frame transfer.
    
    State Cycle: FREE -> WRITING -> READY -> READING -> FREE
    """
    
    LOGGER = Configuration.logger(__name__)
    
    # States
    STATE_FREE = 0      # Available for writing
    STATE_WRITING = 1   # Producer is writing
    STATE_READY = 2     # Data is ready for reading
    STATE_READING = 3   # Consumer is reading
    
    # Layout:
    # Byte 0: State (uint8)
    # Byte 1-4: Width (uint32)
    # Byte 5-8: Height (uint32)
    # Byte 9+: Data
    # Channels defaulted to 3 (BGR)
    HEADER_SIZE = 9
    CHANNELS = 3
    
    @staticmethod
    def get_name_from_camera_id(camera_id: int) -> str:
        """Generate shared memory block name from camera ID.
        
        Args:
            camera_id: The camera ID
            
        Returns:
            Shared memory block name (e.g., "shm_webcam2")
        """
        return f"shm_webcam{camera_id}"
    
    @staticmethod
    def create(name: str, size: int) -> 'SharedMemoryBlock':
        """Create a new shared memory block.
        
        Args:
            name: Unique name for the shared memory block
            size: Total size in bytes
            
        Returns:
            New SharedMemoryBlock instance
        """
        # Create the shared memory
        shm_obj = shm.SharedMemory(name=name, create=True, size=size)
        # Initialize state to FREE
        shm_obj.buf[0] = SharedMemoryBlock.STATE_FREE
        shm_obj.close()
        
        # Now attach to it using the normal constructor
        block = SharedMemoryBlock(name)
        SharedMemoryBlock.LOGGER.info(f"Created shared memory block {name} size {size}")
        return block
    
    def __init__(self, name: str, size: int = None):
        """Initialize the shared memory block.
        
        Args:
            name: Unique name for the shared memory block
            size: Size in bytes (required if creating new shared memory)
        """
        self.name = name
        self.size = 0
        self.shm = None
        
        try:
            # Try to attach to existing shared memory
            self.shm = shm.SharedMemory(name=self.name, create=False)
            self.size = self.shm.size
            self.LOGGER.info(f"Attached to existing shared memory block {self.name}")
        except FileNotFoundError:
            # Create new shared memory if it doesn't exist
            if size is None:
                raise ValueError(f"Size must be provided when creating new shared memory block '{name}'")
            self.shm = shm.SharedMemory(name=self.name, create=True, size=size)
            self.size = self.shm.size
            # Initialize state to FREE
            self.shm.buf[0] = self.STATE_FREE
            self.LOGGER.info(f"Created new shared memory block {self.name} with size {size}")

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a frame to shared memory if FREE.
        
        Args:
            frame: Numpy array (H, W, C) - must be 3 channels
            
        Returns:
            True if written, False if dropped (busy)
        """
        # Check if FREE
        if self.shm.buf[0] != self.STATE_FREE:
            return False
            
        # Set to WRITING
        self.shm.buf[0] = self.STATE_WRITING
        
        height, width, channels = frame.shape
        
        # Validate channels
        if channels != self.CHANNELS:
            self.LOGGER.error(f"Frame must have {self.CHANNELS} channels, got {channels}")
            self.shm.buf[0] = self.STATE_FREE
            return False
        
        data_bytes = frame.tobytes()
        
        # Check size
        if len(data_bytes) + self.HEADER_SIZE > self.size:
            self.LOGGER.error(f"Frame too large for shared memory! {len(data_bytes)} > {self.size - self.HEADER_SIZE}")
            # Revert to FREE
            self.shm.buf[0] = self.STATE_FREE
            return False
        
        # Write Header
        struct.pack_into('I', self.shm.buf, 1, width)
        struct.pack_into('I', self.shm.buf, 5, height)
        
        # Write Data
        self.shm.buf[self.HEADER_SIZE:self.HEADER_SIZE+len(data_bytes)] = data_bytes
        
        # Set to READY
        self.shm.buf[0] = self.STATE_READY
        return True

    def read_frame(self) -> np.ndarray | None:
        """Read a frame from shared memory if READY.
        
        Returns:
            Frame as numpy array, or None if not ready.
        """
        # Check if READY
        if self.shm.buf[0] != self.STATE_READY:
            return None
            
        # Set to READING
        self.shm.buf[0] = self.STATE_READING
        
        # Read Header (channels defaulted to 3)
        width = struct.unpack_from('I', self.shm.buf, 1)[0]
        height = struct.unpack_from('I', self.shm.buf, 5)[0]
        
        data_size = width * height * self.CHANNELS
        
        # Read Data
        # Note: We must create a copy because we're about to free the buffer
        data = bytes(self.shm.buf[self.HEADER_SIZE:self.HEADER_SIZE+data_size])
        
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, self.CHANNELS))
        
        # Set to FREE
        self.shm.buf[0] = self.STATE_FREE
        return frame

    def close(self):
        """Close the shared memory handle."""
        if self.shm:
            self.shm.close()
            
    def unlink(self):
        """Unlink (delete) the shared memory block."""
        if self.shm:
            self.shm.unlink()
