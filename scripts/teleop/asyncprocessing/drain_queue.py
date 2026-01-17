import os

def drain_fifo(fifo_path):
    """
    Drains a FIFO queue and returns the total bytes read.
    
    Args:
        fifo_path: Path to the FIFO (named pipe) on the filesystem
        
    Returns:
        int: Total number of bytes read from the FIFO
    """
    total_bytes = 0
    
    # Open the FIFO in read mode
    # This will block until a writer connects
    fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as fifo:
        while True:
            # Read in chunks for efficiency
            chunk = fifo.read(8192)
            if not chunk:
                # Empty read means the writer closed the pipe
                break
            total_bytes += len(chunk)
    
    return total_bytes

if __name__ == "__main__":
    fifo_path = "/tmp/slobot/fifo/leader_read.fifo"
    total_bytes = drain_fifo(fifo_path)
    print(f"Total bytes read from FIFO: {total_bytes}")