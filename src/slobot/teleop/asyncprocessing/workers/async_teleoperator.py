"""Async Teleoperator - main entry point for all the workers."""

from slobot.teleop.asyncprocessing.fifo_queue import FifoQueue


class AsyncTeleoperator:
    """Main entry point for the async teleoperator control loop.
    
    Manages FIFO queues and spawns worker processes for:
    - Cron (tick generator at a given frequency)
    - Leader Read (reads leader arm position)
    - Follower Control (sends commands to follower arm and reads position)
    - Sim Step (runs Genesis simulation)
    - Webcam Capture (captures webcam frames)
    """
    
    def spawn_cron_worker(self, **kwargs):
        from slobot.teleop.asyncprocessing.workers.cron_worker import CronWorker
        cron_worker = CronWorker(
            leader_read_queue=FifoQueue(FifoQueue.QUEUE_LEADER_READ),
            recording_id=kwargs['recording_id'],
            fps=kwargs['fps'],
        )
        cron_worker.run()

    def spawn_leader_read_worker(self, **kwargs):
        from slobot.teleop.asyncprocessing.workers.leader_read_worker import LeaderReadWorker
        leader_read_worker = LeaderReadWorker(
            input_queue=FifoQueue(FifoQueue.QUEUE_LEADER_READ),
            follower_control_queue=FifoQueue(FifoQueue.QUEUE_FOLLOWER_CONTROL),
            recording_id=kwargs['recording_id'],
            port=kwargs['port'],
        )
        leader_read_worker.run()
        
    def spawn_follower_control_worker(self, **kwargs):
        from slobot.teleop.asyncprocessing.workers.follower_control_worker import FollowerControlWorker
        follower_control_worker = FollowerControlWorker(
            input_queue=FifoQueue(FifoQueue.QUEUE_FOLLOWER_CONTROL),
            webcam_capture_queue=FifoQueue(FifoQueue.QUEUE_WEBCAM_CAPTURE) if kwargs['webcam'] else None,
            sim_step_queue=FifoQueue(FifoQueue.QUEUE_SIM_STEP) if kwargs['sim'] else None,
            recording_id=kwargs['recording_id'],
            port=kwargs['port'],
        )
        follower_control_worker.run()

    def spawn_sim_step_worker(self, **kwargs):
        from slobot.teleop.asyncprocessing.workers.sim_step_worker import SimStepWorker
        sim_step_worker = SimStepWorker(
            input_queue=FifoQueue(FifoQueue.QUEUE_SIM_STEP),
            recording_id=kwargs['recording_id'],
            fps=kwargs['fps'],
            substeps=kwargs['substeps'],
            vis_mode=kwargs['vis_mode'],
            width=kwargs['width'],
            height=kwargs['height'],
        )
        sim_step_worker.run()

    def spawn_webcam_capture_worker(self, **kwargs):
        from slobot.teleop.asyncprocessing.workers.webcam_capture_worker import WebcamCaptureWorker
        webcam_capture_worker = WebcamCaptureWorker(
            input_queue=FifoQueue(FifoQueue.QUEUE_WEBCAM_CAPTURE),
            recording_id=kwargs['recording_id'],
            camera_id=kwargs['camera_id'],
            width=kwargs['width'],
            height=kwargs['height'],
            fps=kwargs['fps'],
        )
        webcam_capture_worker.run()