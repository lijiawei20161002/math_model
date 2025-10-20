"""
Monitoring and logging utilities for RL training.
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and aggregate metrics during training.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
                self.history[key] = []

            self.metrics[key].append(value)
            self.history[key].append(value)

    def get_average(self, key: str) -> Optional[float]:
        """Get moving average for a metric"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        return np.mean(self.metrics[key])

    def get_all_averages(self) -> Dict[str, float]:
        """Get moving averages for all metrics"""
        return {key: self.get_average(key) for key in self.metrics.keys()}

    def get_history(self, key: str) -> List[float]:
        """Get full history for a metric"""
        return self.history.get(key, [])

    def save(self, path: str):
        """Save metrics history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        """Load metrics history from JSON"""
        with open(path, 'r') as f:
            self.history = json.load(f)


class TimingTracker:
    """
    Track timing information for different operations.
    """

    def __init__(self):
        self.timers = {}
        self.counts = {}

    def start(self, name: str):
        """Start a timer"""
        if name not in self.timers:
            self.timers[name] = []
            self.counts[name] = 0

        self.timers[name].append(time.time())

    def stop(self, name: str):
        """Stop a timer and record the elapsed time"""
        if name not in self.timers or len(self.timers[name]) == 0:
            logger.warning(f"Timer '{name}' was not started")
            return

        start_time = self.timers[name].pop()
        elapsed = time.time() - start_time
        self.counts[name] += 1

        # Store in history (reuse timers list for history)
        if len(self.timers[name]) == 0:
            self.timers[name] = []
        self.timers[name].append(elapsed)

    def get_average(self, name: str) -> Optional[float]:
        """Get average time for an operation"""
        if name not in self.timers or len(self.timers[name]) == 0:
            return None
        return np.mean(self.timers[name])

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timers"""
        summary = {}
        for name, times in self.timers.items():
            if len(times) > 0 and isinstance(times[0], float):
                summary[name] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": self.counts[name],
                }
        return summary


class ProgressMonitor:
    """
    Monitor training progress with ETA estimation.
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = deque(maxlen=100)

    def update(self, step: int):
        """Update progress"""
        now = time.time()

        if self.current_step > 0:
            step_time = now - self.last_update_time
            self.step_times.append(step_time)

        self.current_step = step
        self.last_update_time = now

    def get_eta(self) -> float:
        """Get estimated time to completion (in seconds)"""
        if len(self.step_times) == 0:
            return 0.0

        avg_step_time = np.mean(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        return avg_step_time * remaining_steps

    def get_progress(self) -> float:
        """Get progress as a fraction (0 to 1)"""
        return self.current_step / self.total_steps if self.total_steps > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get progress statistics"""
        elapsed = time.time() - self.start_time
        eta = self.get_eta()

        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.get_progress(),
            "elapsed_time": elapsed,
            "eta": eta,
            "steps_per_second": self.current_step / elapsed if elapsed > 0 else 0.0,
        }


class ResourceMonitor:
    """
    Monitor GPU and memory usage.
    """

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics"""
        if not self.gpu_available:
            return {}

        stats = {}
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB

            stats[f"gpu_{i}"] = {
                "device": device_name,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
            }

        return stats

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if self.gpu_available:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)


class TrainingMonitor:
    """
    Comprehensive training monitor combining all tracking utilities.
    """

    def __init__(
        self,
        total_steps: int,
        metrics_window_size: int = 100,
        log_interval: int = 10,
    ):
        self.metrics_tracker = MetricsTracker(window_size=metrics_window_size)
        self.timing_tracker = TimingTracker()
        self.progress_monitor = ProgressMonitor(total_steps)
        self.resource_monitor = ResourceMonitor()
        self.log_interval = log_interval

    def update(self, step: int, metrics: Dict[str, float]):
        """Update all monitors"""
        self.metrics_tracker.update(metrics)
        self.progress_monitor.update(step)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of training state"""
        summary = {
            "metrics": self.metrics_tracker.get_all_averages(),
            "progress": self.progress_monitor.get_stats(),
            "timing": self.timing_tracker.get_summary(),
            "resources": self.resource_monitor.get_gpu_stats(),
        }
        return summary

    def should_log(self, step: int) -> bool:
        """Check if logging should occur at this step"""
        return step % self.log_interval == 0

    def log_summary(self, step: int):
        """Log a summary of training state"""
        if not self.should_log(step):
            return

        summary = self.get_summary()

        logger.info(f"=== Training Summary (Step {step}) ===")

        # Log metrics
        if summary["metrics"]:
            logger.info("Metrics:")
            for key, value in summary["metrics"].items():
                logger.info(f"  {key}: {value:.4f}")

        # Log progress
        if summary["progress"]:
            progress = summary["progress"]
            eta_mins = progress["eta"] / 60
            logger.info(f"Progress: {progress['progress']:.1%} - ETA: {eta_mins:.1f} min")

        # Log GPU usage
        if summary["resources"]:
            logger.info("GPU Usage:")
            for gpu_id, stats in summary["resources"].items():
                logger.info(f"  {gpu_id}: {stats['allocated_gb']:.2f} GB")

    def save(self, path: str):
        """Save monitoring data"""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        self.metrics_tracker.save(output_dir / "metrics.json")

        # Save summary
        summary = self.get_summary()
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.2f}PB"
