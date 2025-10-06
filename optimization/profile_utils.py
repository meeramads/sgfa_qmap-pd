"""
Simple profiling utilities for SGFA qMAP-PD.

Provides easy-to-use decorators and context managers for profiling code performance.
"""

import time
import functools
import cProfile
import pstats
import io
from pathlib import Path
from typing import Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time a function's execution.
    
    Usage:
        @time_function
        def my_function():
            # ... code ...
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        
        logger.info(f"{func.__name__} took {duration:.3f}s")
        return result
    
    return wrapper


def profile_function(output_file: Optional[str] = None, sort_by: str = 'cumulative', top_n: int = 20):
    """
    Decorator to profile a function with cProfile.
    
    Usage:
        @profile_function(output_file='profile.txt', sort_by='cumulative', top_n=20)
        def my_function():
            # ... code ...
    
    Args:
        output_file: Optional file to save profile stats (if None, prints to console)
        sort_by: Sorting key for stats ('cumulative', 'time', 'calls', etc.)
        top_n: Number of top functions to display
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            result = func(*args, **kwargs)
            
            profiler.disable()
            
            # Format stats
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(top_n)
            
            profile_output = stream.getvalue()
            
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(f"Profile for {func.__name__}\n")
                    f.write("=" * 80 + "\n")
                    f.write(profile_output)
                logger.info(f"Profile saved to {output_file}")
            else:
                print(f"\n{'=' * 80}")
                print(f"Profile for {func.__name__}")
                print('=' * 80)
                print(profile_output)
            
            return result
        
        return wrapper
    
    return decorator


class SimpleTimer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with SimpleTimer("My operation"):
            # ... code to time ...
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.verbose:
            logger.info(f"Completed: {self.name} in {self.duration:.3f}s")
    
    def get_duration(self) -> float:
        """Get the duration in seconds."""
        if self.duration is None:
            raise RuntimeError("Timer has not completed yet")
        return self.duration


class MemoryProfiler:
    """
    Simple memory profiler using psutil.
    
    Usage:
        with MemoryProfiler("My operation"):
            # ... code to profile ...
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Args:
            name: Name of the operation being profiled
            verbose: Whether to print memory information
        """
        self.name = name
        self.verbose = verbose
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            logger.warning("psutil not installed, memory profiling disabled")
            self.psutil = None
    
    def __enter__(self):
        if self.psutil:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if self.verbose:
                logger.info(f"Starting: {self.name} (Memory: {self.start_memory:.1f} MB)")
        return self
    
    def __exit__(self, *args):
        if self.psutil:
            self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_change = self.end_memory - self.start_memory
            
            if self.verbose:
                logger.info(
                    f"Completed: {self.name} "
                    f"(Memory: {self.end_memory:.1f} MB, "
                    f"Change: {memory_change:+.1f} MB)"
                )
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        if not self.psutil:
            return {}
        
        return {
            "start_mb": self.start_memory,
            "end_mb": self.end_memory,
            "change_mb": self.end_memory - self.start_memory if self.end_memory else None
        }


def quick_profile(func: Optional[Callable] = None, *, 
                  time_it: bool = True, 
                  profile_it: bool = False,
                  memory_it: bool = False) -> Callable:
    """
    Quick profiling decorator with multiple options.
    
    Usage:
        @quick_profile
        def my_function():
            # ... code ...
        
        @quick_profile(time_it=True, memory_it=True)
        def my_function():
            # ... code ...
    
    Args:
        func: Function to profile (when used without arguments)
        time_it: Whether to time the function
        profile_it: Whether to profile with cProfile
        memory_it: Whether to track memory usage
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            
            if time_it:
                start = time.time()
                result = f(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{f.__name__} execution time: {duration:.3f}s")
            
            if memory_it:
                with MemoryProfiler(f.__name__, verbose=True):
                    result = f(*args, **kwargs) if not time_it else result
            
            if profile_it:
                profiled_func = profile_function()(f)
                result = profiled_func(*args, **kwargs) if not (time_it or memory_it) else result
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    @time_function
    def example_function():
        """Example function with timing."""
        time.sleep(0.5)
        return "done"
    
    @profile_function(top_n=10)
    def example_profiled():
        """Example function with profiling."""
        result = sum(i**2 for i in range(10000))
        return result
    
    # Test timer context manager
    with SimpleTimer("Example operation"):
        time.sleep(0.3)
    
    # Test memory profiler
    with MemoryProfiler("Memory test"):
        data = [i for i in range(1000000)]
    
    # Test quick profile
    @quick_profile(time_it=True, memory_it=True)
    def quick_example():
        return sum(i for i in range(1000000))
    
    print("\nTesting decorated functions...")
    example_function()
    example_profiled()
    quick_example()
    
    print("\nProfiling utilities ready!")
