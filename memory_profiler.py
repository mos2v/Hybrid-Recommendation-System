import tracemalloc
import time
import gc
import numpy as np
from functools import wraps

class MemoryProfiler:
    """
    Handles memory profiling for recommendation system components.
    """
    def __init__(self):
        """Initialize the memory profiler"""
        self.snapshots = []
        
    def profile_function(self, func):
        """Decorator to profile memory usage of a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"Function '{func.__name__}':")
            print(f"  Time taken: {end_time - start_time:.2f} seconds")
            print(f"  Current memory usage: {current / 10**6:.2f} MB")
            print(f"  Peak memory usage: {peak / 10**6:.2f} MB")
            
            # Save snapshot
            snapshot = {
                'function': func.__name__,
                'time': end_time - start_time,
                'current_memory': current / 10**6,
                'peak_memory': peak / 10**6,
                'timestamp': time.time()
            }
            self.snapshots.append(snapshot)
            
            return result
        return wrapper
    
    def profile_block(self, block_name):
        """Context manager to profile a block of code"""
        class _ProfileBlock:
            def __init__(self, block_name, profiler):
                self.block_name = block_name
                self.profiler = profiler
            
            def __enter__(self):
                tracemalloc.start()
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                print(f"Block '{self.block_name}':")
                print(f"  Time taken: {end_time - self.start_time:.2f} seconds")
                print(f"  Current memory usage: {current / 10**6:.2f} MB")
                print(f"  Peak memory usage: {peak / 10**6:.2f} MB")
                
                # Save snapshot
                snapshot = {
                    'block': self.block_name,
                    'time': end_time - self.start_time,
                    'current_memory': current / 10**6,
                    'peak_memory': peak / 10**6,
                    'timestamp': time.time()
                }
                self.profiler.snapshots.append(snapshot)
        
        return _ProfileBlock(block_name, self)
    
    def profile_object(self, obj, name=""):
        """Profile the memory usage of an object"""
        size = 0
        obj_type = type(obj).__name__
        
        if isinstance(obj, np.ndarray):
            size = obj.nbytes / 10**6  # MB
        elif hasattr(obj, 'toarray') and callable(obj.toarray):
            # For sparse matrices
            try:
                size = obj.data.nbytes / 10**6  # MB (approximate for sparse)
                obj_type = f"Sparse {type(obj).__name__}"
            except:
                size = -1
        elif hasattr(obj, '__sizeof__'):
            size = obj.__sizeof__() / 10**6  # MB
        
        print(f"Object '{name}' ({obj_type}):")
        print(f"  Size: {size:.2f} MB")
        
        return {
            'name': name,
            'type': obj_type,
            'size_mb': size,
            'timestamp': time.time()
        }
    
    def trigger_gc(self):
        """Force garbage collection to free memory"""
        gc.collect()
        
    def get_summary(self):
        """Get summary of profiling results"""
        if not self.snapshots:
            return "No profiling data available"
            
        summary = "Memory Profiling Summary:\n"
        summary += "-" * 80 + "\n"
        summary += "| {:<20} | {:<10} | {:<15} | {:<15} |\n".format(
            "Function/Block", "Time (s)", "Current (MB)", "Peak (MB)"
        )
        summary += "-" * 80 + "\n"
        
        for snapshot in sorted(self.snapshots, key=lambda x: x.get('peak_memory', 0), reverse=True):
            if 'function' in snapshot:
                name = snapshot['function']
            elif 'block' in snapshot:
                name = snapshot['block']
            else:
                name = "Unknown"
                
            summary += "| {:<20} | {:<10.2f} | {:<15.2f} | {:<15.2f} |\n".format(
                name, snapshot['time'], snapshot['current_memory'], snapshot['peak_memory']
            )
        
        summary += "-" * 80 + "\n"
        return summary