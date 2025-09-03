import sys
import time
from typing import Optional, Iterable, Union


class ProgressBar:
    """
    A simplified tqdm-like progress bar for tracking training progress.
    
    Features:
    - Real-time progress visualization
    - Automatic rate calculation (items/sec)
    - Elapsed and estimated remaining time
    - Customizable descriptions and postfix information
    """
    
    def __init__(self, 
                 total: Optional[int] = None,
                 desc: str = "",
                 unit: str = "it",
                 ncols: int = 80,
                 disable: bool = False):
        """
        Initialize progress bar.
        
        Args:
            total: Expected number of iterations
            desc: Description prefix
            unit: Unit of measurement (e.g., 'epoch', 'batch', 'it')
            ncols: Terminal width for progress bar
            disable: Disable progress bar completely
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.ncols = ncols
        self.disable = disable
        
        self.n = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.postfix = {}
        
        if not self.disable:
            self._clear_line()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        if self.disable:
            return
            
        self.n += n
        current_time = time.time()
        
        # Only update display every 0.1 seconds to avoid flickering
        if current_time - self.last_print_time >= 0.1 or self.n == self.total:
            self._display()
            self.last_print_time = current_time
    
    def set_postfix(self, **kwargs):
        """Set postfix information (e.g., loss=0.5, acc=0.95)."""
        self.postfix.update(kwargs)
    
    def set_description(self, desc: str):
        """Update description."""
        self.desc = desc
    
    def close(self):
        """Finalize progress bar."""
        if not self.disable:
            if self.total and self.n < self.total:
                self.n = self.total
                self._display()
            print()  # New line after completion
    
    def _display(self):
        """Display current progress."""
        elapsed = time.time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0
        
        # Build progress string
        if self.total:
            percentage = 100 * self.n / self.total
            bar_length = max(10, self.ncols - 50)  # Reserve space for text
            filled_length = int(bar_length * self.n // self.total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Estimate remaining time
            if rate > 0 and self.n < self.total:
                eta = (self.total - self.n) / rate
                eta_str = self._format_time(eta)
            else:
                eta_str = "00:00"
            
            progress_str = f"{self.desc}: {percentage:3.0f}%|{bar}| {self.n}/{self.total} [{self._format_time(elapsed)}<{eta_str}, {rate:.2f}{self.unit}/s"
        else:
            progress_str = f"{self.desc}: {self.n}{self.unit} [{self._format_time(elapsed)}, {rate:.2f}{self.unit}/s"
        
        # Add postfix information
        if self.postfix:
            postfix_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in self.postfix.items()])
            progress_str += f", {postfix_str}"
        
        progress_str += "]"
        
        # Truncate if too long
        if len(progress_str) > self.ncols:
            progress_str = progress_str[:self.ncols-3] + "..."
        
        self._clear_line()
        sys.stdout.write(progress_str)
        sys.stdout.flush()
    
    def _clear_line(self):
        """Clear current line."""
        sys.stdout.write('\r' + ' ' * self.ncols + '\r')
    
    def _format_time(self, seconds):
        """Format time in MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def tqdm(iterable: Optional[Iterable] = None, 
         total: Optional[int] = None,
         desc: str = "",
         unit: str = "it",
         ncols: int = 80,
         disable: bool = False):
    """
    Create a progress bar for iterables (tqdm-like interface).
    
    Args:
        iterable: Iterable to wrap
        total: Expected number of iterations (if iterable not provided)
        desc: Description prefix
        unit: Unit of measurement
        ncols: Terminal width
        disable: Disable progress bar
    
    Returns:
        ProgressBar instance or wrapped iterable
    """
    if iterable is not None:
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        pbar = ProgressBar(total=total, desc=desc, unit=unit, ncols=ncols, disable=disable)
        
        def wrapped_iterable():
            try:
                for item in iterable:
                    yield item
                    pbar.update(1)
            finally:
                pbar.close()
        
        return wrapped_iterable()
    else:
        return ProgressBar(total=total, desc=desc, unit=unit, ncols=ncols, disable=disable)


# Convenience functions for common use cases
def trange(n: int, desc: str = "", unit: str = "it", disable: bool = False):
    """tqdm for range objects."""
    return tqdm(range(n), desc=desc, unit=unit, disable=disable)