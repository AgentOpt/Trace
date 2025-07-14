from typing import Optional
from opto.trainer.trace_memory_db import TraceMemoryDB

class BaseLogger:

    def __init__(self, log_dir='./logs', **kwargs):
        """Initialize the logger. This method can be overridden by subclasses."""
        self.log_dir = log_dir
        pass

    def log(self, name, data, step, **kwargs):
        """Log a message with the given name and data at the specified step.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (e.g., color)
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ConsoleLogger(BaseLogger):
    """A simple logger that prints messages to the console."""
    
    def log(self, name, data, step, **kwargs):
        """Log a message to the console.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (e.g., color)
        """
        color = kwargs.get('color', None)
        # Simple color formatting for terminal output
        color_codes = {
            'green': '\033[92m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'end': '\033[0m'
        }
        
        start_color = color_codes.get(color, '')
        end_color = color_codes['end'] if color in color_codes else ''
        
        print(f"[Step {step}] {start_color}{name}: {data}{end_color}")


class TensorboardLogger(ConsoleLogger):
    """A logger that writes metrics to TensorBoard."""
    
    def __init__(self, log_dir='./logs', verbose=True, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.verbose = verbose
        # Late import to avoid dependency issues
        try:             
            from tensorboardX import SummaryWriter
        except ImportError:
            # try importing from torch.utils.tensorboard if tensorboardX is not available
            from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(self.log_dir)

    def log(self, name, data, step, **kwargs):
        """Log a message to TensorBoard.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (not used here)
        """
        if self.verbose:
            super().log(name, data, step, **kwargs)
        if isinstance(data, str):
            # If data is a string, log it as text
            self.writer.add_text(name, data, step)
        else:
            # Otherwise, log it as a scalar
            self.writer.add_scalar(name, data, step)

class WandbLogger(ConsoleLogger):
    """A logger that writes metrics to Weights and Biases (wandb)."""
    
    def __init__(self, log_dir='./logs', verbose=True, project=None, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.verbose = verbose
        # Late import to avoid dependency issues
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required for WandbLogger. Install it with: pip install wandb")
        
        # Initialize wandb
        self.wandb = wandb
        if not wandb.run:
            wandb.init(project=project, dir=log_dir, **kwargs)

    def log(self, name, data, step, **kwargs):
        """Log a message to Weights and Biases.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (not used here)
        """
        if self.verbose:
            super().log(name, data, step, **kwargs)
        
        # Log to wandb
        if isinstance(data, str):
            # For string data, we can log it as a custom chart or just print it
            # wandb doesn't have a direct equivalent to tensorboard's add_text
            # but we can log it in a structured way
            self.wandb.log({f"{name}_text": data}, step=step)
        else:
            # For numeric data, log as scalar
            self.wandb.log({name: data}, step=step)

class TraceMemoryLogger(ConsoleLogger):
    """A logger that writes metrics to TraceMemoryDB."""
    
    def __init__(self, log_dir='./logs', verbose=True, memory_db: Optional[TraceMemoryDB] = None, 
                 problem_id: str = "training_run", **kwargs):
        super().__init__(log_dir, **kwargs)
        self.verbose = verbose
        self.memory_db = memory_db or TraceMemoryDB()
        self.problem_id = problem_id
        
    def log(self, name, data, step, **kwargs):
        """Log a message to TraceMemoryDB.
        
        Args:
            name: Name of the metric
            data: Value of the metric
            step: Current step/iteration
            **kwargs: Additional arguments (e.g., color, metadata)
        """
        if self.verbose:
            super().log(name, data, step, **kwargs)
        
        # Extract metadata from kwargs
        metadata = kwargs.get('metadata', {})
        metadata['metric_name'] = name
        metadata['color'] = kwargs.get('color', None)
        
        # Determine data_payload based on metric name
        if 'score' in name.lower():
            data_payload = 'score'
        elif 'loss' in name.lower():
            data_payload = 'loss'
        else:
            data_payload = 'metric'
            
        # Log to memory database
        self.memory_db.log_data(
            problem_id=self.problem_id,
            step_id=step,
            data={name: data},
            data_payload=data_payload,
            metadata=metadata
        )

DefaultLogger = ConsoleLogger