.. automodule:: torchft.process_group
    :members:
    :undoc-members:
    :show-inheritance:

NCCL Watchdog
=============

The ``ProcessGroupNCCL`` class includes a watchdog feature for fast aborts of NCCL operations. 
This is particularly useful in distributed training scenarios where NCCL operations might hang 
indefinitely, causing the entire training job to stall.

Usage example::

    from datetime import timedelta
    from torchft.process_group import ProcessGroupNCCL
    
    # Create a process group with watchdog (5 second timeout)
    pg = ProcessGroupNCCL(
        timeout=timedelta(seconds=60.0),
        watchdog_timeout=timedelta(seconds=5.0)
    )
    
    # To disable the watchdog
    pg = ProcessGroupNCCL(watchdog_timeout=None)
