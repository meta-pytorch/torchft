#!/usr/bin/env python3

import torch
import torch.distributed as dist
from torch.autograd.profiler import profile
from torchft.process_group import ProcessGroupBabyGloo
import os

def test_process_group_profiling():
    # Initialize a ProcessGroupBabyGloo instance
    pg = ProcessGroupBabyGloo()
    
    # Create a simple TCP store for configuration
    store_addr = "localhost:12345/test_profiling"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    
    # Configure the process group
    pg.configure(store_addr, rank=0, world_size=1)
    
    # Create test tensors
    tensor = torch.ones(10, device="cpu")
    output_tensor = torch.zeros(10, device="cpu")
    
    # Create allreduce options
    op = dist.ReduceOp.SUM
    
    # Start profiling
    with profile(use_cuda=False) as prof:
        # Perform collective operations
        work = pg.allreduce([tensor], op)
        work.wait()
        
        # Test future
        future = work.get_future()
        future.wait()
    
    # Print profiling results
    print("Profiling Results:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Check for specific profiling spans
    spans = [event.name for event in prof.function_events]
    print("\nDetected profiling spans:")
    pg_spans = [span for span in spans if "ProcessGroupBaby" in span]
    for span in pg_spans:
        print(f"- {span}")
    
    # Verify all expected spans are present
    expected_spans = [
        "ProcessGroupBaby::allreduce",
        "ProcessGroupBaby::allreduce::wait",
        "ProcessGroupBaby::_wait",
        "ProcessGroupBaby::allreduce::get_future",
        "ProcessGroupBaby::_get_future",
    ]
    
    all_found = True
    for expected in expected_spans:
        found = any(expected in span for span in spans)
        if not found:
            print(f"Missing expected span: {expected}")
            all_found = False
    
    if all_found:
        print("\nSuccess: All expected profiling spans were detected!")
    else:
        print("\nSome expected profiling spans were not detected.")
    
    # Clean up
    pg.shutdown()

if __name__ == "__main__":
    test_process_group_profiling()