.. automodule:: torchft.local_sgd
    :members:
    :undoc-members:
    :show-inheritance:

CPU staging for averaged parameters
-----------------------------------

``LocalSGD`` can stage averaged parameters on CPU while waiting for the quorum
to commit. Enable this path when synchronization-time accelerator memory is the
limiting factor:

.. code-block:: python

    with LocalSGD(
        manager=manager,
        model=model,
        optimizer=optimizer,
        sync_every=8,
        offload_averaged_parameters_to_cpu=True,
    ):
        train(model, optimizer)

The option is disabled by default, so existing callers retain the original
behavior. When enabled, LocalSGD all-reduces one parameter at a time and moves
each averaged result to CPU before processing the next parameter. If the quorum
commits, the staged values are copied back to the original parameter devices;
otherwise, they are discarded.

This reduces peak accelerator memory used by averaged parameter copies, but it
has the following trade-offs:

* CPU memory must hold the averaged parameters until the quorum commits.
* Device-to-host and host-to-device transfers add synchronization overhead.
* Waiting for each parameter all-reduce serializes the synchronization path and
  may reduce throughput.
* Only averaged model parameters are staged. Model weights, gradients, and
  optimizer state are not offloaded.
