.. automodule:: torchft.checkpointing
    :members:
    :undoc-members:
    :show-inheritance:

Process group transport
-----------------------

``PGTransport`` uses its constructor timeout when a checkpoint send or receive
omits ``timeout`` or passes ``None``. An explicit per-call timeout takes
precedence, including when ``Manager`` passes its configured timeout.

For example, with a configured process group ``pg``:

.. code-block:: python

    from datetime import timedelta

    import torch
    from torchft.checkpointing.pg_transport import PGTransport

    transport = PGTransport(pg, timeout=timedelta(seconds=30), device=torch.device("cpu"))
    transport.send_checkpoint([1], step=1, state_dict=state_dict)
    transport.send_checkpoint([1], step=2, state_dict=state_dict, timeout=timedelta(seconds=60))

The timeout applies to each communication wait, not the total transfer duration.
The underlying process group's timeout is configured separately.

.. autoclass:: torchft.checkpointing.pg_transport.PGTransport
    :members:
