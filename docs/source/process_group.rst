.. automodule:: torchft.process_group
    :members:
    :undoc-members:
    :show-inheritance:

Native reconfiguration
----------------------

``Manager`` also accepts native PyTorch process groups initialized with
``enable_reconfigure=True`` (PyTorch 2.14 or newer, with a supporting backend).
Unlike the legacy torchft backend wrappers, these groups retain their identity
and native collective implementations when the replica membership changes.
torchft process groups implement the same ``supports_reconfigure``,
``get_reconfigure_handle`` and ``reconfigure(ReconfigureOptions)`` API; the
legacy ``configure()`` method has been removed.
``torchft.process_group.ReconfigureOptions`` is a shim on PyTorch older than
2.14.

A torchft handle is JSON containing the group's ID, the address of a
rendezvous TCPStore hosted by the group, and the global rank set by
``set_rank_info``. A group's rank is the index of its own handle and all ranks
rendezvous on the first handle's store. Without a Manager, use
``reconfigure_with_store`` to exchange handles through a shared store.

For example, for one process per replica, with a separately provisioned native
bootstrap TCPStore shared by all replicas:

.. code-block:: python

    import torch.distributed as dist
    from torchft import Manager

    # replica_slot is a distinct initial rank for each replica; it is not the
    # replica-local RANK used by Manager. Communicators connect at the first quorum.
    native_store = dist.TCPStore(
        bootstrap_host, bootstrap_port, is_master=False, wait_for_workers=False
    )
    dist.init_process_group(
        "gloo",
        store=native_store,
        rank=replica_slot,
        world_size=num_replica_slots,
        enable_reconfigure=True,
    )
    manager = Manager(
        pg=dist.group.WORLD,
        load_state_dict=model.load_state_dict,
        state_dict=model.state_dict,
        min_replica_size=2,
    )

As with legacy groups, configure ``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``,
``WORLD_SIZE``, and ``TORCHFT_LIGHTHOUSE`` for the replica-local manager
rendezvous. A TCPStore must be running at the configured address.

Manager sends a fresh handle with each quorum request. The lighthouse returns
the handles of the same group rank, ordered by replica rank. The quorum ID is
used as the communicator UUID. Manager waits for
``ProcessGroup.reconfigure`` before using the group. Reconfiguration uses
Manager's ``timeout``; configure the native group's collective timeout when
creating it.

The native Gloo and nccl2 backends currently rendezvous using their original
bootstrap store. All replicas must use the same native store namespace and
distinct initial ranks; independent HashStores or replica-local TCPStores will
not work. This bootstrap store must remain available across failures.

Use a dedicated native group for cross-replica communication: its rank and size
change with the quorum. Do not reuse a fixed intra-replica sharding group.
Continue using torchft's managed collectives/DDP integration for error reporting
and gradient normalization; raw native collectives do not swallow errors or
report them to Manager. The caller owns native group shutdown.
