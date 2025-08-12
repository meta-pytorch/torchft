:github_url: https://github.com/meta-pytorch/torchft/assumptions-and-recommendations

Assumptions and Recommendations
===============================

There are a couple of assumptions we make without which this protocol guarantee stops to hold.

* The state machine replication is done in a similar mechanism as any consensus protocol with a leader e.g. Raft. The leader, **lighthouse**, is a single point of failure so users should take care to make sure it's running on a dedicated instance to avoid noisy neighbor induced failures and automatic restarts to maximize availability.

* The **lighthouse** dynamically changes replica membership based on what replicas are healthy and performs a quorum only on the set of replicas that are healthy.

  - This reconfiguration is only safe if there is one replica that ends up overlapping in both memberships. We currently have a flag `join_timeout_ms` flag on the **lighthouse**, to make it more likely to have replicas from the previous membership join.

  - If there are multiple instances of **lighthouse** and replicas connected to different instances, two distinct quorums can be reached because the membership is different for each instance of **lighthouse**. So make sure replicas connect to the same instance of **lighthouse**. A split brain can also be avoided by setting `min_replicas` to be at least as much as a majority of replicas (though a majority from the previous membership should also be able to connect to the same instance of lighthouse as the current membership for the overlap to take place).

* The `min_replicas` setting also determines how many batches the gradients will be averaged over for **DDP** in the worst case. So set this to a reasonable value to reduce the variance in model parameter updates.

* If a call to **quorum** times out, the replica crashes, so it is taken out of the membership group and will be unable to participate in consensus. As is typical for consensus, we do require a majority of replicas to be healthy. There are certain situations where these timeouts can happen leading to cascading failures that can bring down a majority. In general, we recommend keeping the `quorum_timeout` settings on **manager** to be a high value, enough for all ranks to finish training until they need to synchronize after a step (or `sync_every` steps when using **DiLoCo** or **LocalSGD**).

* Similarly, the `timeout` setting on the manager should also be high enough to account for the variability in **FSDP** replicas finishing a step.

* When using **NCCL**, collective operations issued by **ProcessGroupNCCL** for **FSDP** can cause deadlocks (known issue), causing it to kill the training process.

  - Make sure this doesn’t happen for a majority of replicas or try using a different process group for **DDP** e.g. **Gloo**.

  - Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` to be lower than the `quorum_timeout`, so the process can come back up before the quorum times out for other replicas. The timeout for operations issued for **DDP** operations is determined by the `timeout` setting, so make sure `quorum_timeout` is also greater than that.

* The support for **DiLoCo** is currently in alpha – the implementation is yet to be tested at scale.
