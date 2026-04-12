"""Communication protocol and backends for multi-GPU halo exchange.

Defines a :class:`Communicator` protocol and two concrete implementations:

* :class:`P2PCommunicator` -- CuPy peer-to-peer copies via NVLink
  (single-node, requires ``deviceCanAccessPeer``).
* :class:`NCCLCommunicator` -- NCCL-based halo exchange (multi-node).

:func:`auto_select_communicator` probes GPU topology and picks the best
backend automatically.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Communicator(Protocol):
    """Protocol for GPU halo-exchange backends."""

    def send_halo(
        self,
        buf,
        dst_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None: ...

    def recv_halo(
        self,
        buf,
        src_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None: ...

    def exchange_halos(
        self,
        partitions: list,
        halo_width: int,
        axis: int,
    ) -> None: ...

    def barrier(self) -> None: ...


class P2PCommunicator:
    """CuPy peer-to-peer communicator via NVLink (single-node)."""

    def __init__(self, num_gpus: int) -> None:
        self.num_gpus = num_gpus
        self._enable_p2p()

    def _enable_p2p(self) -> None:
        """Enable peer access between all GPU pairs."""
        import cupy as cp

        for i in range(self.num_gpus):
            for j in range(self.num_gpus):
                if i != j:
                    with cp.cuda.Device(i):
                        try:
                            cp.cuda.runtime.deviceEnablePeerAccess(j)
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass  # already enabled

    def send_halo(
        self,
        buf,
        dst_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None:
        """Send halo region from *buf* to *dst_gpu*."""
        import cupy as cp

        ndim = buf.ndim
        slices: list[slice] = [slice(None)] * ndim
        if side == "right":
            slices[axis] = slice(-2 * halo_width, -halo_width)
        else:  # "left"
            slices[axis] = slice(halo_width, 2 * halo_width)

        src_data = buf[tuple(slices)]
        with cp.cuda.Device(dst_gpu):
            # The receiver will pick this up via recv_halo
            return cp.asarray(src_data)

    def recv_halo(
        self,
        buf,
        src_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None:
        """Receive halo region into *buf* from *src_gpu*."""
        # In practice the send_halo/recv_halo pair is fused in
        # exchange_halos below; this stub is for the protocol.
        pass

    def exchange_halos(
        self,
        partitions: list,
        halo_width: int,
        axis: int,
    ) -> None:
        """Exchange halos between adjacent partitions along *axis*.

        For each pair of neighbours ``(i, i+1)``:
        - right halo of partition[i] -> left halo of partition[i+1]
        - left halo of partition[i+1] -> right halo of partition[i]
        """
        import cupy as cp

        ndim = partitions[0].ndim
        n = len(partitions)

        for i in range(n - 1):
            j = i + 1

            # -- partition[i] right boundary -> partition[j] left halo --
            src_sl = [slice(None)] * ndim
            src_sl[axis] = slice(-2 * halo_width, -halo_width)
            dst_sl = [slice(None)] * ndim
            dst_sl[axis] = slice(0, halo_width)

            with cp.cuda.Device(partitions[j].device.id):
                partitions[j][tuple(dst_sl)] = cp.asarray(
                    partitions[i][tuple(src_sl)]
                )

            # -- partition[j] left boundary -> partition[i] right halo --
            src_sl2 = [slice(None)] * ndim
            src_sl2[axis] = slice(halo_width, 2 * halo_width)
            dst_sl2 = [slice(None)] * ndim
            dst_sl2[axis] = slice(-halo_width, None)

            with cp.cuda.Device(partitions[i].device.id):
                partitions[i][tuple(dst_sl2)] = cp.asarray(
                    partitions[j][tuple(src_sl2)]
                )

    def barrier(self) -> None:
        """Synchronize all GPUs."""
        import cupy as cp

        for i in range(self.num_gpus):
            with cp.cuda.Device(i):
                cp.cuda.Device(i).synchronize()


class NCCLCommunicator:
    """NCCL communicator for multi-node halo exchange."""

    def __init__(self, num_gpus: int) -> None:
        self.num_gpus = num_gpus
        self._comms = None
        self._init_nccl()

    def _init_nccl(self) -> None:
        """Initialize NCCL communicators."""
        try:
            from cupy.cuda import nccl

            uid = nccl.get_unique_id()
            self._comms = []
            for i in range(self.num_gpus):
                comm = nccl.NcclCommunicator(self.num_gpus, uid, i)
                self._comms.append(comm)
        except Exception:
            # NCCL not available; exchange_halos will raise
            self._comms = None

    def send_halo(
        self,
        buf,
        dst_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None:
        """Send halo via NCCL."""
        if self._comms is None:
            raise RuntimeError("NCCL not initialized")
        # NCCL send/recv requires multi-process; placeholder for now

    def recv_halo(
        self,
        buf,
        src_gpu: int,
        axis: int,
        side: str,
        halo_width: int,
        stream=None,
    ) -> None:
        """Receive halo via NCCL."""
        if self._comms is None:
            raise RuntimeError("NCCL not initialized")

    def exchange_halos(
        self,
        partitions: list,
        halo_width: int,
        axis: int,
    ) -> None:
        """Exchange halos using NCCL group calls."""
        if self._comms is None:
            raise RuntimeError(
                "NCCL communicators not initialized. "
                "Ensure NCCL is installed and GPUs are available."
            )
        # NCCL group-based halo exchange requires multi-process setup.
        # This implementation provides the interface; full NCCL send/recv
        # requires spawning one process per GPU.
        raise NotImplementedError(
            "NCCL halo exchange requires multi-process setup. "
            "Use P2PCommunicator for single-process multi-GPU."
        )

    def barrier(self) -> None:
        """NCCL barrier (synchronize all GPUs)."""
        import cupy as cp

        for i in range(self.num_gpus):
            with cp.cuda.Device(i):
                cp.cuda.Device(i).synchronize()


def auto_select_communicator(num_gpus: int) -> Communicator:
    """Auto-select the best communicator based on GPU topology.

    Prefers P2P (NVLink) when all GPU pairs support peer access;
    falls back to NCCL otherwise.
    """
    try:
        import cupy as cp

        can_p2p = True
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j and not cp.cuda.runtime.deviceCanAccessPeer(i, j):
                    can_p2p = False
                    break
            if not can_p2p:
                break
        if can_p2p:
            return P2PCommunicator(num_gpus)
    except Exception:
        pass

    return NCCLCommunicator(num_gpus)
