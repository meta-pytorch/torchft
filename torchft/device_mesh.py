from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from torch.distributed import (
    DeviceMesh,
    ProcessGroup as BaseProcessGroup,
    get_rank,
    init_device_mesh,
)
from torch.distributed.tensor.device_mesh import _mesh_resources

from torchft.manager import Manager

if TYPE_CHECKING:
    from torchft.process_group import ManagedProcessGroup, ProcessGroup


def extend_device_mesh(
    mesh: DeviceMesh, pg: ProcessGroup, name: str = "dp", dim: int = 0
) -> DeviceMesh:
    """
    This is a helper method to extend a traditional DeviceMesh with a torchft ProcessGroup for usage with DeviceMesh based APIs such as FSDPv2 with hybrid sharding.

    Resizable PGs aren't natively supported by DeviceMesh so we lie to
    DeviceMesh and say the PG is world size 1. This is fine as long as any
    numeric scaling is handled at the PG level.

    Args:
        mesh: The DeviceMesh to extend
        pg: The ProcessGroup to add to the mesh
        name: The name of the new dimension
        dim: The dimension to add the ProcessGroup to
    """
    groups = mesh.get_all_groups()
    groups.insert(dim, pg)
    mesh_dim_names = list(mesh.mesh_dim_names or [])
    mesh_dim_names.insert(dim, name)

    return DeviceMesh.from_group(
        group=groups,
        device_type=mesh.device_type,
        mesh=mesh.mesh.unsqueeze(dim),
        mesh_dim_names=tuple(mesh_dim_names),
    )


class ManagedDeviceMesh(DeviceMesh):
    replicate_pg_singleton: Optional["ManagedProcessGroup"] = None

    def __init__(
        self,
        mesh: Optional[DeviceMesh],
        mesh_dim_names: tuple[str, ...],
        replicate_pg: ManagedProcessGroup,
        replicate_dim: int,
        parent: Optional["ManagedDeviceMesh"],
    ) -> None:
        if mesh is None and parent is None:
            raise ValueError(
                "ManagedDeviceMesh doesn't support both mesh and parent are None."
            )
        self.mesh = mesh
        self.mesh_dim_names = mesh_dim_names
        self.replicate_pg = replicate_pg
        self.replicate_dim = replicate_dim
        self.replicate_dim_name: str = mesh_dim_names[replicate_dim]
        self.parent = parent
        self.flatten_meshes: Dict[str, DeviceMesh] = {}
        self.device_type: str
        if mesh is not None:
            self.device_type = mesh.device_type
        else:
            assert parent is not None
            self.device_type = parent.device_type
        self._flatten_mesh_list: tuple[DeviceMesh, ...] = tuple()
        self._thread_id: Optional[int] = None
        self._hash: Optional[int] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["replicate_pg"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        assert self.replicate_pg_singleton is not None
        self.replicate_pg = self.replicate_pg_singleton

    def __getitem__(self, mesh_dim_names: Union[str, tuple[str, ...]]) -> DeviceMesh:
        if isinstance(mesh_dim_names, str):
            if mesh_dim_names == self.replicate_dim_name:
                res_submesh = ManagedDeviceMesh(
                    mesh=None,
                    mesh_dim_names=(mesh_dim_names,),
                    replicate_pg=self.replicate_pg,
                    replicate_dim=0,
                    parent=self,
                )
            elif mesh_dim_names in self.flatten_meshes:
                res_submesh = self.flatten_meshes[mesh_dim_names]
            else:
                assert self.mesh is not None
                res_submesh = self.mesh[mesh_dim_names]
        else:
            assert isinstance(mesh_dim_names, tuple)
            if self.replicate_dim_name not in mesh_dim_names:
                assert self.mesh is not None
                res_submesh = self.mesh[mesh_dim_names]
            else:
                mesh_dim_names_wo_replicate = tuple(
                    n for n in mesh_dim_names if n != self.replicate_dim_name
                )
                assert self.mesh is not None
                res_submesh = ManagedDeviceMesh(
                    self.mesh[mesh_dim_names_wo_replicate],
                    mesh_dim_names,
                    self.replicate_pg,
                    mesh_dim_names.index(self.replicate_dim_name),
                    parent=self,
                )

        # TODO: find a better way to do this that doesn't depend on device mesh
        # internals
        root = _mesh_resources.get_root_mesh(self)
        _mesh_resources.child_to_root_mapping[res_submesh] = root

        return res_submesh

    def _real_mesh_dim(self, mesh_dim: int) -> int:
        return mesh_dim - 1 if mesh_dim > self.replicate_dim else mesh_dim

    def get_group(self, mesh_dim: Optional[Union[int, str]] = None) -> BaseProcessGroup:
        if isinstance(mesh_dim, str):
            dim = self.mesh_dim_names.index(mesh_dim)
        else:
            dim = 0 if mesh_dim is None else int(mesh_dim)

        if mesh_dim is None:
            return self.replicate_pg
        elif dim == self.replicate_dim:
            return self.replicate_pg
        else:
            assert self.mesh is not None
            return self.mesh.get_group(self._real_mesh_dim(dim))

    def _flatten(self, mesh_dim_name: Optional[str]) -> "DeviceMesh":
        flatten_mesh = _FlattenDeviceMesh(self)
        if mesh_dim_name is None:
            raise ValueError("ManagedDeviceMesh._flatten requires `mesh_dim_name`")
        if self.parent is None:
            self.flatten_meshes[mesh_dim_name] = flatten_mesh
        else:
            self.parent.flatten_meshes[mesh_dim_name] = flatten_mesh
        return flatten_mesh

    def size(self, mesh_dim: Optional[int] = None) -> int:
        replicate_pg_size = self.replicate_pg.size()
        # We have to lie to the users if there are zero particpants.
        # This is possible during the initialization stage of training.
        replicate_pg_size = 1 if replicate_pg_size == 0 else replicate_pg_size
        if mesh_dim is None:
            if self.mesh is None:
                return replicate_pg_size
            else:
                assert self.mesh is not None
                return self.mesh.size() * replicate_pg_size
        elif mesh_dim == self.replicate_dim:
            return replicate_pg_size
        else:
            assert self.mesh is not None
            return self.mesh.size(self._real_mesh_dim(mesh_dim))

    @property
    def ndim(self) -> int:
        assert self.mesh is not None
        return self.mesh.ndim + 1

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.mesh is not None
        ret: list[int] = list(self.mesh.shape)
        ret.insert(self.replicate_dim, self.replicate_pg.size())
        return tuple(ret)

    def get_rank(self) -> int:
        assert self.mesh is not None
        return self.mesh.get_rank()

    def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        if isinstance(mesh_dim, str):
            dim = self.mesh_dim_names.index(mesh_dim)
        else:
            dim = 0 if mesh_dim is None else int(mesh_dim)

        if mesh_dim is None:
            if self.mesh is None:
                return get_rank(self.replicate_pg)

            assert self.replicate_dim == 0, "replicate_dim must be the first one"
            assert self.mesh is not None
            other_dim_size = self.mesh.size()
            assert self.mesh is not None
            other_dim_rank = self.mesh.get_local_rank()
            replicate_pg_rank = get_rank(self.replicate_pg)
            return other_dim_size * replicate_pg_rank + other_dim_rank
        elif dim == self.replicate_dim:
            return get_rank(self.replicate_pg)
        else:
            assert self.mesh is not None
            return self.mesh.get_local_rank(self._real_mesh_dim(dim))

    def get_coordinate(self) -> Optional[list[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        """
        assert self.mesh is not None
        coordinate = (
            self.mesh._coordinate_on_dim if self.mesh._coordinate_on_dim else None
        )
        if not coordinate:
            return coordinate

        # We need to copy be cause we are going to modify the coordinate.
        coordinate = coordinate.copy()
        coordinate.insert(get_rank(self.replicate_pg), self.replicate_dim)
        return coordinate

    def get_all_groups(self) -> list[BaseProcessGroup]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"ManagedDeviceMesh(mesh={self.mesh})"

    def __hash__(self) -> int:
        # lazily compute hash
        if not self._hash:
            self._hash = hash(
                (
                    self.mesh,
                    self.mesh_dim_names,
                    self.replicate_pg,
                    self.replicate_dim,
                    self.replicate_dim_name,
                    self.parent,
                    self.device_type,
                )
            )
        return self._hash


class _FlattenDeviceMesh(DeviceMesh):
    def __init__(self, managed_mesh: ManagedDeviceMesh) -> None:
        self.managed_mesh = managed_mesh

    def __getitem__(self, mesh_dim_names: Union[str, tuple[str, ...]]) -> DeviceMesh:
        raise NotImplementedError

    def get_group(self, mesh_dim: Optional[Union[int, str]] = None) -> BaseProcessGroup:
        raise NotImplementedError

    def _flatten(self, mesh_dim_name: Optional[str]) -> "DeviceMesh":
        raise NotImplementedError

    def size(self, mesh_dim: Optional[int] = None) -> int:
        assert mesh_dim is None
        return self.managed_mesh.size()

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def get_rank(self) -> int:
        raise NotImplementedError

    def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        assert mesh_dim is None
        return self.managed_mesh.get_local_rank()

    def get_all_groups(self) -> list[BaseProcessGroup]:
        raise NotImplementedError


def ft_init_device_mesh(
    *,
    device_type: str,
    mesh_shape: Union[tuple[int, ...], list[int]],
    mesh_dim_names: Union[tuple[str, ...], list[str]],
    replicate_dim: int,
    manager: "Manager",
) -> "ManagedDeviceMesh":
    # We need to mislead DeviceMesh into thinking that replicate_dim has only
    # 1 rank.
    _mesh_shape = list(mesh_shape)
    _mesh_shape.pop(replicate_dim)
    _mesh_dim_names = list(mesh_dim_names)
    _mesh_dim_names.pop(replicate_dim)
    mesh = init_device_mesh(
        device_type,
        mesh_shape=tuple(_mesh_shape),
        mesh_dim_names=tuple(_mesh_dim_names),
    )

    from torchft.process_group import ManagedProcessGroup

    replicate_pg = ManagedProcessGroup(manager)
    replicate_pg.register(mesh_dim_names[replicate_dim])

    ManagedDeviceMesh.replicate_pg_singleton = replicate_pg

    return ManagedDeviceMesh(
        mesh=mesh,
        mesh_dim_names=tuple(mesh_dim_names),
        replicate_pg=replicate_pg,
        replicate_dim=replicate_dim,
        parent=None,
    )
