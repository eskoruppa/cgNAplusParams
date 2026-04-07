"""Top-level cgNA+ object.

Provides :class:`CGNAPlus`, which combines a
:class:`~cgnaplus_params.CGNAPlusParams` model with a
:class:`~cgnaplus_conf.CGNAPlusConf` configuration and exposes a unified API
for building, accessing, and exporting cgNA+ structures.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cgnaplus_conf import CGNAPlusConf
    from .cgnaplus_params import CGNAPlusParams


class CGNAPlus:
    """Top-level cgNA+ object unifying model parameters and structural configurations.

    Combines a :class:`CGNAPlusParams` instance (sequence-dependent ground
    state and stiffness matrix) with a :class:`CGNAPlusConf` instance (one or
    more pose snapshots).  Both components are optional, enabling use cases
    such as reading a PDB without an associated parameter model.

    Construction
    ------------
    Use the class methods rather than ``__init__`` directly:

    * :meth:`from_ground_state` — from a sequence string (ground state only)
    * :meth:`from_params`       — from a sequence + dynamic deformation array
    * :meth:`from_pdb`          — fit to a single PDB file
    * :meth:`from_trajectory`   — fit to an MD trajectory (multiple snapshots)
    """

    def __init__(
        self,
        params: "CGNAPlusParams | None" = None,
        conf:   "CGNAPlusConf | None"   = None,
    ) -> None:
        self._params = params
        self._conf   = conf

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ground_state(
        cls,
        sequence: str,
        *,
        parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
        orientation: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        origin:      np.ndarray | list = np.zeros(3),
        **kwargs,
    ) -> "CGNAPlus":
        """Build a cgNA+ ground-state configuration from *sequence*.

        Parameters
        ----------
        sequence:
            Nucleotide sequence string, e.g. ``"ATCGATCG"``.
        parameter_set_name:
            Name of the cgNA+ parameter-set ``.mat`` file (without path).
        orientation, origin:
            Initial reference frame; passed to the chain builder.
        **kwargs:
            Additional keyword arguments forwarded to
            :class:`~cgnaplus_params.CGNAPlusParams` (e.g.
            ``euler_definition``, ``group_split``, …).
        """
        from .cgnaplus_params import CGNAPlusParams
        from .cgnaplus_conf import CGNAPlusConf
        params = CGNAPlusParams(
            sequence,
            parameter_set_name=parameter_set_name,
            **kwargs,
        )
        conf = CGNAPlusConf.from_params(
            params,
            orientation=orientation,
            origin=origin,
        )
        return cls(params=params, conf=conf)

    @classmethod
    def from_params(
        cls,
        sequence: str,
        dynamic:  np.ndarray,
        *,
        parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
        orientation: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        origin:      np.ndarray | list = np.zeros(3),
        **kwargs,
    ) -> "CGNAPlus":
        """Build a cgNA+ deformed configuration from *sequence* and *dynamic*.

        Parameters
        ----------
        sequence:
            Nucleotide sequence string.
        dynamic:
            Deformation array, shape ``(n_dof, 6)`` for a single snapshot or
            ``(n_snap, n_dof, 6)`` for an ensemble.
        parameter_set_name, orientation, origin, **kwargs:
            Forwarded to :class:`~cgnaplus_params.CGNAPlusParams`.
        """
        from .cgnaplus_params import CGNAPlusParams
        from .cgnaplus_conf import CGNAPlusConf
        params = CGNAPlusParams(
            sequence,
            parameter_set_name=parameter_set_name,
            **kwargs,
        )
        conf = CGNAPlusConf.from_params(
            params,
            dynamic=dynamic,
            orientation=orientation,
            origin=origin,
        )
        return cls(params=params, conf=conf)

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        sequence: str | None = None,
        *,
        params: "CGNAPlusParams | None" = None,
    ) -> "CGNAPlus":
        """Build a cgNA+ configuration by reading and fitting a PDB file.

        .. note::
            Requires :func:`~cgnaplusparams.io.read_pdb.fit_pdb` to be
            implemented (currently raises ``NotImplementedError``).

        Parameters
        ----------
        pdb_path:
            Path to the PDB file.
        sequence:
            Expected nucleotide sequence; inferred from the PDB when *None*.
        params:
            Optional :class:`~cgnaplus_params.CGNAPlusParams` to attach.
        """
        from .output.read_pdb import fit_pdb
        from .cgnaplus_conf import CGNAPlusConf
        named_poses = fit_pdb(pdb_path, sequence=sequence)
        conf = CGNAPlusConf.from_poses(
            named_poses["bp_poses"],
            watson_base_poses=named_poses.get("watson_base_poses"),
            crick_base_poses=named_poses.get("crick_base_poses"),
            watson_phosphate_poses=named_poses.get("watson_phosphate_poses"),
            crick_phosphate_poses=named_poses.get("crick_phosphate_poses"),
            params=params,
            sequence=sequence,
        )
        return cls(params=params, conf=conf)

    @classmethod
    def from_trajectory(
        cls,
        trajectory_path: str | Path,
        topology_path:   str | Path,
        sequence: str | None = None,
        *,
        params: "CGNAPlusParams | None" = None,
        frames: "slice | list[int] | None" = None,
    ) -> "CGNAPlus":
        """Build a multi-snapshot cgNA+ configuration from an MD trajectory.

        .. note::
            Requires :func:`~cgnaplusparams.io.read_pdb.fit_trajectory` to be
            implemented (currently raises ``NotImplementedError``).

        Parameters
        ----------
        trajectory_path:
            Path to the trajectory file.
        topology_path:
            Path to the topology / structure file.
        sequence:
            Expected nucleotide sequence.
        params:
            Optional :class:`~cgnaplus_params.CGNAPlusParams`.
        frames:
            Frame selection: a :class:`slice`, a list of integer indices, or
            *None* for all frames.
        """
        from .output.read_pdb import fit_trajectory
        from .cgnaplus_conf import CGNAPlusConf
        named_poses = fit_trajectory(
            trajectory_path,
            topology_path,
            sequence=sequence,
            frames=frames,
        )
        conf = CGNAPlusConf.from_poses(
            named_poses["bp_poses"],
            watson_base_poses=named_poses.get("watson_base_poses"),
            crick_base_poses=named_poses.get("crick_base_poses"),
            watson_phosphate_poses=named_poses.get("watson_phosphate_poses"),
            crick_phosphate_poses=named_poses.get("crick_phosphate_poses"),
            params=params,
            sequence=sequence,
        )
        return cls(params=params, conf=conf)

    # ------------------------------------------------------------------
    # Delegation to conf
    # ------------------------------------------------------------------

    def get_pose(self, name: str) -> np.ndarray | None:
        """Delegate to :meth:`~cgnaplus_conf.CGNAPlusConf.get_pose`."""
        return self._conf.get_pose(name)

    def get_junction(self, name: str) -> np.ndarray | None:
        """Delegate to :meth:`~cgnaplus_conf.CGNAPlusConf.get_junction`."""
        return self._conf.get_junction(name)

    def get_parameter(self, name: str) -> np.ndarray | None:
        """Delegate to :meth:`~cgnaplus_conf.CGNAPlusConf.get_parameter`."""
        return self._conf.get_parameter(name)

    def __getitem__(self, key: str) -> np.ndarray | None:
        """Shorthand for :meth:`get_pose`."""
        return self._conf.get_pose(key)

    def to_pdb(self, outfn: str, *, snapshot_idx: int = 0) -> None:
        """Delegate to :meth:`~cgnaplus_conf.CGNAPlusConf.to_pdb`."""
        self._conf.to_pdb(outfn, snapshot_idx=snapshot_idx)

    def to_chimerax(self, base_fn: str, *, snapshot_idx: int = 0) -> None:
        """Delegate to :meth:`~cgnaplus_conf.CGNAPlusConf.to_chimerax`."""
        self._conf.to_chimerax(base_fn, snapshot_idx=snapshot_idx)

    def snapshot(self, i: int) -> "CGNAPlus":
        """Return a single-snapshot :class:`CGNAPlus` for frame *i*."""
        return CGNAPlus(params=self._params, conf=self._conf.snapshot(i))

    def __len__(self) -> int:
        return len(self._conf) if self._conf is not None else 0

    def __iter__(self):
        for i in range(len(self)):
            yield self.snapshot(i)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> "CGNAPlusParams | None":
        """The cgNA+ parameter model."""
        return self._params

    @property
    def conf(self) -> "CGNAPlusConf | None":
        """The structural configuration."""
        return self._conf

    @property
    def sequence(self) -> str | None:
        """Nucleotide sequence string."""
        if self._params is not None:
            return self._params.sequence
        if self._conf is not None:
            return self._conf.sequence
        return None

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots stored."""
        return len(self._conf) if self._conf is not None else 0

    @property
    def nbp(self) -> int:
        """Number of base pairs."""
        return self._conf.nbp if self._conf is not None else 0

    def __repr__(self) -> str:
        return (
            f"CGNAPlus(sequence='{self.sequence}', "
            f"n_snapshots={self.n_snapshots}, nbp={self.nbp})"
        )


# Keep the old stub name accessible during any transition period
cgNAplus = CGNAPlus
