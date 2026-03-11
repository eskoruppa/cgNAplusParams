from __future__ import annotations

from pathlib import Path
import os
import warnings
import numpy as np
# from ..IOPolyMC import iopolymc as iopmc
from ..build_conf import build_conf
from .pdb import gen_pdb
1
# from .. utils.path_methods import create_relative_path


def visualize_chimerax(
    base_fn: str | Path,
    seq: str,
    cg: int,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    first_cg: int = 0,
    bead_radius: float | None = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False
) -> None:
    # Validate exactly one is provided
    if shape_params is None and poses is None:
        raise ValueError("Either 'shape_params' or 'poses' must be provided")
    if shape_params is not None and poses is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'poses', choose one")
    if shape_params is not None:
        if shape_params.ndim !=2:
            raise ValueError(f"shape_params must be a 2D array, got shape {shape_params.shape}")
        if shape_params.shape[1] != 6:
            raise ValueError(f"shape_params must have shape (N,6), got shape {shape_params.shape}")
    if poses is not None:
        if poses.ndim !=3:
            raise ValueError(f"poses must be a 3D array, got shape {poses.shape}")
        if poses.shape[1:] != (4,4):
            raise ValueError(f"poses must have shape (N,4,4), got shape {poses.shape}")
    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if poses is None:
        poses = build_conf(shape_params)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    poses2pdb(pdbfn, poses, seq)
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[first_cg::cg]
    _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, poses, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius is not None and bead_radius > 0:
        spheres = np.zeros((len(cgposes),4))
        spheres[:,:3] = cgposes[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)
    
    
def visualize_pdb(
    base_fn: str | Path,
    seq: str,
    *,
    shape_params: np.ndarray | None = None,
    poses: np.ndarray | None = None,
    disc_len: float = 0.34,
) -> None:
    # Validate exactly one is provided
    if shape_params is None and poses is None:
        raise ValueError("Either 'shape_params' or 'poses' must be provided")
    if shape_params is not None and poses is not None:
        raise ValueError("Cannot provide both 'shape_params' and 'poses', choose one")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    if poses is None:
        poses = build_conf(shape_params)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    poses2pdb(pdbfn, poses, seq)
    

def cgvisual(
    base_fn: str | Path,
    shape_params: np.ndarray,
    seq: str,
    cg: int,
    start_id: int = 0,
    bead_radius: float = None,
    disc_len: float = 0.34,
    include_bps_triads: bool = False
):

    warnings.warn(
        "cgvisual is deprecated and will be removed in a future release. "
        "Please use visualize_pdb or visualize_xyz instead.",
        DeprecationWarning
    )

    if cg < 1:
        raise ValueError(f"cg must be >= 1, got {cg}")
    
    # create relative path
    base_fn = Path(base_fn)
    # Check if base_fn already has an extension
    if base_fn.suffix:
        raise ValueError(f"base_fn should not have an extension, got: {base_fn.suffix}")
    
    outdir = base_fn.parent
    if not outdir.exists():
        os.makedirs(outdir)
        
    # calculate triads
    poses = build_conf(shape_params)
            
    # generate pdb file
    pdbfn = base_fn.with_suffix('.pdb')
    poses2pdb(pdbfn, poses, seq)
    
    # create bild file for triads
    bildfn = base_fn.with_name(base_fn.name + '_triads.bild')
    cgposes = poses[start_id::cg]
    _triads2bild(bildfn, cgposes, alpha=1., scale=1, nm2aa=True, decimals=2)
    
    # create base pair step triad bild file
    bps_triads_bildfn = None
    if include_bps_triads and cg > 1:
        bps_triads_bildfn = base_fn.with_name(base_fn.name + '_bps_triads.bild')
        _triads2bild(bps_triads_bildfn, poses, alpha=1., scale=1, nm2aa=True, decimals=2)
        
    # create chimera cxc file
    cxcfn = base_fn.with_suffix('.cxc')
    if bead_radius > 0:
        spheres = np.zeros((len(cgposes),4))
        spheres[:,:3] = cgposes[:,:3,3]
        spheres[:,3] = bead_radius
    else:
       spheres = None
    _chimeracxc(cxcfn, pdbfn, triadfn=bildfn, spheres=spheres, nm2aa= True, decimals=2, additional_triadfn=bps_triads_bildfn)

   
def _chimeracxc(
    fn: Path | str,
    pdbfn: Path | str,
    triadfn: Path | str | None = None,
    spheres: np.ndarray | None = None,
    nm2aa: bool = True,
    decimals: int = 2,
    additional_triadfn: Path | str | None = None
):
    
    fn = Path(fn)
    if fn.suffix.lower() != '.cxc':
        fn = fn.with_suffix('.cxc')
    
    pdbfn = Path(pdbfn)
    modelnum = 0
    sphereids = []
    with open(fn,'w') as f:
        
        f.write(f'# scene settings\n')
        # white background
        f.write(f'set bgColor white\n')
        # simple lighting
        f.write(f'lighting simple\n')    
        # set silhouettes
        f.write(f'graphics silhouettes true color black width 1.5\n') 
        # open pdb twice
        f.write(f'\n# load pdb\n')
        f.write(f'open {pdbfn.name}\n') 
        f.write(f'open {pdbfn.name}\n')
        modelnum += 2 
        # dna visuals
        f.write(f'\n# set DNA visuals\n')
        f.write(f'style ball\n')
        f.write(f'nucleotides atoms\n')
        f.write(f'color white target a\n')
        f.write(f'color light gray target c\n')
        f.write(f'cartoon style nucleic xsect oval width 3.0 thick 1.2\n')
        f.write(f'hide #2 atoms\n')
        f.write(f'hide #1 cartoons\n')
        
        # open triads
        if triadfn is not None:
            triadfn = Path(triadfn)
            modelnum += 1 
            f.write(f'\n# load triads BILD\n')
            f.write(f'open {triadfn.name}\n') 
            
        # open triads
        if additional_triadfn is not None:
            additional_triadfn = Path(additional_triadfn)
            modelnum += 1 
            f.write(f'\n# load additional triads BILD\n')
            f.write(f'open {additional_triadfn.name}\n') 
        
        if spheres is not None:
            nm2aafac = 1
            if nm2aa:
                nm2aafac = 10
            def pt2str(pt):
                return ','.join([f'{np.round(p*nm2aafac,decimals=decimals)}' for p in pt[:3]])

            f.write(f'\n# Genergate spheres\n')
            for shid,sphere in enumerate(spheres):
                modelnum += 1
                sphereids.append(modelnum)
                f.write(f'shape sphere radius {np.round(sphere[3]*nm2aafac,decimals=decimals)} center {pt2str(sphere)} name sph{shid+1}\n')
                f.write(f'transparency #{modelnum} 75\n')

              
def _triads2bild(
    fn: Path | str,
    poses: np.ndarray,
    alpha: float = 1.,
    ucolor: str = 'default',
    vcolor: str = 'default',
    tcolor: str = 'default',
    scale: float = 1,
    nm2aa: bool = True,
    decimals: int = 2
): 
    
    if ucolor == 'default':
        ucolor = [64/255,91/255,4/255]
        # ucolor = [0.15294118, 0.47843137, 0.17647059]
    if vcolor == 'default':
        vcolor = [61/255,88/255,117/255]
        # vcolor = [0.17647059, 0.15294118, 0.47843137]
    if tcolor == 'default':
        tcolor = [153/255,30/255,46/255]
        # tcolor = [0.47843137, 0.17647059, 0.15294118]
        
    fn = Path(fn)
    if fn.suffix.lower() != '.bild':
        fn = fn.with_suffix('.bild')
        
    dist = np.mean(np.linalg.norm(poses[1:,:3,3]-poses[:-1,:3,3],axis=1))
    size = dist * 0.66 * scale
    nm2aafac = 1
    if nm2aa:
        nm2aafac = 10
    
    def _color2str(color):
        if isinstance(color,str):
            return color
        if hasattr(color, '__iter__') and len(color) == 3:
            return ' '.join([f'{c}' for c in color])
        raise ValueError(f'Invalid color {color}')
    
    def pt2str(pt):
        return ' '.join([f'{np.round(p*nm2aafac,decimals=decimals)}' for p in pt])
    
    shapestr = f'{np.round(size*nm2aafac/20,decimals=decimals)} {np.round(size*nm2aafac/20*2,decimals=decimals)} 0.70'
    with open(fn,'w') as f:
        if alpha < 1.0:
            f.write(f'.transparency {1-alpha}\n')
        for i,tau in enumerate(poses):
            tau = tau[:3]
            f.write(f'# triad {i+1}\n')
            f.write(f'.color {_color2str(ucolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,0]*size)} {shapestr}\n')
            f.write(f'.color {_color2str(vcolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,1]*size)} {shapestr}\n')
            f.write(f'.color {_color2str(tcolor)}\n')
            f.write(f'.arrow {pt2str(tau[:,3])} {pt2str(tau[:,3]+tau[:,2]*size)} {shapestr}\n')
    return fn
    
def params2pdb(fn: Path | str, params: np.ndarray, seq: str) -> None:
    poses2pdb(fn,build_conf(params),seq)
    
def poses2pdb(
    fn: Path | str, 
    poses: np.ndarray, 
    seq: str
    ) -> None:
    fn = Path(fn)
    if fn.suffix.lower() != '.pdb':
        fn = fn.with_suffix('.pdb')
    if len(poses) != len(seq):
        raise ValueError(f'Dimension of poses ({poses.shape}) and seq ({len(seq)}) do not match.')
    
    print(poses.shape)
    gen_pdb(str(fn), poses, sequence=seq, center=False)
    
    