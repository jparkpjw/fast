import os
import mdtraj as md
import numpy as np
from .base_analysis import base_analysis

def _aligned_rmsd_calc(centers, base_struct, rmsd_indices, align_indices):
    rmsds = np.zeros(len(centers))
    for i, center in enumerate(centers):
        center.superpose(base_struct, atom_indices=align_indices)
        rmsds[i] = np.sqrt(3*np.mean((center.xyz[:, rmsd_indices, :] - base_struct.xyz[:, rmsd_indices, :])**2, axis=(1,2)))[0]
    
    return rmsds

class AlignRMSDWrap(base_analysis):
    """Analysis wrapper for calculating state rmsds
    
    Parameters
    ----------
    base_struct : str or md.Trajectory,
        The base structure to compare for RMSD. This
        topology must match the structures to analyse. Can be provided
        as a pdb location or an md.Trajectory object.
    rmsd_indices : str or array,
        The atom indices to use for computing RMSD. Can be
        provided as a data file to load or an array.
    align_indices : str or array,
        The atom indices to use for alignment before computing RMSD. Can be
        provided as a data file to load or an array.
    
    Attributes
    ----------
    output_name : str,
        The file containing ranking results.
    """
    def __init__(
            self, base_struct, rmsd_indices=None, align_indices=None):
        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = self.base_struct
        else:
            self.base_struct_md = md.load(base_struct)
        # determine rmsd indices
        self.rmsd_indices = rmsd_indices
        if type(rmsd_indices) is str:
            self.rmsd_indices_vals = np.loadtxt(rmsd_indices, dtype=int)
        else:
            self.rmsd_indices_vals = self.rmsd_indices
        # determine align indices
        self.align_indices = align_indices
        if type(align_indices) is str:
            self.align_indices_vals = np.loadtxt(align_indices, dtype=int)
        else:
            self.align_indices_vals = self.align_indices

    @property
    def class_name(self):
        return "AlignRMSDWrap"
        
    @property
    def config(self):
        return {
            'base_struct': self.base_struct,
            'rmsd_indices': self.rmsd_indices,
            'align_indices': self.align_indices,
        }

    @property
    def base_output_name(self):
        return "rmsd_per_state"
    
    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load(
                "./data/full_centers.xtc", top=self.base_struct_md)
            struct_sub = self.base_struct_md
            
            # calculate and save rmsds
            rmsds = _aligned_rmsd_calc(centers, struct_sub, 
                rmsd_indices=self.rmsd_indices_vals, align_indices=self.align_indices_vals)
            np.save(self.output_name, rmsds)