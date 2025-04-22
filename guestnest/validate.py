"""
GUESTNEST
Copyright (C) 2025  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def _validate_array(
    array_name: str,
    array: np.ndarray,
    dtype: np.dtype = None,
    shape: tuple[int] = None,
    ndim: int = None,
) -> None:
    """
    Validates the data type (dtype), shape, and dimensionality of an array.

    Args:
        array_name (str): Name of the array to print in error messages.
        array (np.ndarray): Array.
        dtype (np.dtype, optional): Expected data type (dtype) of the array;
            if None, the data type is not validated. Defaults to None.
        shape (tuple[int], optional): Expected shape of the array; if None,
            the shape is not validated. Defaults to None.
        ndim (int, optional): Expected dimensionality of the array; if None,
            the dimensionality is not validated. Defaults to None.

    Raises:
        ValueError: If `array` is not an np.ndarray instance.
        ValueError: If `array.shape` is not equal to `shape` and `shape` is
            not None.
        ValueError: If `array.ndim` is not equal to `ndim` and `ndim` is
            not None.
    """
    
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f'{array_name} should be a np.ndarray instance; got {array}'
        )
    if dtype is not None and array.dtype != dtype:
        raise ValueError(
            f'{array_name} should have a data type of {dtype}; got an array '
            f'with a data type of {array.dtype}'
        )
    if shape is not None and array.shape != shape:
        raise ValueError(
            f'{array_name} should have a shape of {shape}; got an array with '
            f'a shape of {array.shape}'
        )
    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f'{array_name} should have {ndim} dimensions; got an array with '
            f'{array.ndim} dimensions'
        )
