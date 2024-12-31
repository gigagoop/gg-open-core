"""
Note
----
PyCharm's display of types is actually rather annoying, and is addressed in the following issue report:
    https://youtrack.jetbrains.com/issue/PY-42486/Improve-clarity-of-displayed-type-hints-for-type-aliases

In the case of `ArrayLike` below, it gets expanded (as an example to):
    def from_rotation_and_origin(rotation: _SupportsArray[dtype] | _NestedSequence[_SupportsArray[dtype]] | bool | ...,
                                 origin: _SupportsArray[dtype] | _NestedSequence[_SupportsArray[dtype]] | bool | ...):
        ...

making it difficult to read.

It would certainly be nicer to have
    def from_rotation_and_origin(rotation: ArrayLike,
                                 origin: ArrayLike):
        ...

Hopefully at some point, this gets updated ;-P.
"""
from pathlib import Path
from typing import List, TypeAlias

from numpy.typing import ArrayLike as NumpyArrayLike

# If you have something like
#     def compute(x: NumpyArrayLike):
#         ...
#
# and use `compute([1, 2, 3])`, PyCharm will show a warning about passing in `[1, 2, 3]`... well, `[1, 2, 3]` *is*
# array like, so to get around this warning, the following type-hint is introduced to quiet PyCharm.
ArrayLike: TypeAlias = NumpyArrayLike | List

# While a `Path` is preferred, using a `str` is really common, this is a way to hint at either being acceptable.
PathLike: TypeAlias = str | Path

