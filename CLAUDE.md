# BAR Map Sync — project notes

## rocks30 needs a per-asset V-flip (cuspbr exception)

`rocks30` is the **only** BAR feature with `cuspbr = "yes"` in its featureDef
(`rocks30.lua`). Its s3o files are authored with V=0 at the atlas **bottom**,
inverse to every other BAR feature (ad0 trees, fir_tree, crystals, allpines
etc. all have V=0 at the atlas top).

BAR's `featurepbr` shader bin (which `cuspbr=yes` routes to via `cus_gl4.lua`)
applies an implicit V-flip in-shader that the standard `feature` bin does not.
Our glTF export has no such shader, so we have to flip V at build time —
otherwise the rocks30 UVs land in the wrong tile of their 3×2 atlas and every
rock looks like it has a random projected texture (it's sampling a neighbour
tile's content or the padding between tiles).

### How it's implemented

In [glb_feature_builder.py](glb_feature_builder.py):

- `FeatureGLBBuilder.__init__` sets `self._flip_v = False` by default.
- `add_piece_mesh` writes `(s, 1 - t)` into the UV accessor when `_flip_v` is
  True, otherwise `(s, t)` as normal.
- `build_feature_glb` sets `builder._flip_v = True` when the s3o basename
  contains `rocks30`.

**Do not** change the default for other features — V=t is correct for every
other asset and flipping them breaks ad0/trees/crystals/allpines (verified).

### If BAR ever adds another `cuspbr = "yes"` feature

Extend the detection in `build_feature_glb` to parse the `.lua` featureDef for
`cuspbr = "yes"` instead of matching on the `rocks30` name.
