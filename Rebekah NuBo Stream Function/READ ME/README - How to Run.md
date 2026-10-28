
In your stream function folder you will need:

1. stl file of former you are building on, must be flat surface and have no thickness, you can add thickness later
    NOTE:make sure your .stl is subdivided and triangulated properly before you run the stream
    function on it, or else there will not be a feasible wire path for the stream function
2. The B0 field map you are trying to optimize for
3. the "python codes" folder which contains all the helper/main codes you need to run the stream function (main here is design_gradient.py)


How to Run the Stream Function:

1. Specify the parameters you want
  - `FIELD_CSV = 'magnetData.csv'` — the probed B₀ map.
  - `STL_PATH = 'example.stl'` — the former surface, triangulated properly - make sure to change "example" to the name of your stl file
  - `AXIS_MAP = (0, 1, 2)` — identity: the STL already sits in the CSV's frame.
  - `SCALE = 0.001` — the STL is in **mm**; multiply by 0.001 to get **metres**.
  - `ENCODE = {'LR': 0, 'AP': 1}` — maps a design name to which coordinate column
    is the encoding axis (`LR` → x = column 0).
  - `TARGET_SCALE = 1e-3` — an overall multiplier on the target ramp. It does
    **not** move any geometry; it only rescales ψ so the reported per-turn current
    lands in a sane range (a few amps). Efficiency (mT/m/A) is invariant to it.
  - `SUBDIVIDE = 2` — how many times each STL triangle is split 4-ways.
  - `QUAD_SUBDIV = 2` — quadrature resolution for the Biot–Savart integral.
  - `WIRE_PITCH = 3.0e-3` — minimum center-to-center wire spacing (3 mm), used to cap the turn count, lower it for more windings but be mindful of wire spacing
  - `LAMBDA_REG = 1e-11` — Tikhonov regularization weight, raise for smoother wires, but be aware that it will reduce the linearity (R^2)
  *Later in the code there is a section for layering, currently it is set for my 2 layer system with the LR and AP gradients, but you will need to change this for your former thickness/shape
2. Navigate to the python_codes folder in your terminal, then run design_gradient.py
3. In Onshape, import the layered/smoothed files (nested loops) with PathfromPoints Continuous featurescript, or each lobe (continuous wires) with the PathfromPoints featurescript. Don;t mix these up, the regular PathfromPoints script will not be able to import nested loops

