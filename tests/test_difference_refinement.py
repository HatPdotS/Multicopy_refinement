from multicopy_refinement import io
from multicopy_refinement import difference_refinement



mtz1 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.mtz'
mtz2 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.mtz'


mtz1 = io.read_mtz(mtz1)
mtz2 = io.read_mtz(mtz2)

import multicopy_refinement.Model as Model


model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')

model2 = Model.model()
model2.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')

ref = difference_refinement.Difference_refinement(mtz1,mtz2,model,model2)

ref.get_scales()