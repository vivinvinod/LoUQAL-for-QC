import numpy as np
import qml
from tqdm import tqdm

def generate_CM(n=1000):
    compounds = []
    missed = []
    for i in tqdm(range(n),desc='loading compounds...'):
        #modify as required
        comps = qml.Compound(xyz=f"xyzfiles/filename_{str(i)}.xyz")
        compounds.append(comps)
    nats = [m.natoms for m in compounds]
    nlarge = np.max(nats)
    reps = []
    for m in tqdm(compounds,desc='generate CM'):
        m.generate_coulomb_matrix(size=nlarge,sorting='unsorted ')#'row-norm' results in row-norm sorted CMs
        reps.append(m.representation)
    reps = np.asarray(reps)
    np.save(f'UnsortedCM.npy',reps)

generate_CM()