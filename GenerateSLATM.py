import numpy as np
import qml
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm

np.int = int #forces numpy compatiblity with qml


def slatm_glob(n):
    compounds = []
    missed = []
    for i in tqdm(range(n),desc='loading compounds...'):
        #modify as needed
        comps = qml.Compound(xyz=f"xyzfiles/filename_{str(i)}.xyz")
        compounds.append(comps)
    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in tqdm(compounds, desc='get mbtype...')],dtype=object))

    #optionally you can save the mbtypes
    #np.save('mbtypes.npy',allow_pickle=True)
    
    '''
    if at some point you have very many geometries (~100k), then this below code is highly memory intensive.
    In such a case you can simply load the mbtypes that you saved in the previous step and generate SLATM for chunks of the data.
    '''
    
    for i, mol in tqdm(enumerate(compounds),desc='Generate Global SLATM...'):
        mol.generate_slatm(mbtypes,local=False)
    
    X_slat_glob = np.asarray([mol.representation for mol in tqdm(compounds,desc='Saving Reps...')])

np.save('SLATM.npy',X_slat_glob)


slatm_glob(n=100)