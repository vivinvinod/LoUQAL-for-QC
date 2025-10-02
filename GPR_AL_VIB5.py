import numpy as np
from GPR_model import ExactGPModel as GPR
from GPR_model import train_hypers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import gpytorch
import sys
import pandas as pd

def ensemble_UQ(hyper_path,X_pool,X_train,y_train,n_ens=5):
    predictions = np.zeros((X_pool.shape[0],n_ens),dtype=float)
    for n in range(n_ens):
        #select 85% of training data per ensemble
        X_selected,_,y_selected,_ = train_test_split(X_train,y_train,train_size=0.85,random_state=n)
        
        #initialize new model for each ensemble
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPR(X_selected,y_selected, likelihood=likelihood)
        model_state = torch.load(hyper_path)
        model.load_state_dict(model_state)

        predictions[:,n],_,_,_ = predict_with_GPR(model=model,
                                                  likelihood=likelihood, 
                                                  X_test=X_pool)
    #compute mean predictions
    mean_predictions = np.mean(predictions,axis=1)
    for n in range(n_ens):
        predictions[:,n] = predictions[:,n]-mean_predictions
    #compute uncertainty
    uncertainty = np.sqrt((np.sum(predictions**2,axis=1))/(n_ens-1))
    
    return uncertainty


def ensemble_AL(X_train, y_train, 
                   X_test, y_test,
                   n_initial=50, AL_iters=500, 
                   seed=42, mol:str=None, n_ensemble:int=5):
    #initial split of data
    full_pool_indexes = np.arange(0,X_train.shape[0])
    X_train, X_pool, y_train, y_pool,train_index, pool_index= train_test_split(X_train,y_train,full_pool_indexes,random_state=seed,train_size=n_initial/X_train.shape[0])
    print(X_train.shape,X_pool.shape)
    maes = []
    #selected_indices = []
    highest_diff = []
    training_size = []
    selected_ind_list = []
    
    hyper_path = f'ModelData/{mol}_params.pth'

    for i in tqdm(range(AL_iters),desc='Ensemble based AL loop'):
        training_size.append(X_train.shape[0])
        #train model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPR(X_train, y_train, likelihood=likelihood)
        model_state = torch.load(hyper_path)
        model.load_state_dict(model_state)
        
        #predict over test
        preds_test,_,_,_ = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_test)
        #compute mae for test set
        #temp_mae = np.mean(np.abs(preds_test-y_test))
        temp_mae = np.mean(np.abs((preds_test-y_test)))
        maes.append(temp_mae)

        #ensemble method for UQ over pool
        # we will randomly select 85% of the training data per ensemble
        uncertainty = ensemble_UQ(hyper_path = hyper_path,
                                  X_pool = X_pool, X_train = X_train, 
                                  y_train = y_train,
                                  n_ens = 5)
        
        selected_index = np.argmax(uncertainty)
        selected_ind_list.append(pool_index[selected_index])
        #record stuff
        #selected_indices.append(selected_index)
        highest_diff.append(uncertainty[selected_index])
        
        ####extend training data
        #extend X_train with selected point:
        X_selected = X_pool[selected_index]
        X_train = np.vstack((X_train,X_selected))
        #extend y_train
        y_selected = np.copy(y_pool[selected_index])
        #print(y_pool[selected_index],y_selected)
        y_train = np.concatenate((y_train,[y_selected]))
        ####reduce pool
        X_pool = np.delete(X_pool,selected_index,axis=0)
        y_pool = np.delete(y_pool,selected_index,axis=0)
        ##remove index from pool
        pool_index = np.delete(pool_index,selected_index,axis=0)
    selected_ind_list = np.asarray(selected_ind_list)
    np.save(f'PlotData/ensemble_{mol}_selected_indexes.npy',selected_ind_list)

    maes = np.asarray(maes)
    highest_diff = np.asarray(highest_diff)
    training_size = np.asarray(training_size)
    return maes, highest_diff, training_size
    

def predict_with_GPR(model,likelihood,X_test):
    model.eval()
    likelihood.eval()
    if isinstance(X_test, np.ndarray):
        test_x_tensor = torch.from_numpy(X_test).float()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x_tensor))
        preds = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()
    return preds,lower.numpy(),upper.numpy(),observed_pred.variance.numpy()

def single_fidelity_AL(X_train, y_train, 
                       X_test, y_test,
                       n_initial=50, AL_iters=500, 
                       seed=42, mol:str=None):
    #initial split of data
    full_pool_indexes = np.arange(0,X_train.shape[0])
    X_train, X_pool, y_train, y_pool, train_index, pool_index = train_test_split(X_train, y_train, full_pool_indexes, random_state=seed, train_size=n_initial/X_train.shape[0])
    print(X_train.shape,X_pool.shape)
    np.save(f'PlotData/{mol}_initial_selected_indexes.npy',train_index)
    maes = []
    highest_diff = []
    training_size = []
    selected_ind_list = []

    #run hyper-opt with initial samples
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPR(X_train, y_train, likelihood=likelihood)
    model.train()
    likelihood.train()
    hyper_path = f'ModelData/{mol}_params.pth'
    losses = train_hypers(model, likelihood, lr=0.05, maxiter=1000, 
                          save_path=hyper_path, tol=1e-5)
    

    for i in tqdm(range(AL_iters),desc='SFAL loop'):
        training_size.append(X_train.shape[0])
        #train model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPR(X_train, y_train, likelihood=likelihood)
        model_state = torch.load(hyper_path)
        model.load_state_dict(model_state)
        
        #predict over test
        preds_test,_,_,_ = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_test)
        #compute mae for test set
        temp_mae = np.mean(np.abs(preds_test-y_test))
        #temp_mae = np.mean(np.abs((preds_test-y_test)/y_test))
        maes.append(temp_mae)
        #predict over pool
        preds_pool,_,_,_ = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_pool)
        #find abs difference
        abs_diff_pool = np.abs(y_pool-preds_pool)
        #find location of max
        selected_index = np.argmax(abs_diff_pool)
        selected_ind_list.append(pool_index[selected_index])
        #record stuff
        #selected_indices.append(selected_index)
        highest_diff.append(abs_diff_pool[selected_index])

        ####extend training data
        #extend X_train with selected point:
        X_selected = X_pool[selected_index]
        X_train = np.vstack((X_train,X_selected))
        #extend y_train
        y_selected = np.copy(y_pool[selected_index])
        #print(y_pool[selected_index],y_selected)
        y_train = np.concatenate((y_train,[y_selected]))

        ####reduce pool
        X_pool = np.delete(X_pool,selected_index,axis=0)
        y_pool = np.delete(y_pool,selected_index,axis=0)
        ##remove index from pool
        pool_index = np.delete(pool_index,selected_index,axis=0)
    selected_ind_list = np.asarray(selected_ind_list)
    np.save(f'PlotData/SF_{mol}_selected_indexes.npy',selected_ind_list)
    maes = np.asarray(maes)
    highest_diff = np.asarray(highest_diff)
    training_size = np.asarray(training_size)
    return maes, highest_diff, training_size,losses

def variance_AL(X_train, y_train, 
                       X_test, y_test,
                       n_initial=50, AL_iters=500, 
                       seed=42, mol:str=None):
    #initial split of data
    full_pool_indexes = np.arange(0,X_train.shape[0])
    X_train, X_pool, y_train, y_pool,train_index,pool_index = train_test_split(X_train,y_train,full_pool_indexes,random_state=seed,train_size=n_initial/X_train.shape[0])
    print(X_train.shape,X_pool.shape)
    maes = []
    rel_maes = []
    selected_ind_list = []
    #selected_indices = []
    highest_diff = []
    training_size = []
    
    hyper_path = f'ModelData/{mol}_params.pth'

    for i in tqdm(range(AL_iters),desc='Variance based AL loop'):
        training_size.append(X_train.shape[0])
        #train model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPR(X_train, y_train, likelihood=likelihood)
        model_state = torch.load(hyper_path)
        model.load_state_dict(model_state)
        
        #predict over test
        preds_test,_,_,_ = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_test)
        #compute mae for test set
        temp_mae = np.mean(np.abs(preds_test-y_test))
        maes.append(temp_mae)
        #relative Mae
        temp_rmae = np.mean(np.abs((preds_test-y_test)/y_test))
        rel_maes.append(temp_rmae)
        #predict over pool
        _,_,_,var_pool = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_pool)
        #find location of max
        selected_index = np.argmax(var_pool)
        selected_ind_list.append(pool_index[selected_index])
        #record stuff
        #selected_indices.append(selected_index)
        highest_diff.append(var_pool[selected_index])

        ####extend training data
        #extend X_train with selected point:
        X_selected = X_pool[selected_index]
        X_train = np.vstack((X_train,X_selected))
        #extend y_train
        y_selected = np.copy(y_pool[selected_index])
        #print(y_pool[selected_index],y_selected)
        y_train = np.concatenate((y_train,[y_selected]))

        ####reduce pool
        X_pool = np.delete(X_pool,selected_index,axis=0)
        y_pool = np.delete(y_pool,selected_index,axis=0)
        ##remove index from pool
        pool_index = np.delete(pool_index,selected_index,axis=0)
    selected_ind_list = np.asarray(selected_ind_list)
    np.save(f'PlotData/Var_{mol}_selected_indexes.npy',selected_ind_list)

    maes = np.asarray(maes)
    highest_diff = np.asarray(highest_diff)
    training_size = np.asarray(training_size)
    rel_maes = np.asarray(rel_maes)
    return maes, highest_diff, training_size, rel_maes

def random_AL(X_train, y_train, 
                       X_test, y_test,
                       n_initial=50, AL_iters=500, 
                       seed=42, mol:str=None):
    #initial split of data
    X_train, X_pool, y_train, y_pool = train_test_split(X_train,y_train,random_state=seed,train_size=n_initial/X_train.shape[0])
    print(X_train.shape,X_pool.shape)
    maes = []
    rel_maes = []
    #selected_indices = []
    highest_diff = []
    training_size = []
    np.random.seed(seed)
    
    hyper_path = f'ModelData/{mol}_params.pth'

    for i in tqdm(range(AL_iters),desc='Random AL loop'):
        training_size.append(X_train.shape[0])
        #train model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPR(X_train, y_train, likelihood=likelihood)
        model_state = torch.load(hyper_path)
        model.load_state_dict(model_state)
        
        #predict over test
        preds_test,_,_,_ = predict_with_GPR(model=model,likelihood=likelihood, X_test=X_test)
        #compute mae for test set
        temp_mae = np.mean(np.abs(preds_test-y_test))
        maes.append(temp_mae)
        #relative Mae
        temp_rmae = np.mean(np.abs((preds_test-y_test)/y_test))
        rel_maes.append(temp_rmae)
        #randomly pick a sample
        pool_indices = np.arange(X_pool.shape[0])
        selected_index = np.random.choice(pool_indices,1)[0]
        
        ####extend training data
        #extend X_train with selected point:
        X_selected = X_pool[selected_index]
        X_train = np.vstack((X_train,X_selected))
        #extend y_train
        y_selected = np.copy(y_pool[selected_index])
        #print(y_pool[selected_index],y_selected)
        y_train = np.concatenate((y_train,[y_selected]))

        ####reduce pool
        X_pool = np.delete(X_pool,selected_index,axis=0)
        y_pool = np.delete(y_pool,selected_index,axis=0)

    maes = np.asarray(maes)
    training_size = np.asarray(training_size)
    rel_maes = np.asarray(rel_maes)
    return maes, training_size, rel_maes

def multi_fidelity_AL(X_train, y_train_low, y_train_high, 
                      X_test, y_test,
                      n_initial=50, AL_iters=500, seed=42, mol=None):
    #initial split of data
    full_pool_indexes = np.arange(0,X_train.shape[0])
    X_train, X_pool, y_train_low, y_pool_low, y_train_high, y_pool_high, train_index, pool_index = train_test_split(X_train, y_train_low, 
                                                                                                                    y_train_high, full_pool_indexes, 
                                                                                           random_state=seed,
                                                                                           train_size=n_initial/X_train.shape[0])
    print(X_train.shape,X_pool.shape)
    np.save(f'PlotData/{mol}_initial_selected_indexes.npy',train_index)
    #model_save_path = f'ModelData/{mol}_hypers.pth'
    maes = []
    rel_maes = []
    selected_ind_list = []
    #overall index
    
    highest_diff = []
    training_size = []
    hyper_path = f'ModelData/{mol}_params.pth'

    for i in tqdm(range(AL_iters),desc='MFAL loop'):
        training_size.append(X_train.shape[0])
        #train high and low fidelity model
        likelihood_high = gpytorch.likelihoods.GaussianLikelihood()
        model_high = GPR(X_train, y_train_high, likelihood=likelihood_high)
        model_state = torch.load(hyper_path)
        model_high.load_state_dict(model_state)
        
        likelihood_low = gpytorch.likelihoods.GaussianLikelihood()
        model_low = GPR(X_train, y_train_low, likelihood=likelihood_low)
        model_state = torch.load(hyper_path)
        model_low.load_state_dict(model_state)
        ##predict over test
        preds_test,_,_,_ = predict_with_GPR(model=model_high,likelihood=likelihood_high,X_test=X_test)
        #compute mae for test set
        temp_mae = np.mean(np.abs(preds_test-y_test))
        maes.append(temp_mae)
        #relative Mae
        temp_rmae = np.mean(np.abs((preds_test-y_test)/y_test))
        rel_maes.append(temp_rmae)
        #predict low fidelity over pool
        preds_pool_low,_,_,_ = predict_with_GPR(model=model_low,likelihood=likelihood_low,X_test=X_pool)
        #find abs difference
        abs_diff_pool_low = np.abs(y_pool_low-preds_pool_low)
        #find location of max
        selected_index = np.argmax(abs_diff_pool_low)
        selected_ind_list.append(pool_index[selected_index])
        #record stuff
        #selected_indices.append(selected_index)
        highest_diff.append(abs_diff_pool_low[selected_index])

        ####extend training data
        #extend X_train with selected point:
        X_selected = X_pool[selected_index]
        X_train = np.vstack((X_train,X_selected))
        
        #extend y_train
        y_selected_low = np.copy(y_pool_low[selected_index])
        y_selected_high = np.copy(y_pool_high[selected_index])
        y_train_low = np.concatenate((y_train_low,[y_selected_low]))
        y_train_high = np.concatenate((y_train_high,[y_selected_high]))
        
        ####reduce pool
        X_pool = np.delete(X_pool,selected_index,axis=0)
        y_pool_low = np.delete(y_pool_low,selected_index,axis=0)
        y_pool_high = np.delete(y_pool_high,selected_index,axis=0)
        ##remove index from pool
        pool_index = np.delete(pool_index,selected_index,axis=0)
    selected_ind_list = np.asarray(selected_ind_list)
    np.save(f'PlotData/MF_{mol}_selected_indexes.npy',selected_ind_list)
    maes = np.asarray(maes)
    highest_diff = np.asarray(highest_diff)
    training_size = np.asarray(training_size)
    rel_maes = np.asarray(rel_maes)
    
    return maes,highest_diff,training_size, rel_maes

def SF_main(mol='SiH4'):
    X = np.load(f'VIB5/{mol}_CM.npy')
    y_high = np.loadtxt(f'VIB5/{mol}_CCSD-T.dat') #ccsd(t)ccpvdz
    

    X,y_high = shuffle(X,y_high,random_state=42)
    X = np.copy(X[:15000])
    y_high = np.copy(y_high[:15000])
    y_high = y_high - np.mean(y_high)
    

    X_train, X_test, y_train_high, y_test_high = train_test_split(X,y_high,random_state=42,train_size=0.9) #1500 test


    
    maes,highest_diff,training_size,losses = single_fidelity_AL(X_train=X_train, y_train=y_train_high, 
                                                                X_test=X_test, y_test=y_test_high, 
                                                                n_initial=100, AL_iters=2000, seed=42, 
                                                                mol=mol)
    np.save(f'ModelData/sf_VIB5_{mol}_maes.npy',maes)
    #np.save(f'ModelData/sf_VIB5_{mol}_relative_maes.npy',rel_mae)
    np.save(f'ModelData/sf_VIB5_{mol}_highest_diff.npy',highest_diff)
    np.save(f'ModelData/sf_VIB5_{mol}_training_size.npy',training_size)
    np.save(f'ModelData/VIB5_{mol}_training_loss.npy',training_size)

def ensemble_main(mol='CH3Cl'):
    X = np.load(f'VIB5/{mol}_CM.npy')
    y_high = np.loadtxt(f'VIB5/{mol}_CCSD-T.dat') #ccsd(t)ccpvdz
    
    X,y_high = shuffle(X,y_high,random_state=42)
    X = np.copy(X[:15000])
    y_high = np.copy(y_high[:15000])
    y_high = y_high - np.mean(y_high)
    

    X_train, X_test, y_train_high, y_test_high = train_test_split(X,y_high,random_state=42,train_size=0.9) #1500 test
    
    maes,highest_diff,training_size = ensemble_AL(X_train=X_train, y_train=y_train_high, 
                                                    X_test=X_test, y_test=y_test_high, 
                                                    n_initial=100, AL_iters=2000, 
                                                  seed=42, mol=mol, n_ensemble=5)
    np.save(f'ModelData/ensemble_VIB5_{mol}_maes.npy',maes)
    #np.save(f'ModelData/sf_VIB5_{mol}_relative_maes.npy',rel_mae)
    np.save(f'ModelData/ensemble_VIB5_{mol}_highest_diff.npy',highest_diff)
    np.save(f'ModelData/ensemble_VIB5_{mol}_training_size.npy',training_size)
    #np.save(f'ModelData/VIB5_{mol}_training_loss.npy',training_size)

def var_main(mol='CH3Cl'):
    X = np.load(f'VIB5/{mol}_CM.npy')
    y_high = np.loadtxt(f'VIB5/{mol}_CCSD-T.dat') #ccsd(t)
    X,y_high = shuffle(X,y_high,random_state=42)
    X = np.copy(X[:15000])
    y_high = np.copy(y_high[:15000])
    y_high = y_high - np.mean(y_high)

    X_train, X_test, y_train_high, y_test_high = train_test_split(X,y_high,random_state=42,train_size=0.9) #1500 test


    
    maes,highest_diff,training_size,rel_mae = variance_AL(X_train=X_train, y_train=y_train_high, 
                                                                X_test=X_test, y_test=y_test_high, 
                                                                n_initial=100, AL_iters=2000, seed=42, 
                                                                mol=mol)
    np.save(f'ModelData/var_VIB5_{mol}_maes.npy',maes)
    np.save(f'ModelData/var_VIB5_{mol}_maes.npy',rel_mae)
    np.save(f'ModelData/var_VIB5_{mol}_highest_diff.npy',highest_diff)
    np.save(f'ModelData/var_VIB5_{mol}_training_size.npy',training_size)

def random_main(mol='CH3Cl'):
    X = np.load(f'VIB5/{mol}_CM.npy')
    y_high = np.loadtxt(f'VIB5/{mol}_CCSD-T.dat')
    
    X,y_high = shuffle(X,y_high,random_state=42)
    X = np.copy(X[:15000])
    y_high = np.copy(y_high[:15000])
    y_high = y_high - np.mean(y_high)

    X_train, X_test, y_train_high, y_test_high = train_test_split(X,y_high,random_state=42,train_size=0.9) #1500 test
    maes,training_size,rel_mae = random_AL(X_train=X_train, y_train=y_train_high, 
                                                                X_test=X_test, y_test=y_test_high, 
                                                                n_initial=100, AL_iters=2000, seed=42, 
                                                                mol=mol)
    np.save(f'ModelData/random_VIB5_{mol}_maes.npy',maes)
    np.save(f'ModelData/random_VIB5_{mol}_maes.npy',rel_mae)
    np.save(f'ModelData/random_VIB5_{mol}_training_size.npy',training_size)
    
def MF_main(mol='CH3Cl',method='ccpvdz'):
    X = np.load(f'VIB5/{mol}_CM.npy')
    y_high = np.loadtxt(f'VIB5/{mol}_CCSD-T.dat') #ccsd(t)
    
    y_low = np.loadtxt(f'VIB5/{mol}_{method}.dat')
    
    X,y_high,y_low = shuffle(X,y_high,y_low,random_state=42)
    X = np.copy(X[:15000])
    y_high = np.copy(y_high[:15000])
    y_low = np.copy(y_low[:15000])
    
    y_high = y_high - np.mean(y_high)
    y_low = y_low - np.mean(y_low)

    X_train, X_test, y_train_low, y_test_low, y_train_high, y_test_high = train_test_split(X,y_low,y_high,random_state=42,train_size=0.9) #1500 test


    
    maes,highest_diff,training_size,rel_mae = multi_fidelity_AL(X_train=X_train,y_train_low=y_train_low,
                                                        y_train_high=y_train_high, 
                                                        X_test=X_test, y_test=y_test_high, 
                                                        n_initial=100, AL_iters=2000, seed=42,
                                                        mol=mol)
    np.save(f'ModelData/mf_VIB5_{mol}_{method}_maes.npy',maes)
    np.save(f'ModelData/mf_VIB5_{mol}_{method}_maes.npy',rel_mae)
    np.save(f'ModelData/mf_VIB5_{mol}_{method}_highest_diff.npy',highest_diff)
    np.save(f'ModelData/mf_VIB5_{mol}_{method}_training_size.npy',training_size)
    
if __name__=='__main__':
    methods = ['HF-TZ']#['MP2','HF-QZ','HF-TZ']
    SF_main(mol='CH3Cl')
    var_main(mol='CH3Cl')
    random_main(mol='SiH4')
    ensemble_main(mol='CH3Cl')
    for m in methods:
        MF_main(mol='CH3Cl',method=m)


