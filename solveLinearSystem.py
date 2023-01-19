import numpy as np
import time
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse


def get_b(num,zeroB):
    if zeroB:
        b = np.zeros((num,))
    else:
        #b = np.random.uniform(0,1,num)
        b = np.ones(num)
    return b


def solve_system_with_info(A, b, method, M = None):
    if method == 'gmres':
        st = time.time()
        H = sparse.csc_matrix(A)
        #x, info = linalg.gmres(A,b,restart=200)
        x, info = linalg.gmres(H, b, x0=np.zeros(len(b)), M=M, restart=len(b))
        en = time.time()
        if info == 0:
            print('Solution converged. Solve time is: {:.5f} s'.format(en-st))
        elif info > 0:
            print('Solution did not converge. Time passed is: {:.5f} s. Number of iterations: {:d}'
                  .format(en-st, info))
        else:
            print('Error: illegal input or breakdown. Time passed is: {:.5f} s'.format(en-st))
    elif method == 'bicgstab':
        st = time.time()
        H = sparse.csc_matrix(A)
        x, info = linalg.bicgstab(H, b, M=M, maxiter=10000)
        en = time.time()
        if info == 0:
            print('Solution converged. Solve time is: {:.5f} s'.format(en-st))
        elif info > 0:
            print('Solution did not converge. Time passed is: {:.5f} s. Number of iterations: {:d}'
                  .format(en-st, info))
        else:
            print('Error: illegal input or breakdown with exit code : {:d}. Time passed is: {:.5f} s'
                  .format(info, en-st))
    else:
        print('Error: solve method not recognized')