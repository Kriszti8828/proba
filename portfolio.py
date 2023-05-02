import numpy as np

# 1. covar mátrix számítása
#print(np.zeros((2,2)))
from matplotlib import pyplot as plt
from seaborn import scatterplot


def calc_covar(szoras_a,szoras_b, corr):
    a = np.zeros((2, 2))
    a[0, 0] = szoras_a*szoras_a
    a[0,1] = corr*szoras_a*szoras_b
    a[1,0] = a[0,1]
    a[1,1] = szoras_b*szoras_b
    #np.array(szoras_a * szoras_b * corr)
    return a
#print(calc_covar(0.2,0.4,0.1))

#chatgpt
def covariance_matrix(sigma_a, sigma_b, corr_ab):
    cov_ab = sigma_a * sigma_b * corr_ab
    cov_ba = cov_ab

    cov_matrix = np.array([[sigma_a ** 2, cov_ab], [cov_ba, sigma_b ** 2]])
    return cov_matrix
#print(covariance_matrix(0.2,0.4,0.1))

# 2. hozamok generálsása
def generate_returns(mu, cov, n):
    returns = np.random.multivariate_normal(mu, cov, n)
    return returns
proba = generate_returns([0.2,0.1],covariance_matrix(0.2,0.4,-0.9),10000)

#ell:
#print(proba.shape)
#print(proba.mean(axis=0))
#print(np.cov(proba.transpose()))

plt.scatter(proba[:,0],proba[:,1])
#plt.show()

# 3. átkonvertálni effektiv hozamokra
import numpy as np

def calculate_effective_return(mu, cov, n):
    effective_return = np.exp(generate_returns(mu, cov, n)) - 1
    return effective_return
print(calculate_effective_return([0.2,0.1],covariance_matrix(0.2,0.4,-0.9),10000))
# pf hozam számítása



# 4. pfhozam=sulyozott eff hozamok összege
# pf értékénem számtása
