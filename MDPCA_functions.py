#!/usr/bin/env python
# coding: utf-8

# In[1]:


def create_extended_data_matrix(Z, l):
    rows, cols = Z.shape
    Y = Z.copy()
    
    for i in range(1, l + 1):
        shifted = Z.shift(periods=i, axis=1)
        shifted.iloc[:, :i] = np.nan  # Fill the new positions with NaN
        Y = pd.concat([Y, shifted], ignore_index=True)
    
    return Y


# In[2]:


def moving_cross_covariance_matrix(Y, w):
    CC=[]
    for i in range(w, Y.shape[1]-w):
        y_i = Y.iloc[:, i-w:i+w].mean(axis=1)
        dot_sum = np.zeros((len(y_i), len(y_i)))
        CC_i=np.zeros((len(y_i), len(y_i)))
        for t in range(i-w, i+w):
            difference = Y.iloc[:, t] - y_i
            dot_product = np.outer(difference, difference)
            dot_sum += dot_product
        CC_i=dot_sum / (2*w + 1)
        CC.append(CC_i)

    num_rows, num_cols = CC[0].shape
    sum_matrix = np.zeros((num_rows, num_cols))    
    for matrix in CC:
        sum_matrix += matrix
    MCC = sum_matrix / (Y.shape[1]-2*w)
    return MCC


# In[3]:


def number_lag_to_include(data, max_lags, w, soglia=0.05):
    l = 0  
    r_new_list = []
    r=[]
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_data_transposed = scaler.fit_transform(data.T)
    scaled_data=scaled_data_transposed.T
    df_scaled_data = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
    
    while l < max_lags:   
        

        # Build the extended data vector yt by including l lagged series
        lagged_data = create_extended_data_matrix(df_scaled_data, l)
        lagged_data = lagged_data.dropna(axis=1) 
        
        lagged_data.columns = [f'{i}' for i in range(len(lagged_data.columns))]
        
        

        # Apply MDPCA to yt and obtain all MDPCs
        MCC=moving_cross_covariance_matrix(lagged_data, w)
        eigenvalues, eigenvectors = np.linalg.eig(MCC)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[sorted_indices] 
        sum_eigenvalues = np.sum(sorted_eigenvalues)
        
        ratios=np.zeros(len(sorted_eigenvalues))        
        for i in range(len(sorted_eigenvalues)):
            ratios[i] = sorted_eigenvalues[i] / sum_eigenvalues
         
    
        plt.plot(np.arange(1,len(ratios)+1),ratios)
        plt.hlines(soglia,1,len(ratios)+1,color='red')
        plt.yscale('log')
        plt.show()
        
        # Set j = m(l + 1) and r(l) = 0 where r is the number of relations
        j = lagged_data.shape[0]
        r.append(0)

        
        i = len(sorted_eigenvalues)-1
        while (ratios[i] <= soglia) and (i>0):
            r[l] += 1 
            print(r[l], l)
            i-=1
            
            
        r_new_list.append(0)
        # Calculate the number of new relations
        if l == 0:
            r_new_list[l] = r[l]
            print(l, j, r_new_list[l], r[l])
        else:
            r_new_list[l] = r[l] - sum((l - k + 1) * r_new_list[k] for k in range(l))
            print(l, j, r_new_list[l], r[l])

        if r_new_list[l] <= 0:
            print(l, j, r_new_list[l], r[l])
            return (l-1)
        else: l += 1


# In[4]:


def calculate_mse(y, u_hat, C, k):
    
    M, N = y.shape
    mse = 0
    for j in range(0, M):
        for t in range(0, N):
            reconstructed_value = sum(u_hat[j, v] * C[v, t] for v in range(k))
            mse += (y.iloc[j, t] - reconstructed_value) ** 2
    
    mse /= (M * N)
    return mse


# In[5]:


def calculate_max_mse(y):
    
    M, N = y.shape
    sum_squared = 0.0
    for i in range(M):
        for j in range(N):
            sum_squared += y.iloc[i, j] ** 2
    max_mse = sum_squared / (M * N)
    return max_mse


# In[6]:


def calculate_rcc(M, eigenvalues, mse_k, max_mse, k):
        
    # Calculate the RCC
    explained_variance = np.sum(eigenvalues[:k]) / np.sum(eigenvalues)
    reduced_mse = (max_mse - mse_k) / max_mse
    penalty_term = 2 * k / M
    
    rcc_k = 2 - explained_variance - reduced_mse + penalty_term
    return rcc_k

