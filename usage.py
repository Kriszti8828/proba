import machinelearning as m

coeffs=[100,1,0.2]
X,y = m.generate_polinomial_data(coeffs,fromX=-5,toX=7, n_samples=500, noise=1,random_sate=42,filepath='data_csv')
X_train, X_test, y_train, y_test=m.train_test_split(X,y,test_size=0.2,random_state=42)

name,model,mse_on_test_set,coefficients_on_train_set = m.create_train_and_evaulate_polynomial_model(X_train,y_train,X_test,y_test, degree=10)
print(1)