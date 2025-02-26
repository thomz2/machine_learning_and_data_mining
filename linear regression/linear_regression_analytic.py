import numpy as np

def analytic_linear_regression(X, y, debug=False):
    
    # np.c_ é usado para concatenar arrays nas colunas
    # aqui estamos usando para adicionar a coluna de 1s na matriz X
    # obs.: podemos usar o np.r_ para fazer a mesma coisa com linhas
    X_b = np.c_[ 
        np.ones(
            # shape retorna uma tupla que representa as dimensões de X
            # nesse caso retorna (n_samples, n_features)
            X.shape[0]
        ),
        X
    ]

    
    XTX = X_b.T @ X_b

    XTX_inversa = np.linalg.inv(XTX)

    XTX_inversa_por_T = XTX_inversa @ X_b.T
    
    w = XTX_inversa_por_T @ y

    if debug:
        print("\n---\n")
        
        print("X^T por X:")
        print(XTX)
        print("\n---\n")
        
        print("(X^T por X) inversa:")
        print(XTX_inversa)
        print("\n---\n")
        
        print("X^TX inversa por X^T:")
        print(XTX_inversa_por_T)
        print("\n---\n")

        print("Agora por Y (resultado)")
        print(w, "\n\n")


    return w

if __name__ == "__main__":

    X = np.array([
        [1, 0], # X1
        [5, 6], # X2
    ])
    y = np.array([10, 12])

    w = analytic_linear_regression(X, y, True)

    # Fazendo previsões
    def predict_values(X, w):
        # Adiciona coluna de 1s
        X_b = np.c_[
            np.ones(
                X.shape[0] # como o número de linhas aqui = tamanho da coluna...
            ), 
            X
        ]  
        return X_b @ w
    
    def predict_value(x_i, w):
        x_i = np.append(1, x_i)
        return w.T @ x_i

    y_pred  = predict_values(X, w)
    print("Previsões:", y_pred)

    yi_pred = predict_value(X[0], w)
    print("Previsão de x_1:", yi_pred)

    print("\n---\n")