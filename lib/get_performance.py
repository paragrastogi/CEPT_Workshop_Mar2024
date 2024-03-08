

def calc_performance(model, X):

    return model[0]*X.iloc[:,0] + model[1]*X.iloc[:,1] + model[2]