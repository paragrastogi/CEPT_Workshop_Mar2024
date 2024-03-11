import pickle

path_models = './'
with open(path_models + 'models.pickle', 'rb') as file:
    models = pickle.load(file)

model_new = models.loc[:,'model']

print(models)