# Creating a random forest
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# Performing a random forest grid search
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def tuned_forest(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from pprint import pprint
    import pandas as pd
    import numpy as np
    
    #pprint(rf.get_params())

    # Scikit Learn doesn't like null values. I'm filling them with something easy and predictable to nix when we run an OHE
    data = data.fillna(-999)
    test = data.copy()
    del test['title']
    #test = data[['movie_id', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor 4', 'Casting', 'Director', 'Producer', 'Costume Design', 'nomination']]
    OHE = pd.get_dummies(data)

    # Dropping our null columns
    drop_list = OHE.loc[:, OHE.columns.str.endswith('-999')].columns
    OHE = OHE[OHE.columns.difference(drop_list)]

    # Selecting our features. It's pretty much all of the columns except for our one identifier and one predictor
    x = OHE.copy()
    x = x.set_index('movie_id')
    del x['nomination']





    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model


    x_train, x_test, y_train, y_test = train_test_split(OHE, OHE['nomination'], test_size = 0.75, random_state = 42)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(x_train, y_train)



    # Predicting with the model
    predictions = rf_random.predict(x_test)


    # Accuracy!
    print("Accuracy:",metrics.accuracy_score(y_test, predictions))

    # Alright folks, pack it up and call it a day. Our work is done here. That was easy.
    # We should probably check *which* predictions were accurately made though

    # Comparing to see which movies we nailed and missed
    x_test['prediction'] = predictions
    x_test = x_test.reset_index()
    x_test = x_test[['index', 'prediction']]
    results = data.join(x_test, how='left', lsuffix='movie_id', rsuffix='index')
    results = results.dropna(subset=['prediction'])
    results['total_correct'] = np.where((results['nomination'] == results['prediction']), 1, 0)
    results['correct_nominees'] = np.where((results['nomination'] == 1)&(results['nomination'] == results['prediction']), 1, 0)
    results['correct_non_nominees'] = np.where((results['nomination'] == 0)&(results['nomination'] == results['prediction']), 1, 0)


    print(str(sum(results['total_correct'])) + ' total correct predictions out of '+str(len(results))+ ' total attempts')
    print(str(sum(results['correct_nominees'])) + ' total correct nominee predictions out of '+str(sum(results['nomination']))+ ' total attempts')
    print(str(sum(results['correct_non_nominees'])) + ' total correct non-nominee predictions out of '+str(len(results)-sum(results['nomination']))+ ' total attempts')


    # Looks like we're still overfitting again, even after grid searching this one. 

    best_grid = rf_random.best_estimator_
    return best_grid


