# Creating a random forest
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# Performing a random forest grid search
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def tuned_forest(data, iters):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import time
    import warnings
    
    # Pandas doesn't like the way that I slice, but it will have to live with it today
    warnings.filterwarnings("ignore")

    # Timing the model
    start_time = time.time()
    
    # "Best Parameters" results from our grid search
    rf = RandomForestClassifier(bootstrap = False, class_weight = None, criterion = 'gini', max_depth = 90, max_features = 'sqrt', max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, n_estimators = iters, n_jobs = None, oob_score = False, verbose = 0, warm_start = False, random_state = 42)

    # Scikit Learn doesn't like null values. I'm filling them with something easy and predictable to nix when we run an OHE
    data = data.fillna(-999)
    OHE = pd.get_dummies(data)

    # Dropping our null columns
    drop_list = OHE.loc[:, OHE.columns.str.endswith('-999')].columns
    OHE = OHE[OHE.columns.difference(drop_list)]
    
    # Fitting the model... this may take awhile with 16000 columns
    x_train, x_test, y_train, y_test = train_test_split(OHE, OHE['nomination'], test_size = 0.3, random_state = 42)
    rf.fit(x_train, y_train);

    # Predicting with the model
    predictions = rf.predict(x_test)

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

    # Calculating how many are correct in different categories, and which percentage were correct
    total_correct = sum(results['total_correct'])
    total_predictions = len(results)
    correct_nominees = sum(results['correct_nominees'])
    total_nominees = sum(results['nomination'])
    correct_non_nominees = sum(results['correct_non_nominees'])
    total_non_nominees = len(results)-sum(results['nomination'])
    total_non_nominees = len(results)-sum(results['nomination'])
    total_percent = str(round((total_correct/total_predictions)*100,2))+"%"
    nom_percent = str(round((correct_nominees/total_nominees)*100, 2))+"%"
    non_nom_percent = str(round((correct_non_nominees/total_non_nominees)*100,2))+"%"

    # Speed
    forest_speed = ("--- %s seconds ---" % round((time.time() - start_time),2))
    
    # Printing off results
    print("------------ Tuned Forest Results -----------")
    print(str(total_correct)+' total correct predictions out of '+str(total_predictions)+ " total attempts " + "(" + str(total_percent)+")")
    print(str(correct_nominees) + ' total correct nominee predictions out of '+str(total_nominees)+ ' total attempts '+ "(" + str(nom_percent)+")")
    print(str(correct_non_nominees) + ' total correct non-nominee predictions out of '+str(total_non_nominees)+ ' total attempts '+ "(" + str(non_nom_percent)+")")
    print("Time to complete: "+str(forest_speed))
    print("--------------------------------------------")
    print(''' 
    
    ''')
    
    # Returns a metrics list
    metrics = [total_percent, nom_percent, non_nom_percent, forest_speed]
    return metrics



