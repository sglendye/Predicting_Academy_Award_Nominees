# Reference guide that I used when working with CatBoose
# https://heartbeat.fritz.ai/fast-gradient-boosting-with-catboost-38779b0d5d9a

def cat_boost(data, iters):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import time
    import warnings
    
    # Pandas doesn't like the way that I slice, but it will have to live with it today
    warnings.filterwarnings("ignore")

    # Timing the model
    start_time = time.time()
    
    # CatBoost doesn't like null values either. Also filling them with something easy and predictably OOB here
    data = data.fillna(-999)

    # Catboost also doesn't like dealing with floats, so shifting those over to integer
    x = data.copy()
    x = x.fillna(-999)
    x['movie_id'] = x['movie_id'].astype(int)
    x['budget'] = x['budget'].astype(int)
    x['revenue'] = x['revenue'].astype(int)
    x['runtime'] = x['runtime'].astype(int)
    x = x.set_index('movie_id')

    # Breaking off our object data types. CatBoost needs to know which ones to treat as categorical variables
    categorical_features_indices = np.where(x.dtypes == np.object)[0]
    
    # CatBoost model, the silent tells it to stop printing the results of every iteration
    cat = CatBoostClassifier(iterations = iters, silent = True)

    # Fitting the model
    x_train, x_test, y_train, y_test = train_test_split(x, x['nomination'], test_size = 0.3, random_state = 42)
    cat.fit(x_train, y_train, cat_features=categorical_features_indices);

    # Predicting with the model
    predictions = cat.predict(x_test)


    # Comparing to see which movies we nailed and missed
    x_test['prediction'] = predictions
    x_test = x_test.reset_index()
    x_test = x_test[['movie_id', 'prediction']]
    results = data.join(x_test, how='left', lsuffix='movie_id', rsuffix='movie_id')
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
    total_percent = str(round((total_correct/total_predictions)*100,2))+"%"
    nom_percent = str(round((correct_nominees/total_nominees)*100, 2))+"%"
    non_nom_percent = str(round((correct_non_nominees/total_non_nominees)*100,2))+"%"
   
    # Speed
    cat_speed = ("--- %s seconds ---" % round((time.time() - start_time),2))
    
    # Printing results
    print("------------- CatBoost Results -------------")
    print(str(total_correct)+' total correct predictions out of '+str(total_predictions)+ " total attempts " + "(" + str(total_percent)+")")
    print(str(correct_nominees) + ' total correct nominee predictions out of '+str(total_nominees)+ ' total attempts '+ "(" + str(nom_percent)+")")
    print(str(correct_non_nominees) + ' total correct non-nominee predictions out of '+str(total_non_nominees)+ ' total attempts '+ "(" + str(non_nom_percent)+")")
    print("Time to complete: "+str(cat_speed))
    print("--------------------------------------------")
    
    # Returns a metrics list
    metrics = [total_percent, nom_percent, non_nom_percent, cat_speed]
    return metrics   
    
