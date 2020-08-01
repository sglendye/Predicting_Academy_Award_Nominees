# Creating a random forest
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

# Performing a random forest grid search
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def forest(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from pprint import pprint
    import pandas as pd
    import numpy as np
    
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    pprint(rf.get_params())

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



    # Fitting the model... this may take awhile with 16000 columns
    x_train, x_test, y_train, y_test = train_test_split(OHE, OHE['nomination'], test_size = 0.7, random_state = 42)
    rf.fit(x_train, y_train);

    # Predicting with the model
    predictions = rf.predict(x_test)


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


    # Cool. So it looks like we've just done a good job at predicting which movies *are not* likely to get a nomination.
    # While this is still a great find, an enormous type II error on positive identifications isn't exactly what I had in mind for trying to "predict nominees"
    # Perhaps we're overfitting? I'll presume that the creators of random forest didn't set the default hyperparamters for people working with 20,000 columns of sparse data


    # Maybe our trees are too large. Let's see what happens when we prune them a little....


