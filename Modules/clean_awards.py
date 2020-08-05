# Writing the Academy Award Nominees to our database
# Data Set: https://www.kaggle.com/unanimad/the-oscar-award?select=the_oscar_award.csv

def write_awards(dir_path):
    import pandas as pd
    import re

    awards = pd.read_csv(dir_path+"\\"+'data'+"\\"+'academy_awards.csv')

    # Important note to keep in mind, the title of the Best Picture Award has changed several times over the years.
    awards = awards[(awards['category'] == 'BEST PICTURE')|(awards['category'] == 'OUSTANDING PRODUCTION')|(awards['category'] == 'BEST MOTION PICTURE')]

    # These are the only two fields that we need from this data
    awards = awards[['film', 'winner']]

    # Normalizing our film titles to create a more accurate match column with our other data
    def normalize(x):
        x = str(x)
        x = x.lower()
        x = re.sub(' ','', x)
        x = re.sub('[^A-Za-z0-9]+', '', x)
        return x

    awards['match'] = awards['film'].apply(lambda x: normalize(x))

    return awards
