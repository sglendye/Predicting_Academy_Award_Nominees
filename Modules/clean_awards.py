####### NEW DATASET ##########
# https://www.kaggle.com/unanimad/the-oscar-award?select=the_oscar_award.csv

def write_awards(dir_path):
    import pandas as pd
    import re

    awards = pd.read_csv(dir_path+"\\"+'data'+"\\"+'academy_awards.csv')

    # The title for the best picture award has changed several times over the years!
    awards = awards[(awards['category'] == 'BEST PICTURE')|(awards['category'] == 'OUSTANDING PRODUCTION')|(awards['category'] == 'BEST MOTION PICTURE')]

    awards = awards[['film', 'winner']]

    def normalize(x):
        x = str(x)
        x = x.lower()
        x = re.sub(' ','', x)
        x = re.sub('[^A-Za-z0-9]+', '', x)
        return x

    awards['match'] = awards['film'].apply(lambda x: normalize(x))

    return awards
