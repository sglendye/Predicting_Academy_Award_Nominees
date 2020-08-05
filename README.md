# Predicting The Likelihood Of An Academy Award Nomination
<br/>

# Introduction:

Who doesn't love Oscar Season? Every media outlet prepares itself in advance for "likely" candidates to receive a nomination or this year's award and we all sit around take guesses on our likely favorites. But how do they (we) make such predictions? Is it a film's success in the box office? The ones with enormous budgets for special effects? Films directed or produced by a big name? (side note: if you ever see a film directed by William Wyler or Steven Spielberg, with 13 and 11 Best Picture nominations, respectively, it's probably a good guess that the film was/will be nominated). Or even something more mundane such as how long the movie runs, as we all know that us artsy types love our long films - looking at you Lawrence of Arabia. Predicting which films the Academy will decide on is notoriously difficult, and usually boils down to some combination of all of these factors that gives us a feeling when we see a film that "this is the one". As someone with a film director for a partner, and who is quite the film buff as well, trying to replicate this predictive process on a computer seemed like a fun project for merging my passion with my computer skills.

Using the data from Kaggle's TMDB data set (https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_credits.csv) and a list of academy award nominations (also from Kaggle: https://www.kaggle.com/unanimad/the-oscar-award?select=the_oscar_award.csv), I want to initiate the first steps of this predictive process by creating a model that can be fed some data pertaining to a film and classify if it will be nominated for Best Picture or not. Because a film's chance at actually *winning* is so dependent on the quality of other films released in its year; i.e. shockingly competitive fields like 1993, 1994, or 2015 are very different than 2021 will be after Covid closed down all theaters, we will temper our expectations and only look for nominations in the first iteration of this project.

To continue on with the excitement of this analysis, proceed to the above "Analysis.ipnb" notebook provided. Otherwise, continue below to find everything included in this repository, described in painstaking detail.

<br/>

# Contents Of This Repository

<br/>

## Data:
The data included is a number of csv files pertaining to film. The files are...
- acaddemy_awards: A list of nominees and winners for each award ceremony
- tmdb_5000_credits (first half): Credits for 5000 different films. The tmdb_credits file was too large to upload to kaggle at once, so it was broken into two halves, each containing credits for several thousand films
- tmdb_credits (second half)
- tmdb_5000_movies: A list of the above mentioned films and data collected for them (budget, revenue, runtime, etc.)
- inflation_rates: A small list of CPI by year from the U.S. Bureau of Labor Statistics. This is used to normalize our monetary data, since the original data is not adjusted for inflation

The credits and move data were pulled from the TMDB_5000 data set on Kaggle, and the academy award nominess were pulled from the Oscard Awards data set on Kaggle. The CPI list was pulled directly from the U.S. Bureau of Labor Statistics website

- https://data.bls.gov/timeseries/CUUR0000SA0 <--- U.S. Bureau Of Labor Statistics
- https://www.kaggle.com/unanimad/the-oscar-award?select=the_oscar_award.csv <--- Oscar Awards
- https://www.kaggle.com/tmdb/tmdb-movie-metadata <--- TMDB 

<br/>

## Modules:
There's quite a few modules included in this repository, and they are all pretty well documented in the analysis file as well. As such, I'll just give a very brief description of each one here:

- create_database: Creates a SQLite database with the included data. It checks the provided database to ensure that everything is in order, and if not, uses accompanying modules to clean the source data and commit them to tables in the db.
- clean_awards: Called in the create_database module to read in the Oscar Awards data and manipulate it to create a match ID for our other data.
- clean_cast: Called in the create_database module to extract cast data from the TMDB credits. It pulls the cast fields out, unpacks the JSON into columns, and passes it as a table to commit.
- clean_crew: Performs the same function as clean_cast, except for the crew field.
- clean_movies: Called in the create_database module to extract film data from the TMDB movies data. It also merges on our CPI data to adjust all monetaries for inflation by year, and creates a matching ID for our clean_awards resultant table.
- db_query: SQL query to pull relevant fields from all of the above tables and merges them all together into a machine learning friendly data set, creating a binarized target for predicting nominee or not.
- cat_boost_v2: Stored transformations and feed into a CatBoost predictor with our data to predict nominee or not and returns success metrics
- random_forest_default: Stored transformations that one hot encode our data for a random forest, performs the forest with default settings, and returns success metrics.
- forest_grid_search: Performs a grid search for our random forest to find a set of "best" hyperparameters to plug into it for maximized returns
- random_forest_tuned: Performs the random forest with our grid_search's returned hyperparameters.
- Analysis: The code for performing the final analysis included in the main notebook of this repository

<br/>

## Other Included Files:

- Academy_Awards_Data.db: The analysis database is included to spare you the computing time of creating it at the start if you wish
- Analysis.ipynb: The main attraction of this repository! Give this a read to compare our models at work and see the final results of our analysis.
