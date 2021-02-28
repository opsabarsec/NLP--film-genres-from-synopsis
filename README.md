# NLP--film-genres-from-synopsis
![pellicola](film.jpg)
## 1. Background
Public movies’ database such as IMDB provides genreinformation to assist searching. The tagging of movies’genres is still a manual process which involves the col-lection  of  users’  suggestions  sent  to  known  email  ad-dresses of IMDB. Movies are often registered with in-accurate  genres.   Automatic  genres  classification  of  amovie based on its synopsis not only speeds up the clas-sification process by providing a list of suggestion butthe result may potentially be more accurate than an un-trained human.

## 2. The data
Data have been downloaded from Radix challenge Kaggle dataset

https://gitlab.com/radix-ai/challenge

##  3. The model
This Python [notebook]() contains the code to obtain automatically the film genres from the synopsis.

Best results have been obtained with deep learning, coding a very simple Keras LSTM.
Higher accuracy than classical ML models such as Logistic Regression (50% vs. 10%)

![LSTM](LSTM.png)


