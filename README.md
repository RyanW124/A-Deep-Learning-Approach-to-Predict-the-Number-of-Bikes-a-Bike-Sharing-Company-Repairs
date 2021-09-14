# A Deep Learning Approach to Predict the Number of Bikes a Bike Sharing Company Repairs
Ryan wong

**How to Run Experiment**

Simply go to Code/main.ipynb and rerun all the cells. Change the seed number in line 8 of the first cell if you want a different random seed. If you want to retrain any models, simply delete the corresponding .pt file in Data.

Changing the clustering would require much more work.
1. Create a new directory called "Bike" in Data
2. Go to https://s3.amazonaws.com/tripdata/index.html and download all files from 201306 until 202101, except for 201307-201402-citibike-tripdata.zip
3. Unzip the downloaded files into the Bike directory created in step 1
4. Make sure the files appear in the same order as https://s3.amazonaws.com/tripdata/index.html
5. Delete Data/data.dat and Data/stations.dat
6. Rerun all cells in Code/main.ipynb
