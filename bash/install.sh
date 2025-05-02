#!/bin/bash

# this bash script moves all the executable scripts to usrlocalbin
# where they can be run globally

# function to record wav files
sudo chmod +x recordk
sudo cp recordk /usr/local/bin/

# function to train with all wav files
sudo chmod +x ingest
sudo cp ingest /usr/local/bin/

# function to grade all the wav files
sudo chmod +x grade
sudo cp grade /usr/local/bin/

# function to train data from wav file
sudo chmod +x train
sudo cp train /usr/local/bin/

# function to take both csv files and fit a model
sudo chmod +x learn
sudo cp learn /usr/local/bin/

# function to use model and predict keys from wav file
sudo chmod +x predict
sudo cp predict /usr/local/bin

# function to inspect wav file
sudo chmod +x listen
sudo cp listen /usr/local/bin

# function to find background noise value
sudo chmod +x background
sudo cp background /usr/local/bin

# program to count the distribution of letters in given file
sudo chmod +x lcount
sudo cp lcount /usr/local/bin
