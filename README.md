# In The Heatmap Generator from image Project we can run this project using Two options.
## First download Image classification pretrained model from this link :
    https://drive.google.com/file/d/1wNAX0dayz7Eogg8hzTb3i4ZfqXUMmWa4/view?usp=sharing
Then unzip it  save the model folder inside this current directory.
## Run Using start.sh
    we can run this project using the command sh start.sh
    In this start.sh file it first install dependencies using the command 
    pip install -r requirements.txt
    then it run the Poject on a host=0.0.0.0 and port 3000. you can change the host and port number according to your requirements inside the
    start.sh file.
## Run Using Docker file
    we can run this project using Docker compose file.
    In this compose file it first copy the files and install dependencies inside a docker container
    then it run the Docker on a host=0.0.0.0 and port 8004. you can change the host and post number inside the DockerFile and 
    also dockercompose file.
    the following command to run docker compose 
        first run    docker-compose build
        then run   docker-compose up


