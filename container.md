## Create docker container image

Install Docker engine by following instructions from official Docker webpage:
https://docs.docker.com/engine/install/

Next, use Dockerfile provided in the repository to create a container image.

`docker build . -t ibmfl-lib`

You can check the available container images by issues `docker images` command. Next, run the newly created image in an interactive mode.

`docker run -it ibmfl-lib /bin/bash`

You will have access to the IBMFL library inside container. IBMFL is already installed so you do not need to install it again.
