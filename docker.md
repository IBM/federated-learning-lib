# Create IBMFL Docker Container Image

Install Docker engine by following instructions from official Docker webpage: <https://docs.docker.com/engine/install>

Next, use `Dockerfile` provided in the repository to create a container image. The `-t ibmfl` flag can be changed to whichever image name is desired.

```sh
docker build -t ibmfl .
```

Just like installing the wheel file, by default, this will not install any additional machine learning libraries. You must specify the desired model training backend library by using the `--build-arg` flag to specify a `BACKEND` argument. The following backends are supported:

```sh
# Build image with Scikit-learn backend
docker build --build-arg BACKEND=sklearn -t ibmfl .
# Build image with PyTorch backend
docker build --build-arg BACKEND=pytorch -t ibmfl .
# Build image with Keras (TensorFlow v1) backend
docker build --build-arg BACKEND=keras -t ibmfl .
# Build image with TensorFlow v2 backend
docker build --build-arg BACKEND=tf -t ibmfl .
# Build image with RLlib backend
docker build --build-arg BACKEND=rllib -t ibmfl .
```

You can also use multiple backends using a comma separated list. For example:

```sh
# Build image with Scikit-learn and Keras (TensorFlow v1) backend
docker build --build-arg BACKEND=sklearn,keras -t ibmfl
# Build image with PyTorch and TensorFlow v2 backend
docker build --build-arg BACKEND=pytorch,tf -t ibmfl
# Build image with Scikit-learn, TensorFlow v2, and RLlib backend
docker build --build-arg BACKEND=sklearn,tf,rllib -t ibmfl
```

To build the docker with crypto using fully homomorphic encryption (HE), add `crypto` to the `BACKEND` list. Typically this is paired with other backends.

```sh
# Build image with only crypto and no backend
docker build --build-arg BACKEND=crypto -t ibmfl
# Build image with Keras (TensorFlow v1) backend and crypto
docker build --build-arg BACKEND=keras,crypto -t ibmfl
# build image with PyTorch and TensorFlow v2 backend and crypto
docker build --build-arg BACKEND=pytorch,tf,crypto -t ibmfl
```

To build the docker to be run using OpenShift, add `openshift` to the `BACKEND` list. This **must** be paired with other backends and can optionally be paired with crypto.

```sh
# Build image with Keras (TensorFlow v1) backend for OpenShift
docker build --build-arg BACKEND=keras,openshift -t ibmfl
# Build image with Keras (TensorFlow v1) backend and crypto for OpenShift
docker build --build-arg BACKEND=keras,crypto,openshift -t ibmfl
# build image with PyTorch and TensorFlow v2 backend for OpenShift
docker build --build-arg BACKEND=pytorch,tf,openshift -t ibmfl
```
