## Experiment Manager Dashboard
Developers and Federated Learning researchers often need to experiment with combinations of models, datasets and fusion algorithms. While a command line interface provides access to low-level granularities, the Experiment Manager Dashboard facilitates setting up an FL experiment, orchestrating the same on the local machine or remote VMs, and finally collating results from the experiment- all through a single Jupyter Notebook interface.

### Features:
- Setup FL experiments using either of Keras, PyTorch, Scikit-learn and TensorFlow models
- Aggregate during FL training using fusion algorithms such as _Iterative Averaging_, _Coordinate Median_, _Krum_, _Zeno_ and several others, including the _Fed+ family_ of algorithms
- Use pre-populated or custom datasets for parties to train on
- Orchestrate experiments on local machine or on remote VM(s)
- Visualise metrics after training completes, or collect logs for your own postprocessing
### Setup

**Note:**
There are limitations to using `conda` when spawning the many FL processes when orchestrating an experiment. This is also noted in [Issue #7980](https://github.com/conda/conda/issues/7980). 
	
A virtual environment setup through `venv`, a [module](https://docs.python.org/3/library/venv.html) from the standard library has no such issues. Therefore, until there is a native cross-platform fix for this, the dashboard shall only support `venv`.

--- 

The UI components in the Notebook use [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html) which can be installed using:

`pip install ipywidgets` 

In order to hide away the source code, as you interact with the dashboard, install [Jupyter notebook extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions#jupyterlab) as 
`pip install jupyter_contrib_nbextensions` 

Once installed, follow the instructions at their GitHub page to configure it to work on your end.

The dashboard functionality is split between two classes:
- `DashboardUI`: includes all Notebook widgets, their handlers and some related logic
- `ConfigManager`: includes all configuration manipulation logic, as well as various objects that are populated to pass onto the runner module for running experiments

### Run the notebook
Finally, running `jupyter notebook Experiment_Manager_Dashboard.ipynb` from within the IBMFL virtual environment should open up the Notebook dashboard in your web browser. Once the notebook is up and running, you may follow the steps as shown [here](usage_guide.md).

**Note**: As the Notebook is aimed at orchestrating experiments on remote virtual machines as well as on the same machine as the dashboard, please ensure that passwordless `ssh` is enabled for all these machines, *including* `localhost`.

**Steps for setting up passwordless ssh to localhost**:
1. Generate ssh keys if not already present:
	```
	ssh-keygen -t rsa -b 4096
	```
	Press enter to accept the default file location and file name, and then enter again to opt for an empty passphrase.
2. The keys generated will be listed in `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`
3. Copy the public key to:
	- a remote server:
		```
		cat ~/.ssh/id_rsa.pub | ssh remote_username@server_ip_address "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
		```

		This will append the `~/.ssh/id_rsa.pub` file to the remote server's `~/.ssh/authorized_keys` file.

	- localhost:
		```cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys``` and ```chmod 600 ~/.ssh/authorized_keys```
		
		This will append the public key to the local `~/.ssh/authorized_keys` file.

4. To test, try `ssh remote_username@server_ip_address` or `ssh localhost` and it should go through without asking for a password.

Note that while this is the general set of steps, some OSes may require additional settings to be tweaked. 

For macOS, you may have to additionally go to _System Preferences_ > _Sharing_ > _Remote Login_ and add your user to the list of users allowed remote access into the machine (in case you get a permission error)
