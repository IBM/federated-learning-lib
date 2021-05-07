## Experiment Manager Dashboard
Jupyter Notebook frontend for orchestrating Federated Learning experiments

### Setup
The UI components in the Notebook use [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html) which can be installed using:

`pip install ipywidgets` 
or 
`conda install -c conda-forge ipywidgets`

In order to hide away the source code, as you interact with the dashboard, install [Jupyter notebook extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions#jupyterlab) as 
`pip install jupyter_contrib_nbextensions` 
or
`conda install -c conda-forge jupyter_contrib_nbextensions`

Once installed, follow the instructiions at their Github page to configure it to work on your end.

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

For MacOS, you may have to addtionally go to _System Preferences_ > _Sharing_ > _Remote Login_ and add your user to the list of users allowed remote access into the machine (in case you get a permission error)
