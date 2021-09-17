"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import yaml
from pathlib import Path
from shutil import copyfile


def stage_trial_files(generated_files_dir, local_trial_dir, machine_trial_dir,
                      config_agg_dict=None, config_party_dicts=None):
    """
    - Copy all files placed into generated_files_dir by the IBMFL generate_* scripts and place \
      them all flat into local_trial_dir. \
    - Update the paths inside the configs using machine_trial_dir, assuming that they will be \
      copied there before the agg and party processes are started. \
    - Return a dictionary with the keys corresponding to the procs ('agg', 'partyX') whose \
      values are lists of all the files needed for each of those procs.

    :param generated_files_dir: the directory passed to generate_*.py scripts; where they
    generated the ./data and ./configs folders to place their output
    :type ibmfl_dir: `str`
    :param local_trial_dir: where you want the files to be copied to
    :type local_trial_dir: `str`
    :param machine_trial_dir: where you'll place the files before the run, to update the
    configs; just specify "local_trial_dir" if you don't plan to move them again
    :type machine_trial_dir: `str`
     parse it into a dictionary and pass it here
    :type config_agg_dict: `dict`
    :param config_party_dicts: if you want to edit the party  configs before calling this
    function, parse them into dictionaries and pass them here in a list (ordered by party id)
    :type config_party_dicts: `list[dict]`
    :return: a dictionary with key for each process, listing the paths to the files it needs
    :rtype: `dict{str,list}`
    """
    from collections import MutableMapping
    from functools import reduce
    from operator import getitem
    import re

    # flattens a dictionary
    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # accesses a value in a nested dictionary with a list of keys
    def dict_set_nested(nested_dict, key_list, value):
        reduce(getitem, key_list[:-1], nested_dict)[key_list[-1]] = value

    # convert all our input strings into path objects
    generated_files_pobj = Path(generated_files_dir)
    local_trial_pobj = Path(local_trial_dir)
    local_trial_pobj.mkdir(parents=True, exist_ok=True)
    machine_trial_pobj = Path(machine_trial_dir)

    # get all the files in our relevant folders
    generated_files = tuple() \
                      + tuple(Path(f'{generated_files_dir}/configs').rglob('*.*')) \
                      + tuple(Path(f"{generated_files_dir}/data").rglob('*.*'))

    proc_file_map = {}

    # if the file is a config, open it and handle the filepaths in it
    for file in generated_files:
        if 'config_' in str(file):
            proc_label = re.search('config_(.*).yml', str(file)).group(1)
            if config_agg_dict is not None and 'agg' in proc_label:
                orig_config = config_agg_dict
            elif config_party_dicts is not None and 'party' in proc_label:
                orig_config = config_party_dicts[int(proc_label[-1])]
            else:
                with open(file, 'r') as stream:
                    orig_config = yaml.load(stream.read(), Loader=yaml.Loader)
            flat_config = flatten(orig_config)
            proc_file_map[proc_label] = []
        else:
            continue
        for k, v in flat_config.items():
            # determine if this entry contains a filepath we need to handle
            if isinstance(v, str):
                v_pobj = Path(v)
                if str(generated_files_pobj) in str(v_pobj):
                    g_filepath = v_pobj
                else:
                    continue
            else:
                continue
            l_filepath = local_trial_pobj.joinpath(g_filepath.name)
            m_filepath = machine_trial_pobj.joinpath(g_filepath.name)
            # move the file to our local trial dir
            if not l_filepath.is_file() and 'output' not in k:
                copyfile(g_filepath, l_filepath)
            # don't plan to scp it to the machine if it's an output file
            if 'output' not in k:
                proc_file_map[proc_label] += [l_filepath]
            # set the path in the config to the machine trial dir, where the run happens
            dict_set_nested(orig_config, k.split('.'), str(m_filepath))

        with open(f'{local_trial_dir}/{file.name}', "w") as local_trial_config_file:
            yaml.dump(orig_config, local_trial_config_file)
            proc_file_map[proc_label] += ['{}/{}'.format(local_trial_dir, file.name)]

    return proc_file_map
