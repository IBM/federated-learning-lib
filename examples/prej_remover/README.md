## Dataset Setup

Currently, for Federated Prejudice Removal we support following datasets:

* Adult
* Compas

# Running Federated Prejudice Removal in FL

This example explains how to run Federated Prejudice Removal, a federated learning implementation of the [Kamishima Algorithm](https://github.com/algofairness/fairness-comparison/tree/master/fairness/algorithms/kamishima). We use this algorithm in our paper on bias mitigation and federated learning; see it [here]()https://arxiv.org/abs/2012.02447).

This example requires the `gensim` library. Run the following:
    ```
    pip install gensim
    ```

The following preprocessing was performed in `AdultPRDataHandler` on the original dataset:
  * Drop following features: `workclass`, `fnlwgt`, `education`, `marital-status`, `occupation`, `relationship`, `capital-gain`, `capital-loss`, `hours-per-week`
  * Map `race`, `sex` and `class` values to 0/1
  * Split `age` and `education` columns into multiple columns based on value
  * Map `native-country` values to an integer 1-7 based on continent

  Further details in documentation of `preprocess()` in AdultPRDataHandler.

The following preprocessing was performed in `CompasPRDataHandler` on the original dataset:
  * Map `sex` values to 0/1 based on underprivileged/privileged groups
  * Filter out rows with values outside of specific ranges for `days_b_screening_arrest`, `is_recid`, `c_charge_degree`, `score_text`, `race`
  * Quantify length_of_stay from `c_jail_out` and `c_jail_in`
  * Quanitfy `priors-count`, `length_age_cat`, `score_text`, `age_cat`
  * Drop following features: `id`, `name`, `first`, `last`, `compas_screening_date`, `dob`, `age`, `juv_fel_count`, `decile_score`, `juv_misd_count`,
        `juv_other_count`, `priors_count`, `days_b_screening_arrest`, `c_jail_in`, `c_jail_out`, `c_case_number`, `c_offense_date`, `c_arrest_date`,
        `c_days_from_compas`, `c_charge_desc`, `is_recid`, `r_case_number`, `r_charge_degree`, `r_days_from_arrest`, `r_offense_date`, `r_charge_desc`,
        `r_jail_in`, `r_jail_out`, `violent_recid`, `is_violent_recid`, `vr_case_number`, `vr_charge_degree`, `vr_offense_date`, `vr_charge_desc`,
        `type_of_assessment`, `decile_score.1`, `screening_date`, `v_type_of_assessment`, `v_decile_score`, `v_score_text`, `v_screening_date`, `in_custody`,
        `out_custody`, `priors_count.1`, `start`, `end`, `event`
  * Split `age-cat`, `priors_count` and `c_charge_degree` columns into multiple columns based on value

  Further details in documentation of preprocess() in CompasPRDataHandler.

No other preprocessing is performed.

- Set FL environment by running:

    ```
    ./setup_environment.sh <env_name>  # setup environment
    ```
- Activate a new FL environment by running:

    ```
    conda activate <env_name>          # activate environment
    ```
- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d <dataset_name> -pp <points_per_party>
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n <num_parties> -f prej_remover -d <dataset_name> -p <path>
    ```
- In a terminal running an activated FL environment, start the aggregator by running:
    ```
    python -m ibmfl.aggregator.aggregator <agg_config>
    ```
    Type `START` and press enter to start accepting connections
- In a terminal running an activated FL environment, start each party by running:
    ```
    python -m ibmfl.party.party <party_config>
    ```
    Type `START` and press enter to start accepting connections.

    Type  `REGISTER` and press enter to register the party with the aggregator.
- Finally, start training by entering `TRAIN` in the aggregator terminal.
