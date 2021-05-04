## Datasets
Currently for ID3 we support following datasets:
* Adult
* Nursery

### Running Decision Tree in IBM federated learning on Adult Dataset

This example explains how to run ID3 Decision Trees trained on the
[Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult).

The following preprocessing was performed before training:
```text
    * Add column labels as ['1', '2', '3'...'13', 'class'];
    * Feature `fnlwgt` is dropped.
```
    
And in `AdultDTDataHandler` it performs the following preprocessing:
  
        * Categorize `age` feature into 4 categories,
            i.e., [0, 18] -> 'teen', [18, 40] -> 'adult',
            [40, 80] -> 'old-adult', and [80, 99] -> 'elder'.
        * Categorize `workclass` feature into 3 categories,
            i.e., ["?", "Never-worked", "Private", "Without-pay"] -> 'others',
            ["Federal-gov", " Local-gov"] -> 'gov',
            and ->["Self-emp-inc", "Self-emp-not-inc"] -> 'self'.
        * Categorize `education` feature into 5 categories,
            i.e., ["10th"," 11th", " 12th", " 1st-4th", " 5th-6th", " 7th-8th",
            " 9th"] -> "non_college",
            [" Assoc-acdm", " Assoc-voc"] ->"assoc",
            [" Bachelors", " Some-college"] -> "college",
            [" Doctorate", " HS-grad", " Masters"] -> "grad",
            and [" Preschool", " Prof-school"] ->"others"
        * Categorize `education-num` feature into 3 categories:
            i.e., [0, 5] -> '<5', [5, 10] -> '5-10', and [10, 17] -> '>10'.
        * Categorize `capital-gain` feature into 5 categories,
            i.e., [-1, 1] -> 0, [1, 39999] -> 1,[39999, 49999] ->2,
            [49999, 79999] ->3, and [79999, 99999] ->4.
        * Categorize `capital-loss` feature into 6 categories,
            i.e., [-1, 1] -0, [1, 999] ->1, [999, 1999] ->2, [1999, 2999] ->3,
             [2999, 3999] ->4, [3999, 4499] -> 5.
        * Categorize `hours` feature into 3 categories,
            i.e., [0, 20] -> '<20', [20, 40] -> '20-40', [40, 100] -> '>40'.
        * Categorize `native-coutry` into 5 categories,
            i.e., [' ?',] -> 'others',
            [' Cambodia', ' China', ' Hong', ' India', ' Iran', ' Japan',
            ' Laos', ' Philippines', ' South', ' Taiwan', ' Thailand',
            ' Vietnam'] -> 'asia',
            [' Canada', ' Outlying-US(Guam-USVI-etc)', ' United-States'] -> 'north_america',
            [' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador',
            ' El-Salvador', ' Guatemala', ' Haiti', ' Honduras', ' Jamaica',
            ' Mexico', ' Nicaragua', ' Peru', ' Puerto-Rico',
            ' Trinadad&Tobago'] -> 'south_america',
            [' England', ' France', ' Germany', ' Greece',
            ' Holand-Netherlands', ' Hungary', ' Ireland', ' Italy', ' Poland',
            ' Portugal', ' Scotland', ' Yugoslavia'] -> 'europe'.
        * Training label column is renamed to `class`

No other preprocessing was performed.

### Running Decision Tree in IBM federated learning on Nursery Dataset

This example explains how to run ID3 Decision Trees trained on the
[Nursery Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/).

The following preprocessing was performed before training:

    * Add column labels as ['1', '2', '3'...'8', 'class'];
    * Training label column is renamed to class. No other preprocessing was performed

### Steps to run the experiments
- Split data by running:

    ```
    python examples/generate_data.py -n <num_parties> -d adult -pp 500
    ```
- Generate config files by running:
    ```
    python examples/generate_configs.py -n 3 -f id3_dt  -d adult -p examples/data/adult/random
    ```
- In a terminal running an activated IBM FL environment 
(refer to Quickstart in our website to learn more about how to set up the running environment), start the aggregator by running:
    ```
    python -m ibmfl.aggregator.aggregator <agg_config>
    ```
    Type `START` and press enter to start accepting connections
- In a terminal running an activated IBM FL environment, start each party by running:
    ```
    python -m ibmfl.party.party <party_config>
    ```
    Type `START` and press enter to start accepting connections.
    
    Type  `REGISTER` and press enter to register the party with the aggregator. 
- Finally, start training by entering `TRAIN` in the aggregator terminal.