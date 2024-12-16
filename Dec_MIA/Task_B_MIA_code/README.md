# Folders
## Data
Training dataset for target model
## Data_shadow
Training dataset for shadow model
## models:
model output directory includes:
- target model
- shadow model
- attacker lg classifier

# Notebooks
- **0.Data preprocessing**
- **1.Target_model_training**: fine-tune a BERT case model on 13k OLID data samples.
- **2.Shadow_model_training**: fine-tune a BERT case model on 5k OLID and 3k HASOC data samples.
- **3.Attack_model_training_and_testing**: Generate and process data for the attack model training. The model is trained and validated using 5-fold cross-validation.


# MIA_util.py
- input_to_sen_embedding
- plotting
