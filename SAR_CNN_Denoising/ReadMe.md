## STEPS
1. Convert SAR SLC images in .cos format to .npy format 
2. Generate the sublooks with "Sublooks_Gen.py"
3. Generate the patches for the Dataset with "Patches.py"
4. Patches are loaded into the model by "Dataloader_class.py"
5. Trainig model
6. To test the model, the global parameter ONLYTEST is set to "true" and then there are the scripts: "Test.py", "save_patches.py", "input_vs_output.py".

To facilitate the tests there is the file "CONFIGURATION.json" with the global parameters, so it is only necessary to change them in the configuration file. For more flexibility in experimentation "Loss_function.py" and "Activation_functions.py" have some loss functions and activation functions respectively, to select them modify them in "CONFIGURATION.json"

### Dependencies 
To install conda env dependencies run this command

```console
foo@bar:~$ conda create --name <env> --file requirements.txt
```

### Global parameters for traiing
Here is the JSON format.

L,M,m are fixed 

Model: Chose model

CKPT: Checkpoint used to restore training or test

SELECT : Loss function 

ACTIVATE ONLYTEST as true after trainning to see the results

```json
{
    "global_parameters": {
      "L": 1,
      "M": 10.089038980848645,
      "m":-1.429329123112601,
      "MODEL" : "Autoencoder_Wilson_Ver1",
      "ARCHIVE": " Jarvis",
      "CKPT": "/home/tonix/Documents/Dayana/mis_checkpoints/WilsonVer11_Net_co_log_likelihood_leaky_relu_10_20.ckpt",
      "INPUT_FOLDER": "/home/tonix/Documents/Dayana/Dataset/GeneralA",
      "FUNCTION": "leaky_relu",
      "REFERENCE_FOLDER": "/home/tonix/Documents/Dayana/Dataset/GeneralB",
      "SELECT": "co_log_likelihood",
      "learning_rate": 0.005,
      "WEIGHT_DECAY": 0.0,
      "batch_size": 10,
      "epochs": 85, 
      "ONLYTEST" : false, 
      "NUMWORKERS": 16
    }
  }
```


