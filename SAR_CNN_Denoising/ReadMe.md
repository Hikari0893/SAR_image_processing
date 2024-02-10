## STEPS
1-Convert SAR SLC images in .cos format to .npy format 
2-Generate the sublooks with "Sublooks_Gen.py"
3-Generate the patches for the Dataset with "Patches.py"
4-Patches are loaded into the model by "Dataloader_class.py"
5-Trainig model
6-To test the model, the global parameter ONLYTEST is set to "true" and then there are the scripts: "Test.py", "save_patches.py", "input_vs_output.py".

## To facilitate the tests there is the file "CONFIGURATION.json" with the global parameters, so it is only necessary to change them in the configuration file. For more flexibility in experimentation "Loss_function.py" and "Activation_functions.py" have some loss functions and activation functions respectively, to select them modify them in "CONFIGURATION.json"

