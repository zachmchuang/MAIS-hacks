After first grid search testing, learning rate is fixed. 

Vary kernel size and number of layers from 3 to 10 

Checklist : 
* Find bigger dataset
* merge the two datasets together
* Make a live  prediction function (Oscar)
* Make the UI

To add a deploy key generate a public key on the machine and add it to deploy keys on github. 

Then run this command whenever a git command `[GIT COMMAND]` is involved : 

`GIT_SSH_COMMAND='ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes' [GIT COMMAND]`

Or alternatively to run the git command everytime : 
`git config core.sshCommand "ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes"` 

So far I made three different architectures : 

In `model.ipynb` I implemented the `EmotionDetector` class which is a stack of Conv2D followed by a fully connected layer. 
This model wasn't able to go over 45% (validation) 

A slightly tweaked version of `EmotionDetector2`, was able to achieve an accuracy of 57% (validation) 

I also implemented a different architecture with MLP+SLTM that basically extracts the audio features (metrics)from the given audio and feeds them into an MLP. Since the time bin is 1 (I generate those value for a given recording), I then perform a Conv1D on the vector to extract feature maps. The code implementation for this model can be found in `mlp-approach.ipynb`. 

I trained the model on the crema-d-mirror dataset on 50 epochs and got an accuracy of 54%. If you want to do further training use the saved model`[MLP-APPROACH]checkpoint.pth`. 

Note that in order to extract the audio features from the input audios I had to restructure the dataset using Zach's approach for storing the numpy arrays corresponding to the spectrograms of audios in the format 

The initial wav files were stored in `crema-d-minor/AudioWAV`

At the end I have the following 
```
arranged_wav
├── ANG
│   ├── 1001_ITH_ANG_XX.wav
│   └── ...
└── FEA
    ├── 1001_ITH_ANG_XX.wav
    └── ...
```    

Datasets that we can use so far : 
* https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio (RAVDESS)
* https://www.kaggle.com/datasets/ejlok1/cremad (CREMA-D)
* https://utoronto.scholaris.ca/collections/036db644-9790-4ed0-90cc-be1dfb8a4b66/search (uoft TESS)

Key elements to maximize performance are : 
* Data Augmentation : i.e generating dummy spectrograms/audios based on the existing ones by applying small tweaks (rotation, translation) any kind of linear transformation is fine and can be learned
* Batch normalization
* Audio normalization
