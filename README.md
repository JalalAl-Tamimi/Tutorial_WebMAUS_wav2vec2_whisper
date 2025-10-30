# Tutorial_WebMAUS_wav2vec2_whisper

This repository contains all necessary material for the tutorial: 
Jalal Al-Tamimi (2025): *Arabic Forced Alignment: From WebMAUS to Whisper and wav2vec2* delivered during the 11th RJCP (Rencontres Jeunes Chercheurs en Parole) - Workshop TAL (LLF) on 5th of November 2025.

Here is a list of all required files:

## Forced alignemnt with wav2vec2

The original material for the forced-alignment with wav2vec2 comes from [this website](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html), which was adapted to include modules to install locally in addition to generating TextGrids with boundaries for words and sounds. Here are the required files:

1. [forced_alignment_tutorial_JAT.Rmd](): This is the Rmd file using python from within RStudio
2. [forced_alignment_tutorial_JAT.html](): This is the html output
3. [forced_alignment_tutorial_JAT.py](): This is the Python script to be used
4. [forced_alignment_tutorial_JAT.ipynb](): This is the Python notebook in ipynb
5. [merged.TextGrid](): This is the Praat TextGrid file generated with two tiers; one for words and one for segments

## Automatic transcription with whisper + TextGrid

This contains original material to run whisper for automatic transcription. The files are written in the R programming language with python linked to RStudio. Here are the files required:

1. [whisper_R.Rmd](): This is an Rmd file allowing to run code in both R and the Python (towards the end)
2. [whisper_R.nb.html](): This is an R notebook with the output of results
3. [whisper_R.ipynb](): This is an ipynb file. There are issues in running the code and uploading the audio file. Use at your risk.
4. [jfk.wav](): This is the audio file used to run the whisper code. The audio file came originally from [here](https://github.com/bnosac/audio.whisper/tree/master/inst/samples)
5. [words_whisper.TextGrid](): This is the Praat TextGrid file generated with one tier for words






Cite this repository when using. 
