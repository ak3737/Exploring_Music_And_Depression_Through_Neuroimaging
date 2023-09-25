# Exploring_Music_And_Depression_Through_Neuroimaging

## Authors
- Amina Asghar
- Nasri Binsaleh
- Onintsoa Ramananandroniaina

## Date
December 9, 2022

## Abstract
This research project explores the relationship between music and depression, focusing on the use of emotional auditory stimuli as a potential therapeutic tool for individuals with Major Depressive Disorder (MDD). We conducted an in-depth analysis of functional Magnetic Resonance Imaging (fMRI) data from participants with MDD and a control group to identify activated brain regions and assess connectivity patterns between these regions.

## Introduction
Major Depressive Disorder (MDD) is a severe mood disorder characterized by persistent negative emotions and the loss of interest in previously enjoyed activities. Previous research has suggested that individuals with MDD tend to prefer sad music. Additionally, studies have shown that the Anterior Cingulate Cortex (ACC) may play a crucial role in mood disorders. This project aims to investigate the activation of specific brain regions in response to musical and non-musical stimuli, with a particular focus on the ACC.

## Materials and Methods
### Dataset
We obtained our dataset from OpenNeuro, comprising 39 participants, including 19 with MDD and 20 never-depressed control participants. Participants were exposed to positive and negative emotional musical and non-musical auditory stimuli during fMRI scanning.

### Data Preprocessing
We performed data preprocessing using the FMRIPREP pipeline in Python, which included brain extraction, motion correction, susceptibility distortion correction, alignment to the T1 image, MNI152 transformation, and confound regression.

### T-Contrast Analysis
We utilized T-contrast analysis to identify activation patterns in fMRI images. This involved constructing a General Linear Model (GLM) and computing t-values for each voxel to identify highly activated brain regions.

### Functional Connectivity Analysis
We conducted functional connectivity analysis using the Yeo 2011 thick 7 atlas to parcellate the brain into different regions. Correlation matrices were constructed to assess connectivity between regions of interest, particularly the Anterior Cingulate Cortex (ACC) and the Auditory Cortex (AC).

## Results and Analysis
### T-Contrast Analysis
Our analysis revealed that the Auditory Cortex (AC) was highly activated during the experiment, which is consistent with the nature of the auditory stimuli. Additionally, the activation in the AC was lower in the MDD group compared to the control group.

### Functional Connectivity Analysis
The correlation matrix showed that the AC had the highest connectivity with the ACC, confirming our hypothesis. Notably, the connectivity between the ACC and AC was lower in the MDD group than in the control group.

## Discussion and Conclusion
The results of this research support the idea that emotional auditory stimuli, such as music, can significantly impact brain activation patterns, particularly in individuals with MDD. The lower connectivity between the ACC and AC in the MDD group suggests potential implications for targeted auditory treatments for depression.

## References
- [1] J. Truschel, “Depression Definition and DSM-5 Diagnostic Criteria,” Psycom.net, Jul. 24, 2018.
- [2] K. Yucel et al., “Anterior Cingulate Volumes in Never-Treated Patients with Major Depressive Disorder,” Neuropsychopharmacology, vol. 33, no. 13, pp. 3157–3163, Mar. 2008.
- [3] S. Yoon, E. Verona, R. Schlauch, S. Schneider, and J. Rottenberg, “Why do depressed people prefer sad music?,” Emotion (Washington, D.C.), vol. 20, no. 4, p. 10.1037/emo0000573, 2019.
- [4] P. H. Rudebeck et al., “A role for primate subgenual cingulate cortex in sustaining autonomic arousal,” Proceedings of the National Academy of Sciences, vol. 111, no. 14, pp. 5391–5396, Mar. 2014.
- [5] R. Lepping et al., “Neural Processing of Emotional Musical and Nonmusical Stimuli in Depression,” OpenNeuro, Jul. 17, 2018.
