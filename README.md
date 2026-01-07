# NEURO140 Final Project

## "Applying perceptual modulation for qualitative interpretability of deep learning models in sex identification on retinal fundus images"

This is my final project for the course NEURO140: Natural and Artificial Intelligence at Harvard College in the Spring 2024 semester. The project is made possible thanks to the ODIR-5K dataset available at https://odir2019.grand-challenge.org/dataset/. The full project report is available in the repository as *finalreport.pdf* but a short summary comes below.

### Background
This was my first class in the field of computational neurosciece and anything CS-adjacent. Fascinated by the course title, I wrote my first Python code over the preceding winter break and spent the semester teaching myself PyTorch for this course whose main component was a full-semester ML project related to neuroscience. In looking for a project, I was incredibly intrigued, to the point of that solely influencing my college major, by a paper (*"Strong and Precise Modulation of Human Percepts via Robustified ANNs" by G. Gaziv, 2024*) on how a feedback loop of ANNs can give insights about how a CNN works, and most fascinating, how these insights indicate artificial and natural vision work similar in object recognition. From that, this project was born, as additional research (outlined more in the project report) showed that in the specific field of CNNs applied to gender detection from retinal fundus images, there was no general consensus on which anatomical features are most indicative in such a task, and thus there existed (and still does) a question to be answered.

### Result
Admittedly, with the little background experience, I bit off more than I could chew. The main problem of the project turned out to be the limited size of the dataset, which is the part of the project that leaves the most room for future improvement. Initial ideas including model ensembling or artificially growing the dataset with augmented copies (in the end, random augmentation was just applied to the datapoints we had) but were not thoroughly applied. As such, the final model was far from perfect. Despite all of this, however, it still outperformed naive models in terms of accuracy and, most significantly, the visual analysis of the effects of the ANN (where modulation occured in each sample put through the aforementioned feedback loop) showed high potential of this approach, as all modulation occured in anatomical areas that earlier research outlined as possibly explanatory for gender detection (see the *advimages folder*).

### File description:  
<br>
**robust.ipynb** --> application of the robustness library to adversarially train a pre-trained model and then apply targeted modulation attacks  <br>
**compositemodel.ipynb** --> model to pre-process retinal fundus images, load them into DataLoaders and then train a ResNet50 model on them.  <br>
**compsite_dataset.py** --> the dataset class for the retinal fundus images (split into a separate file due to multiprocessing issues when class is defined in the notebook)  <br>
**advimages** --> directory containing approx. 400 examples of the targeted modulations (accuracy on adverserial examples: 56.8% --- accuracy on pre-trained model: 62.3%)  <br>
