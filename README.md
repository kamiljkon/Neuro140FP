Code attempting to apply targeted adverserial attacks as an explainability method for sex determination on retinal fundus images.

File description:
**robust.ipynb** --> application of the robustness library to adversarially train a pre-trained model and then apply targeted modulation attacks
**compositemodel.ipynb** --> model to pre-process retinal fundus images, load them into DataLoaders and then train a ResNet50 model on them.
**compsite_dataset.py** --> the dataset class for the retinal fundus images (split into a separate file due to multiprocessing issues when class is defined in the notebook)

**advimages** --> directory containing approx. 400 examples of the targeted modulations (accuracy on adverserial examples: 56.8% --- accuracy on pre-trained model: 62.3%)
