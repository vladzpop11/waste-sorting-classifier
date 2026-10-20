---
title: Waste Sorting Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: mit
short_description: Sort a waste photo into six recycling categories, in-browser
---

# Waste sorting classifier

Upload a photo of one waste item: the model returns the probability of each of six recycling categories (cardboard, glass, metal, paper, plastic, trash).

- **Runs entirely in your browser** with [ONNX Runtime Web](https://onnxruntime.ai): photos are never uploaded anywhere. The page checks the model file against its recorded SHA-256 before using it.
- **Model:** EfficientNet-B0 fine-tuned on [TrashNet](https://github.com/garythung/trashnet). On 366 test photos of objects it never saw in training: 94.3% accuracy (95% CI 90.3–96.9%), macro F1 0.936. The weakest class is trash (F1 0.89, 19 photos). On the 25 test photos least like any training photo, accuracy was 68%.
- **Same numbers as the Python pipeline:** the page's preprocessing reproduces Pillow's resize exactly. On all 360 validation photos, uploaded through this page in Chrome, it gives the same prediction as the Python evaluation, with probabilities within 3e-6; so do 35 unusual files (every EXIF rotation, transparency, 16-bit, CMYK, MPO), and 5 invalid ones are refused like the API refuses them.
- **Scope:** TrashNet photos show one object on a plain background. Busy scenes, several objects or unusual lighting are outside what the model has learned.
- **Examples:** the first validation photo per class that the model gets right. They show what it handles, not its error rate.
- **Code, data handling and evaluation:** [github.com/vladzpop11/waste-sorting-classifier](https://github.com/vladzpop11/waste-sorting-classifier)
