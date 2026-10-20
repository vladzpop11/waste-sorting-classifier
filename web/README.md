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
- **Model:** EfficientNet-B0 fine-tuned on [TrashNet](https://github.com/garythung/trashnet). On 366 test photos with no near-duplicate in training: 94.3% accuracy (95% CI 90.3–96.9%), macro F1 0.936. The weakest class is trash (F1 0.89, 19 photos). On the 25 test photos least like any training photo of their class, accuracy was 68%.
- **Same numbers as the Python pipeline, in every browser:** the page decodes images itself, with WebAssembly builds of the libraries Pillow uses (libjpeg-turbo 3, libwebp) and its own PNG decoder, and reproduces Pillow's resize exactly. Checked in Chrome, Safari's engine (WebKit) and Firefox: on all 360 validation photos the decoded pixels are identical to Python's and the predictions match, with probabilities within 3e-6; so do 129 unusual files (every EXIF orientation layout, transparency, 1- to 16-bit PNG, MPO, arithmetic and lossless JPEG), and damaged files are refused like the API refuses them. CMYK JPEGs and animated WebP are refused here although the API accepts them.
- **Scope:** TrashNet photos show one object on a plain background. Busy scenes, several objects or unusual lighting are outside what the model has learned. A familiarity check compares each photo with the training photos and says when it is not a photo the model knows (it flagged 92% of everyday photos of people, animals and places, and 5% of new waste photos).
- **Examples:** the first validation photo per class that the model gets right. They show what it handles, not its error rate.
- **Code, data handling and evaluation:** [github.com/vladzpop11/waste-sorting-classifier](https://github.com/vladzpop11/waste-sorting-classifier)
