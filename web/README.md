---
title: Waste Sorting Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: cc-by-nc-sa-4.0
short_description: Sort a waste photo into six recycling categories, in-browser
---

# Waste sorting classifier

Upload a photo of one waste item: the model returns the probability of each of six recycling categories (cardboard, glass, metal, paper, plastic, trash).

- **Runs entirely in your browser** with [ONNX Runtime Web](https://onnxruntime.ai): photos are never uploaded anywhere. The page checks the model and every WebAssembly file against their recorded SHA-256 before using them.
- **Model (version 4):** EfficientNet-B0 fine-tuned on [TrashNet](https://github.com/garythung/trashnet) studio photos and [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) photos of real waste at a landfill. On test photos with no near-duplicate in training: 88.7% accuracy on real waste (version 3, trained on studio photos only: 33.0%) and 94.8% on studio photos.
- **Same numbers as the Python pipeline, in every browser:** the page decodes images itself, with WebAssembly builds of the libraries Pillow uses (libjpeg-turbo 3, libwebp) and its own PNG decoder, and reproduces Pillow's resize exactly. Checked in Chrome, Safari's engine (WebKit) and Firefox on every validation photo and on 139 unusual files; damaged files are refused like the API refuses them. CMYK JPEGs and animated WebP are refused here although the API accepts them.
- **Scope:** one item, photographed from above, on a plain background or an ordinary floor. A familiarity check says when a photo is not one the model knows (it flagged 84% of everyday photos of people, animals and places, and about 5% of new waste photos).
- **Examples:** the first validation photo per class and source that the model gets right. They show what it handles, not its error rate.
- **Licence:** the model is trained on RealWaste (CC BY-NC-SA 4.0) and TrashNet (MIT), so it is released under CC BY-NC-SA 4.0. Code: MIT.
- **Code, data handling and evaluation:** [github.com/vladzpop11/waste-sorting-classifier](https://github.com/vladzpop11/waste-sorting-classifier)
