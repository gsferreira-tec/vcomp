# Speaker Guide — PV Defect Classification Presentation

**Project:** Classification of Defects in Photovoltaic Modules  
**Course:** Computer Vision — Assignment 2, FEUP  
**Authors:** Guilherme Sousa Ferreira, Diogo Loayza Duarte, Nuno Pinho  
**Estimated duration:** 15–18 minutes (+ 2–3 minutes for questions)  
**Slides:** `presentation_slides.html` (open in any browser; use arrow keys or space to advance)

This guide is written as a continuous, developed script divided among three speakers. Each section indicates the corresponding slide number. Speakers should not read verbatim — use this as a narrative backbone and adapt naturally to the audience and time available.

---

## Speaker 1 — Context, Dataset & Preprocessing (~5–6 min)

**Slides 1–7**

---

### Slide 1 — Title

Good morning/afternoon. We are Guilherme Sousa Ferreira, Diogo Loayza Duarte, and Nuno Pinho, and today we will present our Computer Vision assignment on the **classification of defects in photovoltaic modules**.

Our project applies deep learning to thermal infrared images of solar panels, with the goal of automating anomaly detection in a way that does not depend on expert inspectors being physically present at every solar farm.

---

### Slide 2 — Motivation

To understand why this problem matters, we only need to look at recent events in Portugal and Spain, where failures in solar-farm infrastructure have had serious operational and economic consequences. When a defect goes undetected, it can escalate from a localized hot spot into a full module failure, production loss, or even a safety hazard.

The traditional response is manual inspection — experts flying drones or walking rows of panels, interpreting thermal images by eye. That approach does not scale: a modern solar farm can contain tens of thousands of modules, and the inspection window is often limited to specific weather and lighting conditions.

Our project asks a practical question: **can we train a neural network to read these thermal signatures automatically?** Specifically, can it not only tell us *whether* something is wrong, but also *what kind* of defect we are looking at? That is the problem we set out to solve.

---

### Slide 3 — Project Objectives

The assignment gave us a well-defined set of requirements, and we structured the entire notebook around them.

First, we had to build **three AI models** operating on thermal signatures at different levels of granularity:
- a **binary classifier** that separates healthy modules from anomalous ones;
- an **11-class model** that identifies the specific type of defect, ignoring healthy panels entirely;
- and a **12-class model** that performs full taxonomy classification, including the no-anomaly class.

Second, we had to document and justify our **data augmentation** pipeline.

Third, we had to **compare our custom model** against established architectures — we chose ViT, VGG16, ResNet50, and MobileNetV3 — using accuracy, macro-F1, confusion matrices, and parameter count.

Finally, we were asked to **discuss our results in the context of a reference paper** by Ramadan and colleagues, who report state-of-the-art results on the same dataset using a Vision Transformer. That comparison becomes important later, because it frames what our smaller model achieves — and what it does not.

Everything was implemented in **PyTorch**, using the public **InfraredSolarModules** dataset.

---

### Slide 4 — Dataset Overview

The dataset comes from Raptor Maps and contains roughly **twenty thousand infrared images** of photovoltaic modules. Each image is labelled with one of **twelve classes**: eleven distinct defect categories — things like cell defects, hot spots, cracking, shadowing, diode failures, vegetation cover, soiling, and offline modules — plus a **No-Anomaly** class for healthy panels.

The native image resolution is very small — around 40 by 24 pixels — because these are cropped thermal patches captured from UAV overflights. For compatibility with standard deep learning backbones, we resize everything to **224 by 224**.

For evaluation integrity, we adopted the same **80/10/10 split** used in the reference literature: 80% for training, 10% for validation, and 10% held out strictly for testing. The split is **stratified by class** and uses a fixed random seed, so every model in our comparison sees exactly the same data partition. That point is important: it makes our results directly comparable to each other, and it prevents accidental data leakage.

---

### Slide 5 — Key Data Challenge: Class Imbalance

Before we could train anything, we had to confront the dataset's most difficult property: **severe class imbalance**.

At the twelve-class level, the dataset looks balanced on the surface — half the images are No-Anomaly and half are defects. But within the defect categories, the distribution is extremely skewed. Some classes have nearly two thousand samples, while the rarest — such as Diode-Multi or Soiling — have only around one hundred and seventy-five. That is an order-of-magnitude difference.

This imbalance creates a trap for standard training. A model can achieve deceptively high accuracy by learning to predict the majority classes and essentially ignoring the rare ones. That is unacceptable in a safety-critical inspection setting, where missing a diode failure is far worse than a false alarm on a healthy panel.

We addressed this problem on three fronts, and all three work together rather than in isolation. First, **IR-aware data augmentation** on the training set only, to artificially expand diversity without leaking augmented samples into validation or test. Second, a **WeightedRandomSampler** that ensures each training mini-batch contains roughly equal representation from every class. Third, **Focal Loss** for the multi-class tasks, which down-weights easy, well-classified examples and forces the network to focus learning capacity on hard, rare cases.

I will now hand over to [Speaker 2], who will walk you through how we preprocessed the data, designed our model, and set up the training pipeline.

---

## Speaker 2 — Architecture, Augmentation & Training (~5–6 min)

**Slides 6–10**

---

### Slide 6 — Preprocessing Pipeline

Thank you. Let me start with the end-to-end pipeline, because every design choice downstream depends on how we prepared the data.

Raw infrared images first pass through an **Unsharp Mask sharpening filter**. In thermal imaging, many critical defects — micro-cracks, early-stage hot spots, subtle soiling — appear as soft, low-contrast gradients. Standard convolutional downsampling can smooth these away. Sharpening enhances the high-frequency spatial boundaries so that defect edges remain visible to the network. We apply this filter **offline**, once, before training, so we are not paying the computational cost every epoch.

Next, we map labels to the task at hand — binary, eleven-class, or twelve-class — using a flexible dataset class called `PVDataset` that we can reuse across all three tasks without duplicating code.

Augmentation is applied **only during training**. This is a deliberate choice to prevent data leakage: if augmented versions of training images could appear in the test set, our reported accuracy would be optimistically biased.

---

### Slide 7 — Data Augmentation

Our augmentation strategy was designed to mimic the real-world variability of **UAV-mounted thermal cameras**, not to maximize benchmark numbers artificially.

We use the **Albumentations** library and group our transforms into three categories.

For **geometric invariance**, we apply horizontal and vertical flips, random 90-degree rotations, and small affine perturbations — up to five percent translation and scaling. Drone flight direction is arbitrary relative to panel orientation, so the model must not learn that a hot spot on the left of the image is a different class from one on the right.

For **environmental simulation**, we inject Gaussian noise into ten percent of training images to mimic sensor grain and transmission artifacts, and we randomly adjust brightness and contrast by up to twenty percent to reflect changes in solar irradiance, cloud cover, and time of day.

For **structural regularization**, we use CoarseDropout, which masks one to four small rectangular regions in the image. This prevents the model from "cheating" by memorizing a single bright pixel cluster — it must learn distributed contextual features across the entire panel.

At **test time**, we go one step further with **Test-Time Augmentation**: we run inference on four views of each image — the original, a horizontal flip, a vertical flip, and a 180-degree rotation — and average the output logits before making a final prediction. This consistently improves confidence and accuracy without changing the model itself.

---

### Slide 8 — Proposed Model

For our custom architecture, we explored several options during development. A simple five-block CNN was too limited. A ViT trained entirely from scratch was too expensive and unstable on our dataset size. We settled on a **middle ground**: a four-stage **residual CNN augmented with Convolutional Block Attention Modules**, or CBAM.

The network follows the classical pyramid structure — spatial resolution halves at each stage while channel depth increases — with **residual skip connections** so gradients flow stably through eight convolutional blocks. What makes it domain-specific is the attention: in every block, **channel attention** re-weights feature maps based on which thermal signatures matter for each defect type, and **spatial attention** produces a heatmap that focuses computation on the localized hot regions where defects actually appear.

The entire model has approximately **4.12 million trainable parameters** and is trained **from scratch**, with no ImageNet pretraining. That is a deliberate experimental choice: it lets us isolate how much performance comes from the architecture and training recipe versus how much comes from generic natural-image features learned on ImageNet.

---

### Slide 9 — Baseline Models

To put our model in context, we compare against **four transfer-learning baselines** from torchvision, each representing a different architectural family.

**ViT-B/16** is a Vision Transformer that processes the image as a sequence of 16-by-16 patches. We fine-tune only the last two encoder blocks. **VGG16** is a classical deep CNN; we unfreeze the last four feature layers. **ResNet50** uses residual bottlenecks; we fine-tune the entire fourth stage. **MobileNetV3-Large** is our lightweight reference point, with only the last three inverted blocks unfrozen.

All five models — including ours — receive 224-by-224 inputs, ImageNet normalization, and a task-specific classification head. The optimizer is built only over parameters with `requires_grad=True`, so frozen backbone weights consume no optimizer memory.

The parameter counts tell an important story on their own: our ProposedModel at 4.12M sits between MobileNet and VGG, but is roughly **three and a half times smaller than ViT** and **four times smaller than ResNet50**.

---

### Slide 10 — Training Strategy

The training pipeline — implemented in a function called `run_task_pipeline` — encodes several decisions that are easy to overlook but materially affect results.

We use **AdamW** with weight decay, which is the right optimizer for fine-tuning pretrained networks because it decouples weight decay from the adaptive learning rate. For loss functions, binary classification uses **BCEWithLogitsLoss**, while multi-class tasks use **Focal Loss** with gamma equal to 2.0 and label smoothing of 0.1.

The learning rate follows a **linear warmup into cosine annealing** schedule. Warmup is especially important for ViT fine-tuning, where a cold start at full learning rate can destabilize pretrained features. Our from-scratch ProposedModel gets a longer warmup — five epochs versus two — because both its backbone and its head start from random initialization.

We save the best checkpoint based on **validation macro-F1**, not accuracy. On imbalanced data, accuracy is misleading: a model that ignores all rare classes can still look good. Macro-F1 gives equal weight to every class, which aligns with our inspection use case.

Finally, because we train five models sequentially on a single GPU, we explicitly release GPU memory between runs — deleting models, running garbage collection, and calling `torch.cuda.empty_cache()` — to avoid out-of-memory failures mid-experiment.

I will now pass to [Speaker 3], who will present our results, compare them with the literature, and close with our conclusions.

---

## Speaker 3 — Results, Literature & Conclusion (~5–6 min)

**Slides 11–17**

---

### Slide 11 — Binary Results

Thank you. Let us look at what all of this engineering actually produced, starting with the simplest task: **binary classification**.

Here, the job is straightforward — decide whether a module is healthy or anomalous. All five models perform reasonably well, clustering within about five percentage points. But our **ProposedModel achieves the best result**: **92.45% accuracy and 92.23% F1**, slightly ahead of ViT-B/16 at 91.55% and 91.22%, and clearly ahead of VGG, ResNet, and MobileNet.

The takeaway for this task is about **efficiency**, not a dramatic performance gap. We essentially **match or beat a Vision Transformer while using roughly three and a half times fewer parameters**. For a binary triage step — the first filter in an automated inspection pipeline — that efficiency matters, because this model could run on an embedded device aboard a drone, which is exactly the deployment scenario described by Le and colleagues in their edge-device inspection work.

The design choices driving this result are CBAM attention focusing on thermal hot regions, IR-aware augmentation simulating real sensor noise, and balanced mini-batches via the WeightedRandomSampler.

---

### Slide 12 — 11-Class Results

The picture changes significantly when we move to **eleven-class defect-only classification**. This is, in our view, the **hardest task** in the assignment: every input image contains a defect, the classes are visually similar IR patterns, and the class imbalance among defects remains severe.

On this task, our ProposedModel separates clearly from every transfer-learning baseline: **73.10% accuracy and 70.66% macro-F1**. The next-best model is ViT at 61.29% macro-F1 — that is a gap of **more than nine F1 points**. VGG16, in particular, struggles badly at 49.71% macro-F1.

This is the strongest evidence in our entire project that **ImageNet pretraining is not a universal advantage**. Natural-image features help when the task is coarse — anomaly versus no-anomaly — but they do not automatically transfer to the fine-grained problem of distinguishing, say, a single hot spot from a multi-cell hot spot in a low-resolution thermal patch. Our domain-specific CNN, trained from scratch with CBAM attention, Focal Loss, and IR-aware augmentation, learns features that are actually tuned to this thermal domain.

The per-class breakdown is also encouraging for the rarest categories. Despite having only seventeen test samples, **Diode-Multi reaches F1 of 0.94**, and **Diode reaches 0.97**. That confirms our imbalance strategy is working — not perfectly, but meaningfully.

The remaining confusion is mostly between **pairs that differ only in the number of affected cells** — Cell versus Cell-Multi, Hot-Spot versus Hot-Spot-Multi — a limitation we will return to shortly.

---

### Slide 13 — 12-Class Results

When we add the **No-Anomaly class back** to form the twelve-class problem, the dataset's original imbalance reappears: half the test set is healthy panels. This is the scenario that Ramadan and colleagues focus on in their reference paper.

Our ProposedModel again posts the **best accuracy at 78.40%** and the **best macro-F1 at 68.28%**. ViT achieves slightly lower accuracy at 77.45%, but more importantly its macro-F1 is only 63.51% — nearly five points behind ours. That gap tells us ViT's accuracy advantage, where it exists, comes largely from being conservative on the easy No-Anomaly class, not from superior defect classification.

On individual classes, our model reaches strong F1 scores on **Diode at 0.95**, **Diode-Multi at 1.00**, **No-Anomaly at 0.88**, and **Cracking at 0.80** — numbers that are competitive with published results on this same dataset, including Le et al.'s 85.9% multi-class accuracy with a much heavier ResNet ensemble.

---

### Slide 14 — Overall Comparison

Stepping back, the full comparison table makes the pattern clear.

Our ProposedModel **wins five of the six reported metrics** across all three tasks. The only partial exception is multi-12 accuracy, where ViT is marginally ahead but with worse macro-F1 — and macro-F1 is the metric that actually penalizes ignoring rare defects.

Two broader trends stand out.

First, on **binary classification**, all models are competitive — the problem is coarse enough that transfer learning helps everyone.

Second, as **task difficulty increases**, the gap widens dramatically. On multi-11, our model leads the next-best ViT by 9.4 F1 points. On multi-12, the lead is 4.8 F1 points. This supports a clear conclusion: **domain-specific architecture and training recipe matter more than raw model capacity** when the task requires fine-grained discrimination of thermally similar defect patterns.

MobileNetV3 confirms the other side of the trade-off: it is the cheapest model at 2.41M parameters, but loses ten to twelve F1 points on multi-class tasks. Ultra-light backbones work for binary triage, but underperform when full defect typing is required.

---

### Slide 15 — Literature Comparison

The assignment specifically asked us to discuss our results against **Ramadan et al., 2024**, who report impressive headline numbers: **98.2% binary, 96.2% multi-11, and 95.6% multi-12 accuracy** using a roughly **86-million-parameter ViT** with aggressive oversampling.

We are not at those numbers — and we want to be honest about that. But the comparison is more nuanced than it first appears.

**Le et al., 2021** achieve 85.9% multi-12 accuracy with only **1.5 million parameters** by combining a small ResNet ensemble with SMOTE oversampling and Focal Loss — demonstrating that training recipe can matter as much as architecture size.

**Boutana et al., 2024** use the same ViT-B backbone as Ramadan but reach only about **81% multi-12 accuracy** — very close to our own head-tuned ViT at 77.45%. That is a crucial data point: **a large ViT is not automatically better**. Without Ramadan's specific preprocessing and oversampling pipeline, the transformer advantage largely disappears.

Our ProposedModel occupies a deliberate point in this design space: a **small, honestly evaluated, leak-free model** that achieves the **best macro-F1 in our comparison** while using **21 times fewer parameters than Ramadan's ViT**. We believe that is a meaningful result for the **edge and UAV deployment scenario** — the few percentage points we give up against the state of the art are traded for dramatically lower latency, power consumption, and memory footprint.

---

### Slide 16 — Limitations & Future Work

We should also be transparent about what our model still gets wrong.

The most persistent failure mode is **ambiguous minority pairs**: Hot-Spot versus Hot-Spot-Multi, and Cell versus Cell-Multi. These classes differ only in how many cells are affected, and from a single infrared frame that distinction is genuinely hard — Boutana et al. report the same kind of confusion on Cracking. Resolving this likely requires richer input: multi-frame temporal data, multi-modal sensing, or substantially more training examples per class.

We also observe **over-prediction of Offline-Module**, which we attribute to our aggressive inverse-frequency sampling weights. A softer scheme — such as square-root weighting or the effective-number-of-samples approach — should improve precision on medium-frequency classes.

Looking forward, the most promising directions build directly on what already works. **MixUp and CutMix** on minority classes could add two to four macro-F1 points with minimal code changes. **Knowledge distillation** from Ramadan's large ViT into our 4.12M model would let us keep the deployment budget while importing the teacher's finer decision boundaries. **Self-supervised pretraining on unlabelled IR frames** would give our CNN a domain-specific representation prior — the thermal equivalent of what ImageNet provides to our baselines. And **INT8 quantization** would make the model directly deployable on UAV-class hardware such as a Jetson Nano or Coral TPU.

---

### Slide 17 — Conclusion

To conclude: we set out to build automated PV defect classifiers that could operate without expert intervention in the field. We delivered **three models** spanning binary, eleven-class, and twelve-class classification, evaluated under a **strict, leak-free protocol** on the InfraredSolarModules dataset.

Our central finding is that a **4.12-million-parameter CBAM-CNN**, trained from scratch with Focal Loss, weighted sampling, and IR-aware augmentation, is the **best macro-F1 performer on both multi-class tasks** and the **best overall model on binary classification** — beating ViT, ResNet, VGG, and MobileNet while using a fraction of their parameters.

This is not an argument that small models always win. It is an argument that **on a small, biased, domain-specific dataset, the right training recipe and architecture can substitute for raw capacity** — and that trade-off is exactly what matters when the end goal is deployment on a drone, not winning a leaderboard with an 86-million-parameter transformer.

Thank you for your attention. We are happy to take your questions.

---

## Q&A — Suggested Backup Answers

Use these if asked during the question period:

**Why macro-F1 instead of accuracy?**  
On imbalanced data, accuracy rewards ignoring rare classes. Macro-F1 averages per-class F1 with equal weight, so a model that fails on Diode-Multi is penalized the same as one that fails on Cell. For inspection, every defect type matters.

**Why train ProposedModel from scratch instead of fine-tuning?**  
ImageNet features encode natural-image statistics — edges of cats and cars — not thermal gradients of solar cells. We wanted a controlled comparison: same data, same augmentation, same sampler — only the architecture differs. The multi-class results show domain-specific design wins when fine-grained IR discrimination is required.

**Why is Ramadan et al. so much higher?**  
Three main factors: a ~86M-parameter fully fine-tuned ViT (21× our size), aggressive minority oversampling, and potentially different evaluation protocols. Boutana et al. use the same ViT backbone without Ramadan's full pipeline and land much closer to our numbers — suggesting the recipe matters as much as the architecture.

**Could you deploy this today?**  
Binary triage is closest to production readiness (~92% F1, 4M params). Full twelve-class typing at 68% macro-F1 is useful as a decision-support tool but would need human review on ambiguous pairs. Quantization and distillation are the most direct paths to edge deployment.

**What would you do differently with more time?**  
Priority order: (1) knowledge distillation from a large ViT teacher, (2) MixUp/CutMix on minorities, (3) self-supervised IR pretraining, (4) multi-frame input for Hot-Spot vs. Hot-Spot-Multi disambiguation.

---

## Timing Checklist

| Section | Speaker | Slides | Target time |
|---------|---------|--------|-------------|
| Intro & motivation | Speaker 1 | 1–3 | ~2 min |
| Dataset & imbalance | Speaker 1 | 4–5 | ~2 min |
| Preprocessing & augmentation | Speaker 2 | 6–7 | ~2.5 min |
| Model & training | Speaker 2 | 8–10 | ~2.5 min |
| Results (3 tasks + table) | Speaker 3 | 11–14 | ~3.5 min |
| Literature & conclusion | Speaker 3 | 15–17 | ~3 min |
| **Total** | | **17 slides** | **~15–16 min** |

**Tip:** Rehearse transitions between speakers on slides 5→6 and 10→11. Keep an eye on the clock at slide 12 — if running long, shorten the literature slide and expand Q&A instead.
