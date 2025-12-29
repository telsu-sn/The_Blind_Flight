# Synapse Drive – Competition Solution

This repository contains the complete training and inference pipeline used for the Synapse Drive competition submission.

The solution is designed to be fully reproducible, robust to classification errors, and compatible with the official competition dataset structure.

---

## Repository Structure

synapse-drive-solution/
│
├── train/
│ ├── terrain_model.py # Terrain classification (desert / forest / lab)
│ ├── desert_tile_model.py # Desert tile classifier
│ ├── forest_tile_model.py # Forest tile classifier
│ ├── lab_tile_model.py # Lab tile classifier (balanced)
│
├── pathfinding/
│ ├── cost_grid.py # Cost grid construction
│ ├── start_goal.py # Safe start/goal detection
│ ├── dijkstra.py # Path planning algorithm
│
├── inference/
│ └── inference.py # Final submission pipeline
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## Approach Summary

The solution is a classical multi-stage pipeline consisting of:

1. Terrain classification using global image statistics
2. Terrain-specific tile classification using KNN models
3. Cost grid construction combining terrain costs and velocity boosts
4. Safe start and goal detection with fallback strategies
5. Path planning using Dijkstra’s algorithm

No deep learning models or external data were used.

---

## Training

All models are trained using the provided training dataset:

- Terrain classifier:
  - Features: global RGB mean and standard deviation
  - Model: K-Nearest Neighbors with feature scaling

- Tile classifiers:
  - Desert & Forest: RGB mean + standard deviation
  - Lab: RGB + grayscale statistics
  - Class imbalance handled via oversampling

Each terrain has its own independent tile classifier.

---

## Inference

The inference pipeline performs the following steps for each test image:

1. Load RGB image and velocity grid
2. Predict terrain type
3. Predict 20×20 tile class grid using the terrain-specific model
4. Build traversal cost grid
5. Detect start and goal safely
6. Run Dijkstra path planning
7. Apply fallback path if required
8. Save results to CSV

Progress logs and timing statistics are printed during inference.

---

## Reproducibility

- No randomness is required at inference time
- No internet access or pretrained models are used
- All logic is deterministic except for safe fallbacks

---

## Dependencies

Install required packages with:

pip install -r requirements.txt

yaml
Copy code

Required libraries:
- numpy
- pandas
- scikit-learn
- opencv-python
- matplotlib

---

## License

The author grants the Competition Sponsor the license to use, reproduce, and modify this submission as described in the competition rules.

The author confirms that no third-party proprietary data or models were used.

---

## Notes

This repository contains the **exact logic used in the final submission**, with only structural cleanup and comments added for clarity.
