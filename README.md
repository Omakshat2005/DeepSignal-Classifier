# DeepSignal-Classifier
Real-Time RF Modulation Intelligence with Stacked LSTMs
📌 Executive Summary
DeepSignal-Classifier is a signal intelligence (SIGINT) platform that automates the identification of 11 different modulation types from raw I/Q data. By bridging the gap between Radio Frequency (RF) Engineering and Deep Learning, this project replaces traditional, manual feature extraction with a high-performance Stacked LSTM architecture capable of classifying signals even in high-noise environments (-18 dB SNR).

🎯 The Engineering ChallengeIn real-world telecommunications, signals are rarely "clean." They are distorted by atmospheric noise, hardware imperfections, and phase offsets.Problem: Traditional classifiers require manual feature engineering and often fail when the Signal-to-Noise Ratio (SNR) drops.The Solution: An end-to-end pipeline that consumes raw complex-valued temporal data $(128x2)$ and extracts features using the temporal memory of Long Short-Term Memory (LSTM) cells.

🏗️ Technical ArchitectureThe system utilizes a specialized Stacked LSTM design to process temporal phase and amplitude 
variations:Input Layer: Raw I/Q signal vectors $(128 \times 2)$.LSTM Layer 1: 64 units with Batch Normalization to normalize the signal distribution.LSTM Layer 2: 64 units providing hierarchical temporal feature extraction.Dense Classifier: 128-neuron ReLU layer followed by a Softmax output for 11-class probability distribution.

📊 Performance & Insights
Accuracy: Achieved a ~47.6% global accuracy across the full SNR range (-18 dB to +20 dB). Note: This is highly competitive for the RadioML 2016.10A dataset, where low SNR samples are nearly indistinguishable from noise.
Hardware Optimization: Successfully optimized the training pipeline to run on a MacBook, managing thermal constraints and 280% CPU spikes through strategic batching (1024).
Technical Hurdles Overcome:
Phase Rotation: Identified and documented that BPSK signals frequently misclassify due to phase offsets (slanted constellations).
Learning Stability: Implemented ReduceLROnPlateau and EarlyStopping to prevent overfitting and ensure model convergence.

🛠️ Tech Stack
Deep Learning: TensorFlow / Keras (Sequential API).
Data Processing: NumPy, Pandas, Scikit-learn (for data splitting and normalization).=
Visualization: Matplotlib (Constellation Diagrams, PSD, and Waveforms).
Interface: Streamlit (Custom CSS for a warm-toned, professional dashboard).


Developed by Omakshat Sisodia
Electronics & Communication Engineering Student | GEC Surat
