# Literature Review Notes

Use this document to summarize key findings, methodologies, and takeaways from literature relating to counter-UAS systems, RF analysis, acoustic detection, and vision models.

---

## Template for Reviewing Papers

### [Year] - [Paper Title or Citation]
* **Authors**: 
* **Venue**: (e.g., IEEE TIFS, Elsevier Signal Processing)
* **Link/DOI**: 
* **Core Modality**: (RF / Acoustic / Vision / Fusion)

#### Key Contributions & Methodology
* Brief summary of what this paper proposes.
* What dataset was used? (e.g., DroneRF, custom data)
* What is the network structure/preprocessing (e.g., STFT, wavelet)?

#### Main Results
* Accuracy / F1 scores.
* Computational footprint or inference speed mentioned.

#### Takeaways for Vardan
* How does this guide our baseline selection or fusion strategy?
* Any limitations mentioned that Vardan can address?

---

## Active Reading List & Notes

### [Example] 2020 - DroneRF: Deep learning based UAV detection
* **Authors**: M. Ezuma, et al.
* **Venue**: IEEE Sensors Journal
* **Core Modality**: RF
* **Key Contributions**: Collected and published the DroneRF dataset. Used RF signature analysis with deep learning models to identify drone presence and flight mode.
* **Takeaway**: Essential baseline reference.
