```markdown
# 🧠 Generative AI with TensorFlow - Learning Journey

A comprehensive, hands-on exploration of Generative AI built from first principles. This repository documents my journey from TensorFlow basics to building state-of-the-art generative models (VAEs, GANs, Transformers, and Music Generation).

## 📚 About This Project

This is not just another ML course repository - it's a **code-first learning journey** where every concept is built from scratch before using high-level APIs. The focus is on **deep understanding** rather than superficial knowledge.

**Course Context:** Final year B.Tech coursework in Generative AI (Chhattisgarh Swami Vivekananda Technical University)

**Learning Philosophy:** 70% Coding | 20% Theory | 10% Experimentation

---

## 🎯 Learning Objectives

By the end of this journey, I will be able to:
- Build Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) for image generation
- Implement LSTM and Transformer-based models for text generation
- Create music generation systems using MusicGAN and Transformer architectures
- Fine-tune pre-trained generative models for custom applications
- Understand the mathematics and intuition behind each architecture
- Deploy complete generative AI applications

---

## 🗂️ Repository Structure

```bash
generative-ai-tensorflow/
├── README.md                          # This file
├── pyproject.toml                     # Project dependencies (uv)
├── .python-version                    # Python version specification
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── phase-0-foundations/
│   │   ├── 01-tensor-basics.ipynb
│   │   ├── 02-neural-network-from-scratch.ipynb
│   │   ├── 03-keras-introduction.ipynb
│   │   └── 04-cnn-for-images.ipynb
│   ├── unit-1-intro-to-genai/
│   ├── unit-2-image-generation/
│   │   ├── 01-autoencoders.ipynb
│   │   ├── 02-vae-implementation.ipynb
│   │   ├── 03-vanilla-gan.ipynb
│   │   └── 04-dcgan.ipynb
│   ├── unit-3-text-generation/
│   │   ├── 01-text-preprocessing.ipynb
│   │   ├── 02-lstm-text-generation.ipynb
│   │   └── 03-transformer-implementation.ipynb
│   ├── unit-4-music-generation/
│   └── unit-5-advanced-topics/
├── src/                               # Reusable code modules
│   ├── models/                        # Model implementations
│   ├── utils/                         # Helper functions
│   └── data/                          # Data loading utilities
├── docs/                              # Learning notes and reflections
│   ├── phase-0-notes.md
│   ├── backpropagation-deep-dive.md
│   └── weekly-reflections/
├── projects/                          # Mini-projects and capstone
│   ├── mnist-from-scratch/
│   ├── image-generation-suite/
│   ├── text-generation-app/
│   └── music-composer/
├── resources/                         # Papers, references, datasets
│   ├── papers/
│   └── datasets/
└── outputs/                           # Generated images, text, music
    ├── images/
    ├── text/
    └── music/
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (modern Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd generative-ai-tensorflow
   ```

2. **Install uv** (if not already installed)
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create and activate virtual environment**
   ```bash
   # uv automatically creates a virtual environment
   uv venv
   
   # Activate it
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   
   # Or if using pyproject.toml:
   uv pip install -e .
   ```

5. **Install Jupyter (for notebooks)**
   ```bash
   uv pip install jupyter notebook ipykernel
   python -m ipykernel install --user --name=genai-env
   ```

6. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
   python -c "import tensorflow as tf; print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
   ```

### Optional: GPU Support

**For NVIDIA GPU (CUDA):**
```bash
# Install CUDA-enabled TensorFlow
uv pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**For Apple Silicon (M1/M2/M3):**
```bash
# TensorFlow Metal plugin (GPU acceleration on Mac)
uv pip install tensorflow-metal
```

---

## 📦 Dependencies

Core dependencies:
- `tensorflow>=2.15.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `jupyter>=1.0.0` - Interactive notebooks
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - ML utilities

Additional libraries (installed as needed):
- `pillow` - Image processing
- `librosa` - Audio processing
- `mido` - MIDI file handling
- `opencv-python` - Advanced image operations

See `requirements.txt` or `pyproject.toml` for complete list.

---

## 🎓 Learning Path

### Phase 0: TensorFlow Foundations (Weeks 1-2)
**Status:** ✅ In Progress (50% complete)

- [x] Tensor operations and manipulations
- [x] Building neural networks from scratch
- [x] Understanding backpropagation deeply
- [ ] Keras API mastery
- [ ] CNNs for image understanding
- [ ] Data pipelines with tf.data

**Key Achievement:** Built 2-layer neural network from scratch, achieved 95.76% accuracy on MNIST

### Unit I: Introduction to Generative AI (Week 3)
**Status:** 🔴 Not Started

- [ ] Discriminative vs Generative models
- [ ] Latent space representations
- [ ] Probability distributions and sampling
- [ ] Evaluation metrics for generative models

### Unit II: Image Generation (Weeks 4-7)
**Status:** 🔴 Not Started

- [ ] Autoencoders and Variational Autoencoders (VAEs)
- [ ] Generative Adversarial Networks (GANs)
- [ ] DCGAN for image generation
- [ ] Advanced techniques (Progressive GANs, StyleGAN concepts)
- [ ] Image-to-image translation

### Unit III: Text Generation (Weeks 8-11)
**Status:** 🔴 Not Started

- [ ] Text preprocessing and tokenization
- [ ] LSTM-based text generation
- [ ] Transformer architecture from scratch
- [ ] Fine-tuning language models
- [ ] Text generation applications

### Unit IV: Music Generation (Weeks 12-14)
**Status:** 🔴 Not Started

- [ ] Music representation (MIDI, spectrograms)
- [ ] LSTM-based music generation
- [ ] Music Transformer
- [ ] MuseGAN for multi-track generation
- [ ] Music composition applications

### Unit V: Advanced Topics (Weeks 15-16)
**Status:** 🔴 Not Started

- [ ] Transfer learning and fine-tuning
- [ ] Ethical considerations in Generative AI
- [ ] Emerging trends (Diffusion models, Multimodal models)

### Final Project (Weeks 17-18)
**Status:** 🔴 Not Started

Choose one comprehensive capstone project demonstrating mastery.

---

## 📊 Progress Tracker

**Overall Progress:** 15% Complete

| Phase | Progress | Status |
|-------|----------|--------|
| Phase 0: Foundations | ████░░░░░░ 40% | 🟡 In Progress |
| Unit I: Intro | ░░░░░░░░░░ 0% | 🔴 Not Started |
| Unit II: Images | ░░░░░░░░░░ 0% | 🔴 Not Started |
| Unit III: Text | ░░░░░░░░░░ 0% | 🔴 Not Started |
| Unit IV: Music | ░░░░░░░░░░ 0% | 🔴 Not Started |
| Unit V: Advanced | ░░░░░░░░░░ 0% | 🔴 Not Started |

**Time Invested:** 4.5 hours  
**Current Streak:** 2 days 🔥  
**Projects Completed:** 1/20+

---

## 💡 Key Learnings & Insights

### Week 1-2: TensorFlow Foundations
- **Aha Moment:** Backpropagation isn't magic - it's systematic application of the chain rule!
- **Technical Insight:** `tf.Variable` vs regular tensors - variables are mutable and trackable
- **Math Deep Dive:** Xavier initialization prevents gradient explosion/vanishing
- **Practical Skill:** Built neural network from scratch, understanding every line

**Detailed Notes:** See `docs/phase-0-notes.md` and `docs/backpropagation-deep-dive.md`

---

## 🛠️ Development Workflow

### Daily Workflow
```bash
# Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Start Jupyter
jupyter notebook

# Work on notebooks in notebooks/ directory
# Save reusable code to src/ directory
# Document learnings in docs/ directory
```

### Adding New Dependencies
```bash
# Add a new package
uv pip install package-name

# Update requirements
uv pip freeze > requirements.txt
```

### Running Scripts
```bash
# Run a standalone script
python src/models/vae.py

# Run with module imports
python -m src.models.vae
```

---

## 📖 Resources

### Primary Textbooks
1. "Generative AI with Python and TensorFlow 2" - Joseph Babcock & Raghav Bali (2024)
2. "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
3. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron (2023)
4. "Generative Deep Learning" - David Foster (2023)

### Online Resources
- [TensorFlow Official Documentation](https://www.tensorflow.org/tutorials)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Papers with Code](https://paperswithcode.com/)
- [ArXiv ML Papers](https://arxiv.org/list/cs.LG/recent)

### Key Papers
- **GANs:** "Generative Adversarial Networks" - Goodfellow et al. (2014)
- **VAEs:** "Auto-Encoding Variational Bayes" - Kingma & Welling (2013)
- **Transformers:** "Attention is All You Need" - Vaswani et al. (2017)
- **DCGAN:** "Unsupervised Representation Learning with DCGANs" - Radford et al. (2015)

---

## 🎯 Success Metrics

### Technical Competency
- [ ] Build VAEs, GANs, LSTMs, and Transformers from scratch
- [ ] Generate realistic images, coherent text, and musical compositions
- [ ] Fine-tune pre-trained models for custom tasks
- [ ] Debug and optimize complex model architectures
- [ ] Deploy generative AI applications

### Theoretical Understanding
- [ ] Explain the mathematics behind each architecture
- [ ] Understand training dynamics and failure modes
- [ ] Critically evaluate generated content
- [ ] Articulate trade-offs between different approaches

### Practical Skills
- [ ] Read and implement research papers
- [ ] Design experiments and track results
- [ ] Optimize hyperparameters systematically
- [ ] Document code and insights clearly

---

## 🤝 Contributing

This is a personal learning repository, but feel free to:
- Open issues for questions or discussions
- Suggest resources or improvements
- Share your own learning journey

---

## 📝 License

This project is for educational purposes. Code implementations may reference or build upon existing tutorials and papers (properly cited).

---

## 🙏 Acknowledgments

- **Instructor/Mentor:** Claude (Anthropic) for guided learning
- **University:** Chhattisgarh Swami Vivekananda Technical University
- **Community:** TensorFlow, Keras, and open-source ML community

---

## 📧 Contact

**Student:** [Your Name]  
**Program:** B.Tech (Final Year) - Data Sciences  
**University:** CSVTU, Bhilai (C.G.)

**Repository:** [Your GitHub URL]  
**Email:** [Your Email]

---

## 📅 Timeline

**Start Date:** October 2025  
**Target Completion:** February 2026  
**Weekly Commitment:** 7-8 hours

**Last Updated:** October 2025

---

## 🔥 Current Focus

**This Week:** Phase 0 - TensorFlow Foundations
- Completed: Neural network from scratch (95.76% accuracy on MNIST)
- Next: Keras introduction and CNNs for image understanding
- Goal: Finish Phase 0 by end of Week 2

**Next Milestone:** Unit I - Understanding Generative AI paradigm (Week 3)

---

**"The best way to learn is to build from first principles."**

Happy Learning! 🚀
