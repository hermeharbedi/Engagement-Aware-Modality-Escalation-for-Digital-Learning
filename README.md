
# Engagement Aware Modality Escalation for Digital Learning

A multi-modal digital learning assistant that combines real-time engagement detection with intelligent content retrieval and dialogue. This project demonstrates how webcam-based student engagement sensing can be paired with an LLM-powered learning companion to dynamically support learners of sorting algorithms.

## Project Overview

This repository implements an engagement-aware digital learning system that:
- Detects student engagement in real time using webcam video and a YOLO-based model pipeline.
- Provides a Streamlit chatbot interface powered by Google Gemini (`gemini-2.0-flash`).
- Uses retrieval-augmented generation (RAG) for contextual answers from educational content.
- Supports text, image, and video learning modalities for algorithm education.
<p align="center"> <img width="683" height="235" alt="Screenshot 2026-05-29 at 11 18 39 PM" src="https://github.com/user-attachments/assets/bc84abc2-aefc-4833-b571-35b9d69b8fe2" /></p>

## Key Features

- **Real-time Engagement Monitoring**
  - Person detection with `YOLOv8`
  - Custom engagement classification via `best.pt`
  - Rolling engagement score visualization on the live feed

- **Contextual Learning Assistant**
  - Uses `SentenceTransformer` embeddings and FAISS for fast semantic retrieval
  - Retrieves relevant content chunks from PDF course material
  - Answers user questions with an LLM-guided assistant

- **Multi-modal Educational Support**
  - Built-in algorithm information dataset from `algorithm_info.csv`
  - Linked video resources via `video_links.csv`
  - Algorithm visuals in `sorting_images/`

- **Preprocessing Pipeline**
  - PDF text extraction and chunking with `preprocess.py`
  - Generates `chunks.pkl`, `embeddings.npy`, and `faiss.index`

## Repository Structure

- `Final_Code.py` — Main Streamlit interface with algorithm selection, content retrieval, and camera management
- `preprocess.py` — Preprocessing script for PDF text extraction, embedding generation, and FAISS indexing
- `algorithm_info.csv` — Algorithm metadata for curated learning content
- `sorting_images/` — Visualization assets for sorting algorithms
- `chunks.pkl` — Serialized content chunks for retrieval
- `embeddings.npy` — Saved sentence embeddings
- `faiss.index` — FAISS vector store index
- `best.pt` — Custom engagement classifier model
- `yolov8n.pt` — YOLO detection models

## Interact with the System

- Allow webcam access for engagement detection
- Ask questions in the chat panel about algorithm topics
- Monitor live engagement scores and modality escalation behavior
<p align="center"> <img width="809" height="172" alt="Screenshot 2026-05-29 at 11 17 07 PM" src="https://github.com/user-attachments/assets/b94512db-641e-46db-b670-9f16c541b204" /></p>


## Requirements

- Python 3.9+
- Webcam access for real-time engagement tracking
- `sorting_algorithms.pdf` present in the repository for preprocessing
- Google Generative AI API key in `.env`

## Study Design & Evaluation

This project also reflects a controlled user study comparing two instructional conditions:

- **Adaptive condition**: a dynamic interface escalates instructional modality from text to image to video when disengagement is detected.
- **Static condition**: a conventional text-only interface delivers identical educational content without modality changes.

### Learning task

- Participants studied seven sorting algorithms: Bubble, Selection, Insertion, Merge, Quick, Heap, and Radix Sort.
- Each module began with text explanations and steps, with images and videos introduced in the adaptive condition only.

### Engagement inference

- The system captures visual behavioral cues through a webcam and estimates learner engagement in real time.
- A lightweight YOLOv8n model was fine-tuned on student engagement data to trigger modality escalation.
- The detection pipeline is used to adapt instruction rather than to evaluate perception accuracy.

### Participants and measures

- 20 university students participated in the study (10 dynamic, 10 static).
- Measures included NASA-TLX cognitive workload dimensions, user experience questionnaire ratings, and qualitative feedback.

### Key findings

- Adaptive modality escalation reduced mental demand, effort, and frustration compared to static text-only delivery.
- Participants reported higher engagement, satisfaction, and concentration with the adaptive interface.
- Adaptive modality changes were generally perceived as helpful rather than distracting.
- Reported limitations focused on responsiveness and smoother disengagement detection.

## Contributions and Acknowledgement

- The project is a fruit of combined efforts put by Ms. Saumya Yadav (PhD, IIITD) and Mr. Hermehar P.S. Bedi (PhD, IIITD).
- The contributors would also like to acknowledge the efforts put in by Mr. Harsh Bhardaj (B.Tech, IIITD) and Mr. Dharani Kumar S. (B.Tech, IIITD) during the development phase of the initial version as the course project.
- This was started as a course project for Human Centered AI course offered by Dr. Rajiv Ratn Shah at IIIT Delhi.
