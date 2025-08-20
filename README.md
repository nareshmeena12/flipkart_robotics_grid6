# Flipkart Grid 6.0 – Robotics Track

## 📌 Project Overview
This repository contains my work for the **Flipkart Grid 6.0 Robotics Track**, where I developed a **deep learning–based product recognition system**.  

The goal was to design a model that could **accurately detect and classify retail products** in real-world warehouse and e-commerce scenarios, enabling faster and more reliable automation in supply chain logistics.

---

## 🎯 Objectives
- Build a **CNN-based product recognition model** to classify images of retail items.
- Achieve **high accuracy and robustness** under varying lighting and background conditions.
- Integrate recognition into a pipeline that could potentially be deployed on **warehouse robots**.

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Cleaned and augmented the dataset using rotation, scaling, and Gaussian noise.  
   - Normalized images for stable training.  

2. **Model Architecture**  
   - Designed a **Convolutional Neural Network (CNN)** optimized for product classification.  
   - Used multiple convolutional + pooling layers followed by dense layers.  

3. **Training**  
   - Loss Function: Categorical Cross-Entropy  
   - Optimizer: Adam  


4. **Evaluation**  
   - Tested robustness on unseen images.  
   - Conducted error analysis to improve classification on visually similar products.  

---

## 📊 Results
- Reached **Level 2 of Flipkart Grid 6.0 Robotics Track** with this model.  
- Demonstrated strong performance in real-world recognition scenarios.  
- Proved applicability in **robotic warehouse automation**.  

---

## 📂 Implementation
The repository contains:
- Data preprocessing scripts  
- CNN model definition and training loop  
- Evaluation + visualization scripts  

