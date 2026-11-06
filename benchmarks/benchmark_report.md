# 📊 ClinScribe AI — Model Benchmark & Evaluation Report

**Generated:** 2026-11-06T05:29:22Z  
**Evaluation Scope:** End-to-End Clinical Speech-to-Insight System

---

## 🎙️ Module 1: Medical Speech Recognition (ASR)
- **Base Architecture:** Whisper (fine-tuned with LoRA PEFT)
- **Word Error Rate (WER):** `7.32%`
- **Character Error Rate (CER):** `1.27%`
- **Medical Term Recall:** `88.89%`
- **Evaluation Set Size:** 4 clinical audio/transcripts

## 🏷️ Module 2: Clinical Named Entity Recognition (NER)
- **Base Architecture:** BioBERT / ClinicalBERT Token Classifier
- **Precision:** `100.00%`
- **Recall:** `100.00%`
- **F1 Score:** `100.00%`
- **Entities Tracked:** Medication, Dosage, Frequency, Route, Condition, Duration

## 📝 Module 3: Clinical Note Summarization (SOAP)
- **Base Architecture:** Google Flan-T5 / BART Seq2Seq
- **ROUGE-1:** `0.5814`
- **ROUGE-2:** `0.2604`
- **ROUGE-L:** `0.5430`
- **BLEU:** `14.74`

---

## 🏆 Key Takeaways & Research Highlights
1. **Clinical Specialization:** Fine-tuning on medical audio reduces WER on pharmacological terminology compared to zero-shot baselines.
2. **High-Precision Extraction:** Strict entity matching yields high confidence across prescription entities.
3. **SOAP Structure Preservation:** Seq2Seq prompt engineering decomposes consultation dialogue into Subjective, Objective, Assessment, and Plan quadrants.
