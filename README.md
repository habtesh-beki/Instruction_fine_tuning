# GPT-2 Instruction Fine-Tuning (PyTorch + HuggingFace)

This repository contains a simple and clean implementation of **instruction fine-tuning GPT‑2** using an open-source instruction dataset. The project walks through every major step — dataset preparation, tokenization, padding, dataloader construction, and a full training loop (with optimizer, scheduler, and evaluation).

The goal of this project is to help beginners understand how GPT-2 fine‑tuning works while keeping the code readable and ready for real experimentation.

---

## 🚀 Features

- Instruction-style dataset support (Q/A or instruction → response)
- Custom dataset + dataloader (with padding)
- GPT‑2 fine-tuning using HuggingFace **Transformers**
- PyTorch training loop (AdamW + LR Scheduler)
- 3‑epoch fine‑tuning by default
- Easy to switch between **GPU** and **CPU**
- Simple evaluation after training

---

## 📦 Requirements

Make sure you have Python 3.9+.

Install the required libraries:

```bash
pip install torch torchvision torchaudio
pip install transformers
```

If you want to run on CPU only (no GPU required), PyTorch CPU version works fine.

---

## 📥 Clone the Repository

```bash
git clone https://github.com/habtesh-beki/Instruction_fine_tuning.git
cd gpt2-instruction-finetune
```

---

## 📚 Dataset

This project uses an **open-source instruction dataset**. The dataset is divided into training and validation splits.

You can replace it with your own dataset by putting your JSON/text files inside the `data/` folder.

---

## ⚙️ Configuration

If your machine does **not** have a GPU, simply set the device in `training.py`:

```python
device = "cpu"
```

If you have a GPU:

```python
device = "cuda"
```

The training loop automatically adapts.

---

## ▶️ Run Training

Once everything is installed, start fine‑tuning with:

```bash
python training.py
```

This will:

- Load GPT‑2
- Load the instruction dataset
- Tokenize + pad
- Run 3 epochs of training
- Save checkpoints

During training, you will see batch loss and training progress.

---

## 🧪 Testing the Model

After training, you can test the model in `training.py`.

inside the training.py you can change the prompt then

## You will see how your fine‑tuned GPT‑2 responds based on the training.

## 📁 Project Structure

```
📦 gpt2-instruction-finetune
├── download_data.py
├── training.py
├── data_prepare.py
├── dataset_dataloader.py
├── train.jsonl
├── test.jsonl
├── val.jsonl
├── dataset_loader.py
├── utils.py
└── README.md
```

---

## ⭐ Future Improvements

- Add mixed precision training (FP16)
- Add TensorBoard logging
- Integrate LoRA for faster training
- Add dataset auto‑download script

---

## 🤝 Contributing

Pull requests are welcome! If you want to add features or improve training performance, feel free to open an issue.

---

## 📜 License

MIT License — free for personal and commercial use.

---

### If you found this project helpful, consider giving the repo a ⭐ on GitHub!
