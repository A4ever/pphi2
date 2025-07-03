#!/bin/bash

# تنظیمات
REPO_DIR="."
JSON_DIR="json"
SCRIPTS_DIR="scripts"
CONFIG_DIR="config"
MODELS_DIR="models"
DATA_DIR="training_data"

# ایجاد ساختار دایرکتوری‌ها
echo "making dirctorys..."
mkdir -p $SCRIPTS_DIR $CONFIG_DIR $MODELS_DIR

# کپی فایل‌های JSON از پوشه مجاور
echo "copying json files..."
if [ -d "../$DATA_DIR" ]; then
    cp -r "../$DATA_DIR" .
else
    echo "error: does'nt find training_data directory"
    exit 1
fi

# ایجاد اسکریپت نصب Phi-2
cat > $SCRIPTS_DIR/install-phi2.sh << 'EOL'
#!/bin/bash

# تغییر به دایرکتوری اصلی پروژه
cd "$(dirname "$0")/.."

# نصب پیش‌نیازها
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git git-lfs

# ایجاد محیط مجازی
python3 -m venv venv
source venv/bin/activate

# نصب کتابخانه‌های پایتون
pip install torch transformers accelerate peft trl \
python-telegram-bot datasets sentencepiece protobuf

# دانلود مدل Phi-2
git lfs install
git clone https://huggingface.co/microsoft/phi-2 models/microsoft/phi-2

echo "Phi-2 installed completly<div></div>"
EOL

# ایجاد اسکریپت فاین‌تیونینگ
cat > $SCRIPTS_DIR/tuning-phi2.sh << 'EOL'
#!/bin/bash

# تغییر به دایرکتوری اصلی پروژه
cd "$(dirname "$0")/.."

# فعال‌سازی محیط مجازی
source venv/bin/activate

# اجرای فاین‌تیونینگ
python finetune.py

echo "fine tunning completed, find at models/fine-tuned"
EOL

# ایجاد اسکریپت اجرای ربات
cat > $SCRIPTS_DIR/bot.sh << 'EOL'
#!/bin/bash

# تغییر به دایرکتوری اصلی پروژه
cd "$(dirname "$0")/.."

# فعال‌سازی محیط مجازی
source venv/bin/activate

# اجرای ربات
python bot.py
EOL

# ایجاد فایل کانفیگ تلگرام
cat > $CONFIG_DIR/telegram_config.env << 'EOL'
# تنظیمات ربات تلگرام
TELEGRAM_TOKEN="YOUR_BOT_TOKEN_HERE"
MODEL_PATH="models/fine-tuned"
MAX_RESPONSE_LENGTH=500
LOG_FILE="bot.log"
EOL

# ایجاد فایل فاین‌تیونینگ
cat > finetune.py << 'EOL'
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

# تنظیمات
BASE_MODEL_PATH = "models/microsoft/phi-2"
TRAINING_DATA_PATH = "training_data/*.json"
OUTPUT_DIR = "models/fine-tuned"

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# بارگیری مدل و توکنایزر
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = get_peft_model(model, LORA_CONFIG)

# بارگیری داده‌ها
dataset = load_dataset("json", data_files=TRAINING_DATA_PATH, split="train")

# توکنایز کردن
def tokenize_function(examples):
    combined_text = [p + " " + r for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(
        combined_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# پارامترهای آموزش
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

# آموزش
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("starting tunning")
trainer.train()

# ذخیره مدل
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("tuuing completed")
EOL

# ایجاد فایل ربات
cat > bot.py << 'EOL'
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from dotenv import load_dotenv

# تنظیمات لاگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# بارگیری تنظیمات
load_dotenv("config/telegram_config.env")
TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")
MAX_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", 500))
LOG_FILE = os.getenv("LOG_FILE")

# بارگیری مدل
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# دستورات ربات
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! من یک روانشناس هوش مصنوعی هستم. هر سوال یا مشکلی دارید بپرسید."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    
    # تولید پاسخ
    inputs = tokenizer(
        f"درمانگر: {user_input}\nدستیار:",
        return_tensors="pt",
        return_attention_mask=True,
        max_length=256,
        truncation=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_LENGTH,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    await update.message.reply_text(response)

def main():
    app = Application.builder().token(TOKEN).build()
    
    # دستورات
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logging.info("robot started")
    app.run_polling()

if __name__ == "__main__":
    main()
EOL

# دادن مجوز اجرا به اسکریپت‌ها
chmod +x $SCRIPTS_DIR/install-phi2.sh
chmod +x $SCRIPTS_DIR/tuning-phi2.sh
chmod +x $SCRIPTS_DIR/bot.sh

echo "install completed, next steps:"
echo "1.put your telegram token bot and setting on config/telegram_config.env"
echo "2. install model: $SCRIPTS_DIR/install-phi2.sh"
echo "3. praktice model: $SCRIPTS_DIR/tuning-phi2.sh"
echo "4. run the bot: $SCRIPTS_DIR/bot.sh"