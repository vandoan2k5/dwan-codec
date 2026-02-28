

# (lumina) /kaggle/dwan-audiocodec# python ./data/libritts.py
import os
from datasets import load_dataset, Audio
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from tqdm import tqdm

# --- Cấu hình ---
DATASET_NAME = "mythicinfinity/libritts_r"
SAVE_DIR = "./data/"
TARGET_SAMPLING_RATE = 24000
LIMIT = 100000
MAX_WORKERS = 4  # Tăng số luồng để tận dụng băng thông mạng

os.makedirs(os.path.join(SAVE_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
def process_and_save(item):
    example, index = item
    try:
        if(index < 1000):
            file_path = os.path.join(SAVE_DIR, "val", f"libri_{index:06d}.wav")
        else:
            file_path = os.path.join(SAVE_DIR, "train", f"libri_{index:06d}.wav")

        
        # Nếu file đã tồn tại thì bỏ qua (tiện khi bị crash chạy lại)
        if os.path.exists(file_path):
            return True
            
        audio_data = example["audio"]["array"]
        sf.write(file_path, audio_data, TARGET_SAMPLING_RATE)
        return True
    except Exception:
        return False

def main():
    print(f"🚀 Đang kết nối dataset (Streaming mode)...")
    dataset = dataset = load_dataset("mythicinfinity/libritts_r", "all", split="train.other.500", streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    
    # Tạo generator để không bị load hết vào RAM
    def data_generator():
        for i, ex in enumerate(dataset):
            if i >= LIMIT:
                break
            yield (ex, i)

    print(f"🔥 Bắt đầu tải và xử lý song song...")
    # Dùng ThreadPoolExecutor để xử lý I/O Bound (tải mạng + ghi đĩa)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdm bọc quanh map để theo dõi tiến độ thời gian thực
        list(tqdm(executor.map(process_and_save, data_generator()), total=LIMIT))

if __name__ == "__main__":
    main()