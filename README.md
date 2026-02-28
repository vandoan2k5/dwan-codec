# DwanCodec 

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Language](https://img.shields.io/badge/Language-Vietnamese-blue.svg?style=for-the-badge)

DwanCodec là một hệ thống nén và tái tạo âm thanh (Audio Compression) hiệu quả dựa trên kiến trúc **VQ-VAE** kết hợp với mô hình **GAN (Generative Adversarial Network)**. Mô hình tập trung vào việc chuyển đổi tín hiệu âm thanh thô thành các mã số nén (discrete codes) và giải mã ngược lại với chất lượng cao nhờ vào bộ giải mã Vocos và kỹ thuật ISTFT.

## 1. Giới thiệu sơ lược về mô hình

Hệ thống bao gồm các thành phần chính sau:

* **Encoder**: Sử dụng các khối ConvNeXt để trích xuất đặc trưng từ waveform đầu vào, hỗ trợ giảm tần số lấy mẫu (downsampling) thông qua các bước stride tùy chỉnh.
* **Quantizer**: Hỗ trợ nhiều cơ chế lượng hóa khác nhau như **FSQ (Finite Scalar Quantization)**, LFQ (Lookup Free Quantization), và VQ truyền thống. Mặc định dự án sử dụng FSQ để tránh các vấn đề về codebook collapse.
* **Decoder**: Dựa trên kiến trúc Vocos kết hợp với **ISTFT Head** để tái tạo âm thanh từ không gian tiềm ẩn một cách trung thực nhất.
* **Discriminator**: Hệ thống sử dụng Multi-Resolution Discriminator để đánh giá độ chi tiết của âm thanh ở nhiều thang đo khác nhau trong quá trình huấn luyện GAN.

## 2. Cài đặt môi trường

Dự án yêu cầu Python >= 3.12. Bạn có thể cài đặt các thư viện cần thiết thông qua tập tin `pyproject.toml` hoặc sử dụng `pip`:

```bash
pip install torch torchaudio einops wandb datasets soundfile torchinfo

```

*(Lưu ý: Các phiên bản thư viện yêu cầu tối thiểu được liệt kê trong `pyproject.toml`).*

## 3. Hướng dẫn huấn luyện (Training)

Quá trình huấn luyện sử dụng kết hợp nhiều loại hàm mất mát (Loss functions) bao gồm: L1 Waveform Loss, Multi-Resolution STFT Loss, Codebook Loss, Adversarial GAN Loss và Feature Matching Loss.

### Chuẩn bị dữ liệu

Bạn cần chuẩn bị thư mục chứa các file âm thanh `.wav` cho hai tập `train` và `val`.

### Chạy lệnh huấn luyện

Sử dụng file `model_train.py` để bắt đầu quá trình train. Dưới đây là ví dụ về lệnh chạy cơ bản:

```bash
python model_train.py \
    --train-data-path "./data/train" \
    --val-data-path "./data/val" \
    --sample-rate 24000 \
    --batch-size 32 \
    --lr 1e-4 \
    --quantizer fsq \
    --experiment-name "Dwan_Codec_v1"

```

### Các tham số quan trọng:

* `--quantizer`: Chọn loại quantizer (`fsq`, `lfq`, `ema`, `origin`).
* `--levels`: Cấu hình các mức cho FSQ (mặc định tạo ra khoảng 16k codes).
* `--lr`: Tốc độ học của Generator.
* `--lr-disc`: Tốc độ học của Discriminator (thường cao hơn Generator).
* `--save`: Thư mục lưu trữ checkpoint (mặc định tạo tự động trong `.checkpoints/`).

Hệ thống có tích hợp **Weights & Biases (WandB)** để theo dõi các chỉ số loss và tốc độ học theo thời gian thực.

## 4. Sử dụng mô hình (Inference)

Dự án cung cấp lớp wrapper `DwanCodec` trong `models/models.py` để dễ dàng tích hợp vào các ứng dụng khác:

```python
from models.models import DwanCodec
from argparse import Namespace

# Cấu hình config tương ứng với lúc train
config = Namespace(in_channel=1, channel=512, ...) 
checkpoint_path = "path/to/your/model.pt"

# Khởi tạo codec
codec = DwanCodec(config, checkpoint_path, sample_rate=24000)

# Mã hóa audio thành codes
codes = codec.encode("input_audio.wav")

# Giải mã codes về lại waveform
wav_recon, sr = codec.decode(codes)

```

## 5. Cấu trúc thư mục chính

* `models/`: Chứa định nghĩa về Encoder, Decoder, VQ-VAE và Discriminators.
* `quantizers/`: Các module lượng hóa (FSQ, LFQ, VQ).
* `utils/`: Các công cụ hỗ trợ về loss, dataset, argument và scheduler.
* `model_train.py`: Script chính để thực hiện huấn luyện.

## 6. Một số tham khảo và hỗ trợ

* **Gemini**
* [duchenzhuang fsq training source ](https://github.com/duchenzhuang/FSQ-pytorch/tree/main)
* [fsq paper](https://arxiv.org/abs/2309.15505)
