# notebooks/ — Notebook thăm dò

## Nguyên tắc

Notebook dùng để **thăm dò và trực quan hóa**, không phải nơi chứa logic
chính. Khi một đoạn code trong notebook đã ổn định, hãy chuyển nó vào
`src/xai_football/` rồi import ngược lại vào notebook.

Lý do: code nằm trong notebook thì không test được, không tái sử dụng được,
và diff trên git rất khó đọc.

## Cách import

Chạy một lần ở thư mục gốc dự án:

```bash
pip install -e .
```

Sau đó mọi notebook đều import được trực tiếp, không cần chỉnh `sys.path`:

```python
from xai_football.graphs import build
```

## Quy ước đặt tên

```
NN_mo-ta-ngan.ipynb
```

Ví dụ:

```
01_kham-pha-du-lieu-statsbomb.ipynb
02_thong-ke-mo-ta-possession.ipynb
03_kiem-tra-do-thi-possession.ipynb
04_phan-tich-ket-qua-xai.ipynb
```

## Trước khi commit

Xóa output của notebook (Kernel → Restart & Clear Output) để tránh làm phình
repo và gây xung đột khi merge.
