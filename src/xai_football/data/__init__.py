"""Bước 1-2, 4 — Thu thập, tiền xử lý và chia tập dữ liệu.

Module:
    collect.py       Tải dữ liệu từ StatsBomb Open Data (competitions/matches/
                     events/lineups), lưu JSON thô theo
                     data/raw/{competition_id}/{season_id}/{match_id}.json
    canonicalize.py  Chuyển raw JSON thành canonical event representation,
                     persist vào data/interim/events.parquet
    possession.py    Trích xuất possession sequence từ event data (Bước 2.1)
    clean.py         Làm sạch: loại possession quá ngắn, hiệp phụ, thiếu tọa độ
                     (Bước 2.2); chuẩn hóa hệ tọa độ sân 120x80 (Bước 2.4)
    labeling.py      Gán nhãn nhị phân possession -> sút/bàn thắng (Bước 2.3)
                     XEM docs/decisions/0001-quy-tac-gan-nhan.md — CHƯA CHỐT
    validation.py    Data quality / sanity checks (PASS/WARNING/FAIL)
    splits.py        Chia train/val/test theo TRẬN, stratified, cố định seed
                     (Bước 4) — chống rò rỉ dữ liệu
"""
