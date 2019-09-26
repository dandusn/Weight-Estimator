# Weight-Estimator
Berikut adalah repository untuk tugas akhir yang disusun oleh Dandu Satyanuraga. Repository ini berisi tentang pemanfaatan machine learning berbasis foto untuk mengukur berat hewan sapi dan domba.

## Dalam penelitian ini menggunakan
- Mask R-CNN
- Regresi Linier & Regresi KNN

## Mask R-CNN menggunakan
- Arsitektur Resnet 50 / Resnet 101
- Arsitektur lain dapat digunakan namun dalam default menggunakan restnet
- Model MRCNN (https://github.com/matterport/Mask_RCNN)

## Regresi
- menggunakan K-Means untuk klustering

## Cara penggunaan
Perintah dalam cmd atau terminal
1. EXPORT FLASK_APP = app.py
2. EXPORT FLASK_ENV = (ENV Anda)
3. python -m flask run

## Hal yang dibutuhkan
1. Foto Hewan (Sapi atau Domba)
2. Umur dari hewan

## Author
1. Dr. Ir. rila Madala, M.Eng
2. Nugraha Priya Utama, S.T, M. A, Ph.D
2. Dandu Satyanuraga (dandu.nuraga@gmail.com)
