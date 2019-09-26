# Weight-Estimator
Berikut adalah repository untuk tugas akhir yang disusun oleh Dandu Satyanuraga. Repository ini berisi tentang pemanfaatan machine learning dan metode regresi untuk mengukur berat hewan sapi dan domba.

## Dalam penelitian ini menggunakan
- Mask R-CNN
- Regresi Linier & Regresi KNN

## Mask R-CNN menggunakan
- Arsitektur default Resnet 50 / Resnet 101
- Arsitektur lain dapat digunakan namun dalam default menggunakan Restnet 50 (template)
- [Model MRCNN](https://github.com/matterport/Mask_RCNN)

## Regresi
- menggunakan regresi linier untuk pembanding
- menggunakan K-Means untuk klustering
- Menggunakan regresi KNN

##Dataset Gambar (sapi dan Domba)
Data set saat ini menggunakan data yang ada dalam Engine/data untuk menambah data dapat mendownload dari link dibawah:
- [Sapi](https://doc-0g-a4-docs.googleusercontent.com/docs/securesc/dhg99fnpvn3ar29f3tfkhpud1dooq5jt/rm3clsgamb4nv2t1530bu2onepumcv19/1569463200000/04338857337762483249/05089704764381139844/0B7dS7AFpUzt1b3N6Q0FIcEpWWmc?e=download)
- [Domba](https://doc-04-a4-docs.googleusercontent.com/docs/securesc/dhg99fnpvn3ar29f3tfkhpud1dooq5jt/bhjggfig5dcrkkd2pvmjvlc7qe9ou1va/1569463200000/04338857337762483249/05089704764381139844/0B7dS7AFpUzt1cGNkT1lVMldxdHc?e=download&nonce=pnjh8o6728j0q&user=05089704764381139844&hash=vp9kb9kf0q463cda2u669lrfhn8n1lat)

##Arsitektur Sistem
![arch](asset/arch.png)

##Struktur Komponen
![komp](asset/graph.png)

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

Nb* untuk data video akan segera diupload karena file tergolong besar