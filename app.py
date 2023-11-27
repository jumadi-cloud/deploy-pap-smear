import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu

from proses import classify, set_background

st.set_page_config(
    page_title="papsmear - Demo",
    page_icon="🤖"
)

# Navigasi
with st.sidebar:
        selected = option_menu(
             menu_title="Pilih", 
             options=["Klasifikasi Cervical Cancer", "About"], 
             icons=['house', "list-task"], 
             menu_icon="cast", 
             default_index=0,
             )

if selected == "Klasifikasi Cervical Cancer":
     set_background('./doc/bg1.png')
     # set title
     st.title('Classification Of Cervical Cancer')
     # set header
     st.subheader('Please upload a picture of cervical cancer')
     # upload file
     file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
     # load classifier
     model = load_model('./model/keras_model.h5')
     # load class names
     with open('./model/labels.txt', 'r') as f:
           class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
           f.close()
           # display image
           if file is not None:
                 image = Image.open(file).convert('RGB')
                 st.image(image, use_column_width=True)
                 class_name, conf_score = classify(image, model, class_names)
                 st.write("## {}".format(class_name))
                 st.write("### Prediksi Score: {}%".format(int(conf_score * 1000) / 10))

if selected == "About":
         st.markdown("""
                     # 🤖 RepoMedUNM
        ### Hallo Sahabat Kelas Awan Pintar 👋

        Data set citra Pap smear dibutuhkan dalam identifikasi dini cervical cancer. 
        Saat ini identifikasi dini berbasis pengolahan citra terus dikembangkan. 
        Sejalan pelaksanaan tes Pap smear konvensional sebagai upaya menanggulangi kematian akibat cervical cancer pada wanita usia produktif.
        Kebutuhan data set citra Pap smear sangat penting karena pada penelitian dan proses medical image analisis membutuhkan data tranning dan data tes seiring berkembangnya algoritma baru. 
        Saat ini peneliti di Indonesia melakukan riset dengan data set privat dan terbatas.
        Belum tersedia repository data set khusus citra Pap smear yang dapat dimanfaatkan bersama.

        Untuk lebih detail bisa dilihat disini **[RepoMedUNM](http://repomed.nusamandiri.ac.id/)**.
        """, True)
         st.write("---")

         st.markdown("""
            ### 📑Informasi Data
            Database RepoMedUNM terdiri dari 7437 citra sel berkelompok. 
                    
            Citra-citra diperoleh melalui mikroskop optik OLYMPUS CX33RTFS2 dan mikroskop X52-107BN dengan kamera Logitech.    
            
            Untuk lebih detail bisa dilihat disini **[RepoMedUNM](http://repomed.nusamandiri.ac.id/)**.
        """, True)

         st.write("---")

         st.markdown("""
            ### 📑Kategori Sel
            
            **Sel Normal** merupakan sebagian besar sel yang ditemukan dalam tes Pap. Biasanya berbentuk pipih dengan bentuk sitoplasma bulat, oval atau poligonal, sebagian besar.
            
            **L-Sil** merupakan sel abnormal ringan atau sedang yang mencerminkan sebuah infeksi sementara dengan HPV (i.e. perinuclear halo, or koilocites).
            
            **H-Sil** merupakan Atipiae selular yang sering dikaitkan dengan infeksi persisten HPV dengan risiko tinggi berkembang menjadi kanker dimana dysplasia parah, anisositosis, kromatin, bergranulasi atau nucleus yang lebih besar ukurannya.

            **Sel Koilocyt** sering berkorespondensi dalam skuamosa sel dewasa (intermediate and superficial) dan kadang-kadang berkorespondensi pada jenis sel koilocyt metaplastic. Koilocyt sering terlihat cyanophilic, dimana sedikit bernoda dengan ditandai rongga preinuklear yang besar. Tepi sitoplasma bernoda sangat padat. Koilocyt biasanya membesar dan terletak eksentrik, hiperkromatik serta menunjukkan ketidak teraturan pada membaran nukelus. Pada beberapa kasus dalam koilocyt terdapat sel yang memiliki nucleus nya dua atau bahkan banyak. Sel koilocyt merupakan patognomonik dalam infeksi HPV dimana koilocyt biasanya menunjukan derajat degenerasi yang bergantung pada berbagai virus yang berbeda.

            Untuk lebih detail bisa dilihat disini **[RepoMedUNM](http://repomed.nusamandiri.ac.id/)**.    
        """, True)