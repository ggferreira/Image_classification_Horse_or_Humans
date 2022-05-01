import streamlit as st
from img_classification import image_classification
from PIL import Image

st.title('Classificação de Imagem')
st.subheader("Reconhece a imagem de um Cavalo ou de um Ser Humano")
st.text("Insira a imagem de um Cavalo ou Ser Humano")


uploaded_file = st.file_uploader("Escolha uma foto", type="jpg")
st.caption("Obs: a imagem tem que ser no formato .jpg")
if uploaded_file is None:
    st.caption("Você ainda não inseriu nem uma imagem")

else:
    image = Image.open(uploaded_file)
    st.image(image, caption='Foto escolhida', use_column_width=True)
    st.write("")
    st.write("Classificando...")
    label = image_classification(image, 'horse_or_human_model.h5')
    if label == 0:
        st.write("Provavelmente é a imagem de um Cavalo")
    else :
        st.write("Provavelmente é a imagem de um Ser Humano")
