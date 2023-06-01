import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import torch
from torchvision import transforms
from cnn_classifier import Classifier

# Detecção da face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Erro ao iniciar detecção de face.")

categorias = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'}  # Dicionário com as categorias e seus índices correspondentes

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceMask(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.label_colors = {
            'Com Máscara': (0, 255, 0),    # Verde
            'Sem Máscara': (255, 0, 0),    # Vermelho
            'Uso Incorreto': (0, 0, 255)   # Azul
        }

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pre_trained_weights = torch.load('mask_detection_model_stdict2.pt')  # Carregar pesos do modelo
        classificador = Classifier()
        classificador.load_state_dict(pre_trained_weights)
        classificador.eval()

        faces = face_cascade.detectMultiScale(
            image=img, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
            if np.sum([roi]) != 0:
                # Pré-processar a ROI e converter para tensor
                roi = roi.astype('float') / 255.0
                roi = transforms.ToTensor()(roi).unsqueeze(0)
                # Pega a ROI pré processada e passa para o classificador
                with torch.no_grad():
                    prediction = classificador(roi)
                    maxindex = int(torch.argmax(prediction))
                    finalout = categorias[maxindex]
                    output = str(finalout)
            label_position = (x, y)
            label_color = self.label_colors[output]  # Obter a cor do rótulo com base no output
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        return img
    

def home():
    st.title('Boas vindas ao Face Mask Detection')
    st.subheader("Uma aplicação para detecção de máscara facial em tempo real feita com OpenCV, PyTorch e Streamlit.")
    st.write(
        '''
            No menu a esquerda você irá encontrar as principais funcionalidades dessa aplicação.
            1. Detecção do uso de máscara facial em tempo real usando webcam.
            2. Classificação do uso de mascara facial através de imagens enviadas pelo usuário.
        '''
    )


def realtime_classification():
    st.header("Classificação em Tempo Real")
    st.write("Clique em start para usar iniciar a webcam e detectar se você está usando máscara ou não")
    webrtc_streamer(key="example", video_transformer_factory=FaceMask)


def image_classification():
    # Carrega do modelo pré treinado
    pre_trained_weights = torch.load('mask_detection_model_stdict2.pt')  
    classificador = Classifier()
    classificador.load_state_dict(pre_trained_weights)
    classificador.eval()

    st.header("Classificação por Imagem")
    st.write("Faça o upload de uma foto e receba a classificação de máscara facial.")

    # Cria um campo de upload de arquivo
    uploaded_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

    # Faz o pré processamento da imagem
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem enviada', use_column_width=True)
        image = image.resize((224, 224), resample=Image.BILINEAR)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)

        with torch.no_grad():
            # Pega a imagem pré processada e passa para o classificador
            prediction = classificador(image_tensor)
            maxindex = int(torch.argmax(prediction))
            finalout = categorias[maxindex]
            output = str(finalout)

        # Exibe o resultado da classificação
        st.subheader("Resultado da classificação")
        st.write(f"A imagem é classificada como: {output}")


def about():
    st.header("Sobre esse projeto")
    st.write(
        '''
            Essa é uma aplicação utiliza um modelo CNN treinado com PyTorch e uma interface Web feita com OpenCV e Streamlit.

            Desenvolvido por:
            - Amon Menezes Negreiros
            - Henrique Barkett
            - Pedro Carvalho Almeida
            - Wander Araújo Buraslan
        '''
    )


pages = {
    'Home': home,
    'Classificação em Tempo Real': realtime_classification,
    'Classificação por Imagem': image_classification,
    'Sobre': about,
}

page = st.sidebar.selectbox('Escolha uma página', pages.keys())

if page:
    pages[page]()
