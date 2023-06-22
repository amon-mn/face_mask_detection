import cv2
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import torch
import numpy as np
from torchvision import transforms
from cnn_classifier5 import Classifier5
import face_recognition 



# Classificação do uso de Máscara
pre_trained_weights = torch.load('mask_detection_model5_weights.pt', map_location=torch.device('cpu'))
classificador = Classifier5()
classificador.load_state_dict(pre_trained_weights)
classificador.eval()

# Dicionário com as categorias e seus índices correspondentes
categorias = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceMask(VideoTransformerBase):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carregar o modelo CNN para a classificação da máscara.
        pre_trained_weights = torch.load('mask_detection_model5_weights.pt', map_location=torch.device('cpu'))
        self.classificador = Classifier5()
        self.classificador.load_state_dict(pre_trained_weights)
        self.classificador.eval()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detectar rostos usando o RetinaFace.
        face_locations = face_recognition.face_locations(img, model="retina")

        for (top, right, bottom, left) in face_locations:
            face_img = img[top:bottom, left:right]

            # Pré-processamento da imagem do rosto para o modelo CNN.
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Converter para RGB.
            face_img = transforms.ToPILImage()(face_img)
            face_img = transforms.Resize((224, 224))(face_img)
            face_img = transforms.ToTensor()(face_img)
            face_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face_img)
            face_img = face_img.unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.classificador(face_img)
                maxindex = int(torch.argmax(prediction))
                finalout = categorias[maxindex]
                output = str(finalout)

            # Desenhar retângulo ao redor do rosto.
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img, output, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

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
    webrtc_streamer(
        key="example",
        video_transformer_factory= FaceMask
    )
    


def image_classification():
    st.header("Classificação por Imagem")

    # Cria um campo de upload de arquivo.
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    # Faz o pré processamento da imagem.
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Converter a imagem para OpenCV para detecção de rostos.
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detectar rostos na imagem usando RetinaFace.
        face_locations = face_recognition.face_locations(img_cv, model="retina")

        padding = 20  # Valor para aumentar a área do rosto
        face_locations = [(top - padding, right + padding, bottom + padding, left - padding)
                            for (top, right, bottom, left) in face_locations]

        if len(face_locations) == 0:
            # Nenhum rosto foi detectado.
            # Fazer o pré-processamento da imagem original para a classificação.
            img_pil = image.resize((224, 224), resample=Image.BILINEAR)
            img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)

            with torch.no_grad():
                # Passar a imagem original pré-processada para o classificador.
                prediction = classificador(img_tensor)
                maxindex = int(torch.argmax(prediction))
                finalout = categorias[maxindex]
                output = str(finalout)

            # Exibir a imagem original e o resultado da classificação.
            st.subheader("Imagem original")
            st.image(image, use_column_width=True)
            st.subheader("Resultado da classificação")
            st.write(f"A imagem é classificada como: {output}")
        else:
            # Pelo menos um rosto foi detectado.
            # Desenhar o retângulo em cada rosto detectado.
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(img_cv, (left, top), (right, bottom), (255, 0, 0), 2)

                # Fazer o pré-processamento da imagem para a classificação.
                face_image = image.crop((left, top, right, bottom))
                face_image = face_image.resize((224, 224), resample=Image.BILINEAR)
                face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)

                with torch.no_grad():
                    # Passar a imagem do rosto pré-processada para o classificador.
                    prediction = classificador(face_tensor)
                    maxindex = int(torch.argmax(prediction))
                    finalout = categorias[maxindex]
                    output = str(finalout)

                # Adicionar o rótulo embaixo do retângulo.
                cv2.putText(img_cv, output, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            # Converter a imagem de volta para PIL para exibição no Streamlit.
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            # Exibir as imagens lado a lado.
            st.subheader("Imagem original e Imagem com detecção facial")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True)
            with col2:
                st.image(img_pil, use_column_width=True)

    else:
        st.info("Faça o upload de uma imagem para começar.")    




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