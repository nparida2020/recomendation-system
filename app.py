# Creating a FashionRecommend class to include all features 

import numpy as np
import pandas as pd
import os 
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from PIL import Image  
import joblib 

class FashionRecommend:
    """ Production class for recommendations of fashion from similarity """
    
    def __init__(self, image_path, df_emb, styles_csv_path):
        self.img_path = image_path
        self.df_embeddings = df_emb
        self.styles_path = styles_csv_path
    # Helper functions 
    def get_styles_dataframe(self):
        """ Load a dataframe contains styles details and images """
        styles_df = pd.read_csv(self.styles_path) 
        styles_df['image'] = styles_df.apply(lambda x: str(x['id']) + ".jpg", axis=1) # (id.jpg)
        return styles_df
    
    def get_model(self):
        
        # Pre-Trained ResNet50 Model
        base_model_ResNet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        base_model_ResNet50.trainable = False
        # Add Layer Embedding
        model = keras.Sequential([
            base_model_ResNet50,
            GlobalMaxPooling2D()
        ])
        
        return model

    def predict(self, model, img_path):
        """ Load and preprocess image then make prediction """
        # Reshape
        img = image.load_img(self.img_path, target_size=(100, 100)) # đoạn này có thể chuyển thành tải ảnh theo link về
        # img to Array
        img = image.img_to_array(img)
        # Expand Dim (1, w, h)
        img = np.expand_dims(img, axis=0)
        # Pre process Input
        img = preprocess_input(img)
        return model.predict(img)

    def get_sample_similarity(self):
        """ Get cosine similarity of custom image """
        model = self.get_model()
        df_embeddings = self.df_embeddings
        sample_image = self.predict(model, self.img_path)
        df_sample_image = pd.DataFrame(sample_image).reset_index(drop=True)
        sample_similarity = linear_kernel(df_sample_image, df_embeddings)
        return sample_similarity
    
    def normalize_similarity(self):
        """ Normalize similarity results-min/max method """
        cosine_similarity = self.get_sample_similarity()
        x_min = cosine_similarity.min(axis=1)
        x_max = cosine_similarity.max(axis=1)
        norm = (cosine_similarity-x_min)/(x_max-x_min)[:, np.newaxis]
        return norm
    
    def get_recommendations_dataframe(self):
        """ Get recommended images """
        normalized_similarity = self.normalize_similarity()
        df = self.get_styles_dataframe()
        # pairwsie similarity scores of all clothes with respect to one (index, value)
        sim_scores = list(enumerate(normalized_similarity[0]))

        # Sort based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get 0:5 similar clothes
        sim_scores = sim_scores[0:5]
        print(sim_scores)
        # Get the Apparel indices
        indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar products
        return df['image'].iloc[indices]
    
    def show_recommendations(self,output_dir ='output_dir'):
        """ Print the top 5 most similar products"""
        rec_frame = self.get_recommendations_dataframe()
        rec_list = rec_frame.to_list()
        os.makedirs(output_dir, exist_ok=True)
        for idx, image_name in enumerate(rec_list):
            imgraw = mpimg.imread("./images/" + image_name)
            plt.imshow(imgraw)
            plt.axis("off")
            plt.title("Recommended Image")
            plt.savefig(os.path.join(output_dir, f'recommended_image_{idx}.png'))
            plt.close()
            

def load_and_preprocess_image(image):
    """ Preprocess the uploaded image """
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("AI-FASHION Recommendation System")
    st.write("Upload an image !")
    img_file = st.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img = load_and_preprocess_image(img_file)
        if img is not None:
            df_embeddings = joblib.load('df_embeddings.joblib')
            styles_path = './styles.csv'
            obj = FashionRecommend(img_path=img_file, df_embeddings=df_embeddings, styles_path=styles_path)
            obj.show_recommendations()
            st.write("Recommended Images...")
            for i in range(5):
                st.image(f'output_images/recommended_image_{i}.png')

if __name__ == "__main__":
    main()