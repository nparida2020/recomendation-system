
Project Objective:
Fashion Recommendation System Overview
In today’s fast-paced digital shopping landscape, a Fashion Recommendation System plays a crucial role in enhancing user experience by delivering personalized and relevant product suggestions. By analyzing user preferences, seasonal trends, and contextual factors like gender, category, and usage, such a system helps customers quickly discover items that align with their tastes and needs. Our system filters the vast product catalog using intelligent segmentation, making fashion browsing more intuitive and efficient. Below are specific filter-based segments that help refine recommendations for different customer profiles and fashion contexts:

1. Summer Ethnic Kurtas for Women
This filter caters to women seeking ethnic fashion options during the summer season. Specifically focusing on kurtas under the apparel category, this selection is ideal for those who prefer traditional or cultural attire in warm weather. These garments are typically made of breathable fabrics like cotton and are designed to offer both comfort and elegance. Perfect for casual daywear, small gatherings, or festive occasions, this collection ensures women can stay stylish and culturally expressive while beating the summer heat.

<img width="267" alt="Image" src="https://github.com/user-attachments/assets/76bf1605-d12b-4bdd-abd0-8292ec964157" />

2. Fall Casual T-shirts for Women
Targeting a more relaxed, everyday style, this filter highlights casual t-shirts for women that are well-suited for the fall season. As the weather begins to cool, these items often come in slightly thicker fabrics and warmer tones. The selection focuses on comfort without compromising on style, making it ideal for students, professionals, or anyone looking for easy-to-wear options during transitional weather. These pieces often serve as wardrobe staples that can be paired with jeans, jackets, or layered with other fall essentials.

<img width="355" alt="Image" src="https://github.com/user-attachments/assets/d4950642-e0a5-45b4-a7ec-5843ef5e13ea" />


3. Fall Formal Shirts for Men
This filter is designed for men looking for formal apparel options during the fall. By narrowing the selection to formal shirts, it focuses on garments typically worn in office settings, business meetings, or formal events. These shirts are tailored for a polished look, often coming in solid colors, subtle patterns, or classic designs that complement the professional fall wardrobe. Made from materials suitable for slightly cooler temperatures, this collection balances style, functionality, and season-appropriate comfort.

<img width="375" alt="Image" src="https://github.com/user-attachments/assets/2b396b8a-6204-41ce-9deb-4b825e61484c" />

5. Summer Casual Wear for Men
This segment focuses on casual apparel for men during the summer season, prioritizing comfort and laid-back style. Whether it's lightweight t-shirts, shorts, or relaxed-fit shirts, this category is ideal for everyday use, vacations, or outdoor activities. Breathable materials like cotton and linen are common, ensuring that the clothing helps in staying cool while maintaining a modern and effortless aesthetic. This filter serves those who want to look stylish without sacrificing comfort during hot weather.

<img width="378" alt="Image" src="https://github.com/user-attachments/assets/4c838230-0833-4094-9495-d6199e7ed8bf" />


Model Architecture:

<img width="598" alt="Image" src="https://github.com/user-attachments/assets/1ed2ce00-7e5b-47bb-93ad-2a2e60296577" />

The following layers involed in data architecture:

Data Preprocessing Layer:
There are two inputs , one is styles.csv file and other one is image directory which contains images of the items. The datas are loaded indivusually and combined in to one frame. This stage cleans and the product metadata (season, gender, category, article type, usage), and image or textual descriptions. The text/image data can be embedded using pretrained models using ResNet.

Feature Extraction Layer
The input data are segmented by applyting 4 filters of different features. For image-based features, a ResNet50 model extracts visual features. Textual descriptions are converted into vector embeddings using models .
Below are the 4 feature extracted using filters in the input dataframe using Gender,season,articleType and Usage in the product data.

df_summer_woman_ethinic = df[(df["season"] == 'Summer') & (df["gender"] == 'Women') & (df["masterCategory"] == 'Apparel')  & (df['articleType']=='Kurtas') & (df['usage'] =='Ethnic') ]
df_fall_woman_casual = df[(df["season"] == 'Fall') & (df["gender"] == 'Women') & (df["masterCategory"] == 'Apparel')  & (df['articleType']=='Tshirts') & (df['usage'] =='Casual') ]
df_fall_man_formal = df[(df["season"] == 'Fall') & (df["gender"] == 'Men') & (df["masterCategory"] == 'Apparel')  & (df['articleType']=='Shirts') & (df['usage'] =='Formal') ]
df_summer_man_casual = df[(df["season"] == 'Summer') & (df["gender"] == 'Men') & (df["masterCategory"] == 'Apparel')  & (df['articleType']=='Tshirts') & (df['usage'] =='Casual') ]

Distance Matrix:
Distance matrix computed using embedded vectors and the image in intreset.

Recommendation Engine
Cosine similarites algorthim used for recomending similar items for the target item.

Ranking & Filtering Layer
Final candidate items are scored and top 5 ranked based on relevance. This layer ensures that the recommendations are not only personalized but context-aware 

Model Evaluation and Feedback Loop
The system is continuously improved using A/B testing, evaluation metrics such as precision and recall .New data is periodically fed back into the system to retrain and update models.

Model Artifacts:
Embbeding and modes are stored in local folders and loaded when required for predication.
Mode stored in the file ResNet50_model.h5.
Embeddings are stored for 4 features in the below files:
df_summer_woman_ethinic_emb.joblib
df_summer_man_casual_emb.joblib
df_fall_woman_casual_emb.joblib
df_fall_man_formal_emb.joblib

Model And Application Deployment:
I have created a class file named 'FashionRecommend' which includes all the api for the model creation, embbiding input images and styles, computes similarity (cosine) and distance matrix and recomend top 5 samples of the image under selection. 
User needs to create an "images" folder in the current directory to store all the images input to the system ,output_images for stroring recommended images. Also copy styles.csv input file and model atrifacts file in the current directory. 
application , app.py uses FashionRecommend class instanes to recommend the predications. 
Before running app.py user needs to install python environment dependent packages :---->$> pip install -r requirements.txt
To run the web server, simply execute streamlit with the main recommender app:
    
    $>streamlit run main.py

Conclusions:

The segmentation of fashion data using filtered DataFrames has proven to be a highly effective approach for building a targeted and personalized recommendation system. By categorizing products based on key attributes such as season, gender, category, article type, and usage, we are able to deliver precise recommendations tailored to specific user preferences and contexts. This not only enhances the user shopping experience but also increases engagement and conversion rates. The defined segments—ranging from summer ethnic kurtas for women to fall formal shirts for men—demonstrate the power of data-driven filtering in curating relevant fashion collections. Moving forward, these filtered insights can be integrated into a larger recommendation engine, enriched with user interaction data and deep learning models, such as ResNet50 for visual features, to offer even more refined and personalized suggestions.

Github link of the project:

https://github.com/nparida2020/recommendation-system

