#import libraries
import torch
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
from albumentations import (Compose, Normalize, Resize)
import matplotlib.pyplot as plt

#calculate time
start_time = time.time()

#predictor function
def predictor(img):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, 5)
    FILE = "resnext50_32x4d_fold4_best.pth"
    device = torch.device('cpu')
    loaded_model = model
    loaded_model.load_state_dict(torch.load(FILE, map_location=device), strict=False)
    loaded_model.to(device)
    loaded_model.eval()

    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    transform1 = Compose([
           Resize(512, 512),
           Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
           ),
           ToTensorV2(),
       ])
    img = transform1(image=img)['image']
    img = img.unsqueeze(0)
    print(img)
    
    result = loaded_model(img)
    # print(result)
    
    disease_label = torch.argmax(result, dim=1)
    # print(disease_label)
    
    disease_name = "Cassava Bacterial Blight (CBB)" if disease_label == 0 else "Cassava Brown Streak Virus Disease (CBSD)" if disease_label == 1 else "Cassava Green Mottle (CGM)" if disease_label == 2 else "Cassava Mosaic Disease (CMD)" if disease_label == 3 else "Healthy"
    # print(disease_name)
    
    output = []
    output.append(result)
    output.append(disease_label)
    output.append(disease_name)
    print(output)
    return output

#frontend start here
st.title("Cassava Leaf Diseases Classification")
st.write("The main aim of this web application is to predict the types of cassava leaf disease accurately by using the images of the cassava crops.")
st.write("")

#save uploaded images
def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join(uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    

    except:
        return 0

#create Upload
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file): 
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        prediction = predictor(display_image)

        st.write("")
        st.write("")
        st.write("")
        st.write("Predicted Condition: ", prediction[2])
        st.write("Time taken: %.3f seconds" % (time.time() - start_time))
        st.snow()
        # delete uploaded saved picture after prediction
        os.remove(uploaded_file.name)

#explain the diseases
st.write("")
st.write("")
st.write("")
st.header("Types of Cassava Leaf Diseases:")

#CBB
with st.expander("Cassava Bacterial Blight (CBB)", expanded=False):
    CBBimg = Image.open('Cassava Bacterial Blight (CBB).jpg')
    st.image(CBBimg, caption='Cassava Bacterial Blight (CBB) (Image Source: Lucid Central)')    
    st.write("Description & Symptoms:")
    st.write("""  
        Cassava Bacterial Blight (CBB) is caused by the bacterium called Xanthomonas axonopodis pv. Manihotis (X. axonopodis) which was firstly discovered in Brazil in 1912. 
        Infections of CBB are most visible at the end of the rainy season and the start of the dry season. 
        The presence of aberrant leaf shedding and drying of leaves that are still attached to the stem should be the primary symptoms of the disease. 
        The parts of leaves that have infected will coalesce and the side of the leaves started having burnt appearance. 
        The spots will enlarge to the neighbouring spots and merge together to become a large brown spot which will affect the leaves to perform photosynthesis process. 
        Furthermore, this disease will discolour the mature stems of the infected plants in the severe phase.
    """)
    st.write("")
    st.write("Disease Control Strategies:")
    st.write("""  
        1. Crop Rotation
        2. Weed Removal 
        Crop rotation can be applied for CBB prevention because X. axonopodis will grow easily in a piece of land with poor quality soil and transmitted with its spores to infect the healthy cassava crops. 
        Before planting cassava again, soil turnover is required, and a six-month timeframe is used to observe the presence of the X. axonopodis bacteria in the soil. 
        This timeframe will be able to clean the crop fields of inoculum. 
        In addition, the weeds in the cassava fields will be required to be cleared out as the spores of the X. axonopodis can stay alive on the weeds.
    """)
     
#CBSD
with st.expander("Cassava Brown Streak Virus Disease (CBSD)", expanded=False):
    CBSDimg = Image.open('Cassava Brown Streak Virus Disease (CBSD).jpg')
    st.image(CBSDimg, caption='Cassava Brown Streak Virus Disease (CBSD) (Image Source: Lucid Central)')  
    st.write("Description & Symptoms:")
    st.write("""  
        Cassava Brown Streak Virus Disease (CBSD) is given the term brown streak because of the brown lesions that occasionally emerge on young green stems. 
        Although stem lesions were the first indications of the illness to be identified, they are not the most common symptom of infection and only occur seldom. 
        They are characterised by a distinctive yellow or necrotic vein banding that can expand and consolidate into huge yellow areas. 
        Tuberous root symptoms include dark-brown necrotic patches within the tuber and a loss in root size and the lesions in roots can cause crop deterioration after harvest.
    """)
    st.write("")
    st.write("Disease Control Strategies:")
    st.write("""  
        1. Control Usage of Insecticides
        2. Biological Pest Control Method
        The spreading vector of CBSD is the whitefly (Bemisia Tabaci). 
        Therefore, the cassava farmers should control the usage of the insecticides such as pyrethroids and organophosphates to prevent the developments of resistant populations of whiteflies. 
        Furthermore, over usage of insecticides will kill the natural predators of whiteflies such as the hoverflies and ground beetles which will lead to the increment of whiteflies population in the cassava fields. 
        Thus, the farmers can try biological pest control method by having natural predators of whiteflies such as the hoverflies and ground beetles in the cassava fields to kill the whiteflies instead of utilising vast amount of insecticides to kill the whiteflies. 
        With the biological pest control method, the cassava farmers are advised to control the usage of insecticides in their cassava fields. 
    """)

#CGM
with st.expander("Cassava Green Mottle (CGM)", expanded=False):
    CGMimg = Image.open('Cassava Green Mottle (CGM).jpg')
    st.image(CGMimg, caption='Cassava Green Mottle (CGM) (Image Source: Lucid Central)')    
    st.write("Description & Symptoms:")
    st.write("""  
        Cassava green mottle (CGM) is another ordinary cassava leaf disease which is caused by cassava green mottle nepovirus from the Secoviridae family. 
        Young leaves have curled edges and faint to obvious yellow dots with green mosaics. 
        The cassava shoots may recover from the symptoms and look to be in good condition. 
        However, the cassava crops will become severely stunted in the growth, and edible roots are non-existent or, if present, tiny and woody when cooked. 
        Cassava crops that contain these symptoms should be removed and burned as soon as possible. 
        The farmers must not hesitate until cultivation season to treat the cassava crops since the cassava crops may have healed from the symptoms and will be very difficult for the farmers to recognise the infected crops.
    """)
    st.write("")
    st.write("Disease Control Strategies:")
    st.write("""  
        1. Infected Crops Removal
        2. Cultural Control
        Cassava crops that contain symptoms of CGM should be removed and burned as soon as possible. 
        The farmers must not hesitate until cultivation season to treat the cassava crops since the cassava crops may have healed from the symptoms and will be very difficult for the farmers to recognise the infected crops.
        Furthermore, cultural control of the farmers is vital for cultivating cassava crops that are free from CGM. The farmers should only take the cuttings for planting from cassava crops that are symptom-free during harvest. 
        The farmers should not take the cassava cuttings from the infected fields to plant in their fields. 
        This is due to the cuttings from the infected fields may be contaminated by the CGM virus.
    """)          

#CMD
with st.expander("Cassava Mosaic Disease (CMD)", expanded=False):
    CMDimg = Image.open('Cassava Mosaic Disease (CMD).jpg')
    st.image(CMDimg, caption='Cassava Mosaic Disease (CMD) (Image Source: CIAT | The International Center for Tropical Agriculture)')  
    st.write("Description & Symptoms:")
    st.write("""  
        Cassava mosaic disease (CMD) is transmitted by the whitefly (Bemisia Tabaci) that can limit the output of the cassava crops in the Sub-Saharan Africa. 
        Mosaic, mottling, malformed leaves, twisted leaflets, and an overall decrease in the size of leaves are all foliar signs of CMD. 
        Leaf chlorosis can be a pale yellow or practically white colour with a tint of green. 
        The chlorotic patches are often noticeable and the range in size from a complete leaflet to minute specks or dots. 
        Leaflets may have a homogeneous mosaic design or a mosaic pattern that is limited to a few regions which are usually at the bottoms. 
        These leaves that have undergone chlorosis will be unable to perform photosynthesis which will eventually cause the cassava crops to become withered. 
        If this situation becomes severe, the large farming area of cassava crops will wither which will cause the harvest will be badly affected. 
    """)
    st.write("")
    st.write("Disease Control Strategies:")
    st.write("""  
        1. Intercropping with Legumes 
        2. Uprooting Infected Crops
        Cassava farmers can practice intercropping with legumes to fight against CMD in the farming area. 
        Legumes that are recommended to intercrop with cassava are cowpeas, groundnuts, and green gram as these legumes will not compete with cassava crops for the nutrients and water very much in the same fields. 
        Intercropping cassava crops with legumes can reduce the population of the whiteflies significantly due to the legumes are natural repellents for whiteflies. 
        Besides, the cassava crops that have infected CMD should be uprooted within a week after the occurrences of the CMD symptoms. 
        The uprooted infected cassava crops should be taken out of the field and dried under the sun before being burnt to destroy the viruses. 
        The contaminated brunt debris should be thrown into dustbin instead of buried in the field to prevent the outbreak of CMD viruses again.
    """)
st.write("")
st.write("")
st.write("")

#model performance plots
st.header("Model Performance:")
df1 = pd.read_csv("log1.csv")
# print(df1)
df2 = pd.read_csv("log2.csv")
# print(df2)
df3 = pd.read_csv("log3.csv")
# print(df3)
df4 = pd.read_csv("log4.csv")
# print(df4)

#Graph of Accuracy against Epoch - Graph 1
with st.expander("Graph of Accuracy against Epoch", expanded=False):
    plot1_1, = plt.plot(df1['epoch'], df1['accuracy'])
    plot2_1, = plt.plot(df2['epoch'], df2['accuracy'])
    plot3_1, = plt.plot(df3['epoch'], df3['accuracy'])
    plot4_1, = plt.plot(df4['epoch'], df4['accuracy'])

    plot1_1 = plt.title("Graph of Accuracy against Epoch")
    plot1_1 = plt.xlabel('Epoch') 
    plot1_1 = plt.ylabel('Accuracy') 

    plot1_1 = plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4'])

    plot1_1 = plt.show()
    plot2_1 = plt.show()
    plot3_1 = plt.show()
    plot4_1 = plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot1_1)

st.write("")
#Graph of Average Train Loss against Epoch - Graph 2
with st.expander("Graph of Average Train Loss against Epoch", expanded=False):
    plot1_2, = plt.plot(df1['epoch'], df1['avg_train_loss'])
    plot2_2, = plt.plot(df2['epoch'], df2['avg_train_loss'])
    plot3_2, = plt.plot(df3['epoch'], df3['avg_train_loss'])
    plot4_2, = plt.plot(df4['epoch'], df4['avg_train_loss'])

    plot1_2 = plt.title("Graph of Average Train Loss against Epoch")
    plot1_2 = plt.xlabel('Epoch') 
    plot1_2 = plt.ylabel('Average Train Loss') 

    plot1_2 = plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4'])

    plot1_2 = plt.show()
    plot2_2 = plt.show()
    plot3_2 = plt.show()
    plot4_2 = plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot1_2)

st.write("")
#Graph of Average Validation Loss against Epoch - Graph 3
with st.expander("Graph of Average Validation Loss against Epoch", expanded=False):
    plot1_3, = plt.plot(df1['epoch'], df1['avg_val_loss'])
    plot2_3, = plt.plot(df2['epoch'], df2['avg_val_loss'])
    plot3_3, = plt.plot(df3['epoch'], df3['avg_val_loss'])
    plot4_3, = plt.plot(df4['epoch'], df4['avg_val_loss'])

    plot1_3 = plt.title("Graph of Average Validation Loss against Epoch")
    plot1_3 = plt.xlabel('Epoch') 
    plot1_3 = plt.ylabel('Average Validation Loss') 

    plot1_3 = plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4'])

    plot1_3 = plt.show()
    plot2_3 = plt.show()
    plot3_3 = plt.show()
    plot4_3 = plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot1_3)
    
st.write("")
#Graph of Best Score against Epoch - Graph 4
with st.expander("Graph of Best Score against Epoch", expanded=False):
    plot1_4, = plt.plot(df1['epoch'], df1['best_score'])
    plot2_4, = plt.plot(df2['epoch'], df2['best_score'])
    plot3_4, = plt.plot(df3['epoch'], df3['best_score'])
    plot4_4, = plt.plot(df4['epoch'], df4['best_score'])

    plot1_4 = plt.title("Graph of Best Score against Epoch")
    plot1_4 = plt.xlabel('Epoch') 
    plot1_4 = plt.ylabel('Best Score') 

    plot1_4 = plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4'])

    plot1_4 = plt.show()
    plot2_4 = plt.show()
    plot3_4 = plt.show()
    plot4_4 = plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot1_4)

st.write("")
#Graph of Average Validation Loss against Average Train Loss - Graph 5
with st.expander("Graph of Average Validation Loss against Average Train Loss", expanded=False):
    plot1_5, = plt.plot(df1['avg_train_loss'], df1['avg_val_loss'])
    plot2_5, = plt.plot(df2['avg_train_loss'], df2['avg_val_loss'])
    plot3_5, = plt.plot(df3['avg_train_loss'], df3['avg_val_loss'])
    plot4_5, = plt.plot(df4['avg_train_loss'], df4['avg_val_loss'])

    plot1_5 = plt.title("Graph of Average Validation Loss against Average Train Loss")
    plot1_5 = plt.xlabel('Average Train Loss') 
    plot1_5 = plt.ylabel('Average Validation Loss') 

    plot1_5 = plt.legend(['Fold1', 'Fold2', 'Fold3', 'Fold4'])

    plot1_5 = plt.show()
    plot2_5 = plt.show()
    plot3_5 = plt.show()
    plot4_5 = plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot1_5)
