import streamlit as st
import os


#load saved model
predictor_model = load_model(r'resnext50_32x4d_fold4_best.pth')

st.header("Cassava Leaf Diseases Detection")

#save uploaded images
def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    

    except:
        return 0

#create Upload
uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file): 
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        prediction = predictor(os.path.join('static/images',uploaded_file.name))
        os.remove('static/images/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction
        # drawing graphs
        st.text('Predictions :-')
        fig, ax = plt.subplots()
        ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)
        ax.set(xlabel='Confidence %', ylabel='Breed')
        st.pyplot(fig)
        
