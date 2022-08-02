
# Imports Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
import sklearn
from PIL import Image

# Sets Page Configuration
st.set_page_config(page_title = 'Drift Capacity Web App')

# Cache Feature for Loading ML Model
with open('xgb_model.pkl','rb') as f:
    xgb_model = pickle.load(f)

# Cache Feature for Loading Excel Spredsheet
@st.experimental_memo
def load_excel(spreadsheet):
    nlp = pd.read_excel(spreadsheet)
    return (nlp)

# Loads Sidebar Tab 
st.sidebar.header("Navigation")
selectbox1 = st.sidebar.radio("Select a page",("Introduction", "Description of Inputs", "Model Use & Performance"))
st.sidebar.info("Read the Full Paper at: [ResearchGate](https://www.researchgate.net/publication/359148969_Explainable_Machine_Learning_Model_for_Predicting_Drift_Capacity_of_Reinforced_Concrete_Walls)")

# Loads Introduction Page 
if selectbox1 == "Introduction":
    st.title("Explainable Machine Learning Model For Predicting The Drift Capacity Of Reinforced Concrete Walls")
    st.markdown("This app estimates the drift capacity, which is the drift at which lateral strength degrades by 20% from the peak strength, of reinforced concrete walls with special boundary elements. It is based on the Extreme Gradient Boosting (XGBoost) machine learning algorithm and uses three input parameters.")
    image1 = Image.open('homepage.png')
    st.image(image1)
    st.markdown("")
    st.markdown("This app also compares the results of this machine learning model to a pre-existing empirical model. The equation that this model consists of was adopted in ACI 318-19 and developed by Abdullah and Wallace (2019) to also predict the mean drift capacity ($\dfrac{\delta_{x}}{h_{w}}$) of walls with SBEs. The equation uses the same three input parameters and is shown below.")
    st.markdown("$\dfrac{\delta_{x}}{h_{w}}$" + "(_%_)" + "$=3.85-\dfrac{\lambda_{b}}{α} - \dfrac{v_{max}}{10\sqrt{f'_{c}(psi)}}$")



# Loads Description of Inputs Page
if selectbox1 == "Description of Inputs":
    st.title("Description of Inputs")
    st.write("The three most influential parameters in determining the drift capacity were selected as inputs to develop the predictive machine learning model. These same three parameters also play a role as particular variables within the empirical model developed by Abdullah and Wallace (2019).")
    selectbox3 = st.radio("Select",("Slenderness Parameter", "Shear Stress Demand", "Configuration of the Boundary Transverse Reinforcement"))
    
    # Loads Section for Slenderness Parameter
    if selectbox3 == "Slenderness Parameter":
        st.subheader("Slenderness Parameter ($\lambda_{b}$)")
        st.write("A parameter that accounts for the slenderness of the cross section and the compression zone.")
        st.markdown("$\lambda_{b}=\dfrac{l_{w}c}{b^2}$")
        st.write("where $l_{w}$ is the wall length, $c$ is the depth of neutral axis computed at a concrete compressive strain of 0.003, and $b$ is the width of flexural compression zone.")
        image2 = Image.open('wall.png')
        st.image(image2)
    
    # Loads Section for Shear Stress Demand
    if selectbox3 == "Shear Stress Demand":
        st.subheader("Shear Stress Demand ($\dfrac{v_{max}}{\sqrt{f'_{c}(psi)}}$)")
        st.write("The maximum experimental shear stress normalized by the square root of the concrete compressive strength.")
    
    # Loads Section for Configuration of Boundary Transverse Reinforcement
    if selectbox3 == "Configuration of the Boundary Transverse Reinforcement":
        st.subheader("Configuration of Boundary Transverse Reinforcement")
        st.write("1. Overlapping Hoops")
        image3 = Image.open('OH.png')
        st.image(image3)
        st.write("2. Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks")
        image4 = Image.open('PH-90-135.png')
        st.image(image4) 
        st.write("3. Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks")
        image5 = Image.open('PH-135-135.png')
        st.image(image5)
        st.write("4. Combination of a Perimeter Hoop and Crossties with Headed Bars")
        image6 = Image.open('PH-HB.png')
        st.image(image6)
        st.write("5. Single Hoop without Intermediate Legs of Crossties")
        image7 = Image.open('SH.png')
        st.image(image7)
        st.markdown("Note: For the empirical model, α=60 for overlapping hoops and α=45 for a combination of a single perimeter hoop with supplemental crossties (which are the last four cases shown above).")

    


# Loads Comparison Page
if selectbox1 == "Model Use & Performance":
    st.title('Model Use & Performance')
    st.markdown("The drift capacity predicted by both the empirical model developed by Abdullah and Wallace (2019) and the machine learning model this web app seeks to highlight can be determined below.")
    st.warning('Please input your desired parameters.')
    
    # Loads Section for Slenderness Parameter
    st.markdown("1. Slenderness Parameter ($\lambda_{b}=\dfrac{l_{w}c}{b^2}$)")
    length = st.number_input("l_w (in)", min_value = 0.000000000, step = 0.000000001, format="%.9f")
    depth = st.number_input("c (in)", min_value = 0.000000000, step = 0.000000001, format="%.9f")
    width = st.number_input("b (in)", min_value = 0.000000000, step = 0.000000001, format="%.9f")

    if length == 0 or depth == 0 or width == 0:
        st.warning('Wall dimensions have to be non-zero')
        slender_parameter = 0
    else:
        slender_parameter = (length*depth)/(width**2)    
    df1 = pd.DataFrame(columns=['Slenderness Parameter'])
    df1['Slenderness Parameter'] = [slender_parameter]
    st.write(df1)
    
    # Loads Section for Shear Stress Demand 
    st.markdown("2. Shear Stress Demand ($\dfrac{v_{max}}{\sqrt{f'_{c}(psi)}}$)")
    shear_stress = st.number_input("enter value", min_value = 0.000000000, step = 0.000000001, format="%.9f")
    
    # Loads Section for Configuration of Boundary Transverse Reinforcement 
    st.markdown("3. Configuration of Boundary Transverse Reinforcement")
    selectbox = st.selectbox("select", ("Overlapping Hoops",
    "Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks",
    "Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks",
    "Combination of a Perimeter Hoop and Crossties with Headed Bars",
    "Single Hoop without Intermediate Legs of Crossties"))
    if selectbox == "Overlapping Hoops":
        alpha = 60
        config = 2.6482
    if selectbox == "Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks":
        alpha = 45
        config = 2.8069
    if selectbox == "Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks":
        alpha = 45
        config = 2.7916
    if selectbox == "Combination of a Perimeter Hoop and Crossties with Headed Bars":
        alpha = 45
        config = 3.7967
    if selectbox == "Single Hoop without Intermediate Legs of Crossties":
        alpha = 45
        config = 2.8750

    # Provides Input Parameters into Empirical Model
    model = 3.85 - slender_parameter/alpha - 0.1*shear_stress
 
    # Provides Input Parameters into ML Model
    input = np.array([slender_parameter,shear_stress,config])
    input = np.array(input).reshape((1,-1))
    predicted = xgb_model.predict(input)
    predicted = predicted[0]
 
    # Display Results of Both Models
    st.write('')
    st.write('')
    st.write('')
    with st.container():
        st.warning('The drift capacity predictions can be found below.')
        data = [model, predicted]
        df2 = pd.DataFrame(data, index =['Drift Capacity from Empirical Model(%)', 'Drift Capacity from XGBoost Model(%)'], columns=['Output'])
        st.table(df2)

    # Plot histograms for input parameters & drift capacity perimeter
    st.write('')
    st.write('')
    with st.container():
        if st.button('Load Histograms for Comparison of Inputs to Test Data'):
            testData = load_excel('SBEDataset.xlsx')
            lambdaHistogram = alt.Chart(testData).mark_bar().encode(alt.X('lambda_b', bin=True, title = 'Histogram of Test Data With Inputted Slenderness Parameter Plotted'),y='count()')
            lambdaLine = alt.Chart(pd.DataFrame({'Input': [slender_parameter], 'color': ['red']})).mark_rule().encode(x='Input', color=alt.Color('color:N', scale=None))
            shearHistogram = alt.Chart(testData).mark_bar().encode(alt.X('lambda_b', bin=True, title = 'Histogram of Test Data With Inputted Shear Stress Demand Parameter Plotted'),y='count()')
            shearLine = alt.Chart(pd.DataFrame({'Input': [shear_stress], 'color': ['red']})).mark_rule().encode(x='Input', color=alt.Color('color:N', scale=None))
            lambdaFig = lambdaHistogram + lambdaLine
            shearFig = shearHistogram + shearLine
            st.altair_chart(lambdaFig, use_container_width=True)
            st.altair_chart(shearFig, use_container_width=True)

        if st.button('Load Histograms for Comparison of Results to Test Data'):
            testData = load_excel('SBEDataset.xlsx')
            driftHistogram = alt.Chart(testData).mark_bar().encode(alt.X('drift', bin=True, title = 'Histogram of Test Data With Prediction Plotted'),y='count()')
            verticalLine1 = alt.Chart(pd.DataFrame({'Predicted Value': [model],'color': ['red']}), title = 'Empirical Model Prediction').mark_rule().encode(x='Predicted Value', color=alt.Color('color:N', scale=None))
            verticalLine2 = alt.Chart(pd.DataFrame({'Predicted Value': [predicted],'color': ['red']}), title = 'ML Model Prediction').mark_rule().encode(x='Predicted Value', color=alt.Color('color:N', scale=None))
            driftFig1 = driftHistogram + verticalLine1
            driftFig2 = driftHistogram + verticalLine2
            st.altair_chart(driftFig1, use_container_width=True)
            st.altair_chart(driftFig2, use_container_width=True)