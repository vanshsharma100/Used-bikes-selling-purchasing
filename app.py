import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import streamlit as st
import base64
import os 

from sklearn.model_selection  import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge,ElasticNet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance



data=pd.read_csv('finaa.csv')

x=data[['owner','location','cc','mileage_','power_','bike_age','km_per_year']]
y = np.log1p(data['price'])

# num_features = x.select_dtypes(include=['int64','float64']).columns
# cat_features = x.select_dtypes(include=['object']).columns

# prep = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), num_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
#     ]
# )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model=joblib.load('bike_final.joblib')

y_pred=model.predict(x_test)
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)



csv_file="new_data.csv"





def purchasing(brand,model_year,km_driven,ownership,location,cc,mileage,power_bhp,bike_age,km_per_year):
    inp=pd.DataFrame([[ownership,location,cc,mileage,power_bhp,bike_age,km_per_year]],columns=['owner','location','cc','mileage_','power_','bike_age','km_per_year'])
    out=np.expm1(model.predict(inp))
    st.write('Your bike price is: ₹',round(out[0],2))

def selling(brand,model_year,km_driven,ownership,location,cc,mileage,power_bhp,bike_age,km_per_year):
    inpp=pd.DataFrame([[ownership,location,cc,mileage,power_bhp,bike_age,km_per_year]],columns=['owner','location','cc','mileage_','power_','bike_age','km_per_year'])
    outt=np.expm1(model.predict(inpp))[0]
    st.write('Approximate selling price of your bike: ₹',round(outt,2))
    new_data = pd.DataFrame([{
        "brand": brand,
        "model_year": model_year,
        "km_driven": km_driven,
        "owner": ownership,
        "location": location,
        "cc": cc,
        "mileage_": mileage,
        "power_": power_bhp,
        "price": int(round(outt))
    }])

    if os.path.exists(csv_file):
        new_data.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        new_data.to_csv(csv_file, index=False)

    st.success("✅ New data added to CSV successfully")

    






def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="800px"
        type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)

def imagee():
    from PIL import Image

    img = Image.open("modell.png")
    st.image(img, caption="Model", use_container_width=True)






st.set_page_config(page_title="Used bike", layout="wide")
st.title("Used bike selling/purchasing")
st.sidebar.header("Menu")
menu_option = st.sidebar.radio("Choose action", (
    "selling/purchasing",
    "visualization",
    "new data"))

if menu_option=="selling/purchasing":
    st.header("selling/purchasing")
    tab1,tab2=st.tabs(["purchasing","selling"])
    with tab1:
        st.subheader("Tell Us Your Bike Requirements")
        with st.form("add_form"):
            brand= st.selectbox("Which bike brand do you prefer?",['bajaj','ktm','royal enfield','hero','honda','tvs','yamaha','suzuki','others'])
            model_year=st.number_input("which model year you prefer between 1990-2020  ? ")
            km_driven=st.number_input("Enter how many kilometers the bike has been driven")
            ownership=st.selectbox("ownership",['first owner','second owner'])
            location=st.selectbox("enter your location",['hyderabad', 'bangalore', 'chennai', 'delhi', 'mumbai','gurgaon', 'pune', 'ahmedabad', 'jaipur', 'faridabad','Other'])
            cc=st.number_input('Select Engine Capacity (CC)')
            mileage=st.number_input('Enter your minimum mileage requirement (km/l)')
            power_bhp=st.number_input('enter your power(bhp) requirnment ')
            bike_age=2025-model_year
            km_per_year=km_driven//bike_age
            button=st.form_submit_button("check price")
            if button:
                purchasing(brand,model_year,km_driven,ownership,location,cc,mileage,power_bhp,bike_age,km_per_year)
    with tab2:
        st.subheader("Tell Us Your Bike Performance")
        with st.form("add_form1"):
            brand= st.selectbox("What is the brand of your bike?",['bajaj','ktm','royal enfield','hero','honda','tvs','yamaha','suzuki','others'])
            model_year=st.number_input("In which year was your bike manufactured? (1990–2020)")
            km_driven=st.number_input("How many kilometers has your bike been driven?")
            ownership=st.selectbox("What is the ownership status of your bike?",['first owner','second owner'])
            location=st.selectbox("In which city is your bike currently located?",['hyderabad', 'bangalore', 'chennai', 'delhi', 'mumbai','gurgaon', 'pune', 'ahmedabad', 'jaipur', 'faridabad','Other'])
            cc=st.number_input('What is the engine capacity (CC) of your bike?')
            mileage=st.number_input('What mileage does your bike give? (km/l)')
            power_bhp=st.number_input('What is the engine power of your bike? (in BHP)')
            bike_age=2025-model_year
            km_per_year=km_driven//bike_age
            button=st.form_submit_button("check approximate price")
            if button:
                selling(brand,model_year,km_driven,ownership,location,cc,mileage,power_bhp,bike_age,km_per_year)

elif menu_option == "visualization":
    st.header("Visualizations")
     

    tab1, tab2, tab3 = st.tabs(
        ['Visualization report', 'My Model', 'Model Graphs']
    )


    with tab1:
        st.subheader("Report of your model")
        show_pdf("visualization.pdf")

    with tab2:
        st.subheader("Model visualization")
        imagee()

    with tab3:
        st.subheader("Feature Graphs")
        grph=st.radio("select graph",("Feature importance","Correlation Heatmap","Actual vs Predicted","Residual plot"),horizontal=True)
        if grph=="Feature importance":
            st.markdown("Feature Importance --> Highlights which input features contribute most to the model’s predictions, helping explain feature influence on price estimation")
            perm=permutation_importance(model,x_test,y_test,n_repeats=10,random_state=42,scoring="r2")
            feature=model.named_steps['prep'].get_feature_names_out()
            perm_df=pd.DataFrame({"importance":perm.importances_mean,"feature":x_test.columns}).sort_values(by='importance',ascending=False)
            fig,ax=plt.subplots(figsize=(8,5))
            sns.barplot(x="importance",y="feature",data=perm_df,ax=ax)
            st.pyplot(fig)
            plt.close()
          
        elif grph=="Correlation Heatmap":
            st.markdown("Feature Correlation --> Visualizes the strength and direction of relationships between numerical features to identify multicollinearity and key drivers. ")
            corr=data.select_dtypes(include=np.number).corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)
            st.pyplot(fig)
        elif grph=="Actual vs Predicted":
            st.markdown("Actual value vs predicted --> Compares predicted values against actual values to evaluate overall model accuracy and identify systematic deviations.")
            fig, ax = plt.subplots()
            ax.scatter(y_test,y_pred,alpha=0.5)
            ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],"r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
        elif grph=="Residual plot":
            st.markdown("Residual Distribution --> Shows how prediction errors are distributed, helping detect bias, variance, and overall model reliability.")
            res=  y_test-y_pred
            fig,ax=plt.subplots()
            sns.histplot(res,kde=True,ax=ax)
            st.pyplot(fig)


        
elif menu_option=="new data":
    st.header("New available bikes data")
    if os.path.exists(csv_file):
        dff=pd.read_csv(csv_file)
        st.dataframe(dff,use_container_width=True)
        buttonn=st.download_button(label="⬇️ Download CSV",data=dff.to_csv(index=False),file_name="new_data.csv",mime="text/csv")
        if buttonn:
            st.success("File downloaded sucessfully !")
    else:
        st.warning("no data file found")
    

    



