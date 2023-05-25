import pandas as pd
import streamlit as st
from sklearn import preprocessing
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import plotly.express as px

PAGE_CONFIG = {"page_title": "Predict Your Weekly Weight",
               "page_icon": "chart_with_upwards_trend:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpapers.com/images/featured/o7xpod19e080jvm0.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def showGraphList():
    graph = ["Prediction","Slider"]
    opt = st.radio("Prediction", graph)
    return opt

def sidebar():
    global df1, filename, option, opt, columnList
    df1 = None
    allowedExtension = ['csv', 'xlsx']
    with st.sidebar:
        uploaded_file = st.sidebar.file_uploader(
            label="Upload your data", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            filename = uploaded_file.name
            extension = filename[filename.index(".")+1:]
            filename = filename[:filename.index(".")]

            if extension in allowedExtension:
                df1 = pd.read_csv(uploaded_file)
                columnList = df1.columns.values.tolist()
                # option = st.selectbox("Select Column", columnList)
                # st.subheader("Filters")
                opt = showGraphList()
            else:
                st.write("File Format is not supported")

def mainContent():
    if df1 is not None:
        df_o = df1
        df_o = df_o.drop(['Actual User ID'], axis = 1)
        bool_cols = [col for col in df_o.columns if df_o[col].dtype == 'bool' or df_o[col].dtype =='object']
        label_encoder = preprocessing.LabelEncoder()
        for i in bool_cols[1:]:
            df_o[i] = label_encoder.fit_transform(df_o[i])
        df_o['Weekend date'] = pd.to_datetime(df_o['Weekend date'])
        df_o['Weekend date'] = df_o['Weekend date'].dt.date
        df_o = df_o.drop(['Weekend date', 'ID'], axis=1)
        df_o = df_o[['Height', 'Gender', 'Age', 'Regular cycle', 'Irregular cycle',
       'No cycle', 'On birth control', 'Menopause', 'Pcos', 'Endo',
       'Calorie target', 'Protein target', 'Carbs target', 'Fat target',
       'Avg water intake', 'Avg calorie', 'Avg protein', 'Avg fat', 'Avg carb',
       'Avg weight', 'Avg steps', 'Calorie accuracy', 'Protein accuracy',
       'Fat accuracy', 'Carb accuracy', 'Total calories', 'Guessed calories',
       'Guessed tracked %', 'Average maintenance calories (for the week)',
       'Total maintenance calories for week', 'Total calorie deficit per week',
       'Estimated fat loss per week', 'Sleep hours', 'Quality of sleep',
       'Stress level', 'How do you feel physically',
       'How do you feel emotionally', 'Total training days',
       'phone before bed totals per week',
       'Phone before bed? (as a % of total days so 7 days = 100%)',
       'Fibre average', 'Menstrual cycle day',
       'Users all menstrual cycles type']]
        mean_weight_all = df_o['Avg weight'].mean(skipna=True)
        df_o['Avg weight'].fillna(mean_weight_all, inplace=True)
        corr = df_o.drop(['Height', 'Gender','Age','Regular cycle', 'No cycle', 'Calorie target', 'Carb accuracy', 'On birth control', 'Protein target', 'Carbs target', 'Fat target','Calorie accuracy', 'Average maintenance calories (for the week)', 'Total maintenance calories for week', 'Estimated fat loss per week', 'Users all menstrual cycles type'],axis = 1).corr(method='pearson')
        corr['index'] = corr.index
        df = df_o[["Avg weight"]]
        model = ARIMA(df['Avg weight'],order = (2,1,2))
        model_fit = model.fit()
        data = pd.DataFrame()
        data["weight"] = model_fit.predict(start=51,end=58,dynamic=True)
        # df['date'] = df.index
        # a = pd.DataFrame({'date': pd.date_range(start=df.date.iloc[-1], periods=8, freq='d', closed='right')})
        d_final = pd.DataFrame()
        d_final['Week'] = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8']
        # d_final['Date'] = pd.to_datetime(d_final['Date']).dt.date
        col_list = data.weight.values.tolist()
        d_final['Estimated Weight'] = col_list
        # st.write(d_final)
        df_graph = df_o[0:4] # for line chart

        if opt == "Prediction":
            corr_coeffs = corr.corr()['Avg weight']
            st.header("Weight Trend")
            st.write(d_final)
            st.header("Top 5 features contributing to your weight are:")
            sorted_coeffs = corr_coeffs.abs().sort_values(ascending=False)
            top_5_features = sorted_coeffs.index[1:6]
            for i in top_5_features:
                st.write(i)
            # st.header("Relationship between Weight and Menstrual Cycle Day")
            fig = px.line(df_graph, y = "Menstrual cycle day", x = "Avg weight")
            # st.plotly_chart(fig)
            
        elif opt =="Slider":
            corr_coeffs = corr.corr()['Avg weight']
            corr_coeffs = corr_coeffs.to_frame()
#             st.write(corr_coeffs)
            avg_sleep  = df_o["Sleep hours"].mean()
            avg_sleep = round(avg_sleep)
            
            avg_calorie = df_o["Avg calorie"].mean()
            avg_calorie = round(avg_calorie)
            avg_steps = df_o["Avg steps"].mean()
            avg_steps = round(avg_steps)
            avg_stress = df_o["Stress level"].mean()
            avg_stress = round(avg_stress)
            avg_weight = df_o["Avg weight"].mean()
            
            stress_cor = corr_coeffs.at["Stress level", "Avg weight"]
            step_cor = corr_coeffs.at["Avg steps", "Avg weight"]
            sleep_cor = corr_coeffs.at["Sleep hours", "Avg weight"]
            calorie_cor = corr_coeffs.at["Avg calorie", "Avg weight"]


            stress = st.slider('Stress Level', 0, 10, avg_stress)
            w1 = avg_weight+((stress_cor+(stress*0.2)))
            st.write("Your Average Stress Level : ", avg_stress)
            st.write("Your Average Weight : ", avg_weight)
            st.write("New Weight : ", avg_weight if stress == avg_stress else w1)

            a1  = round(avg_steps - (0.25*avg_steps))
            a2 = round(avg_steps + (0.25*avg_steps))
            steps = st.slider('Steps', a1, a2, avg_steps)

            if(step_cor>0):
                step_cor = -(step_cor)

            if steps<avg_steps:
                w2 = avg_weight-(steps*(-step_cor)*0.001)+4
            else:
                w2 = avg_weight+(steps*step_cor*0.001)

            st.write("Your Average Steps : ", avg_steps)
            st.write("Your Average Weight : ", avg_weight)
            st.write("New Weight : ", avg_weight if steps == avg_steps else w2)
            sleep = st.slider('Sleep', 0, 15, avg_sleep)
            if(sleep_cor<0):
                sleep_cor = -(sleep_cor)

            if sleep<avg_sleep:
                w3 = avg_weight-(sleep*(-sleep_cor)*0.01)-1
            else:
                w3 = avg_weight+(sleep*sleep_cor*0.01)

            st.write("Your Average Sleep : ", avg_sleep)
            st.write("Your Average Weight : ", avg_weight)
            st.write("New Weight : ", avg_weight if sleep == avg_sleep else w3)
            b1 = round(avg_calorie - (0.25*avg_calorie))
            b2 = round(avg_calorie + (0.25*avg_calorie))
            calorie = st.slider('Calorie', b1,b2, avg_calorie)
            if(calorie_cor<0):
                calorie_cor = -(calorie_cor)

            if calorie<avg_calorie:
                w4 = avg_weight-(calorie*(-calorie_cor)*0.001)-1
            else:
                w4 = avg_weight+(calorie*calorie_cor*0.001)

            st.write("Your Average Calorie Intake : ", avg_calorie)
            st.write("Your Average Weight : ", avg_weight)
            st.write("New Weight : ", avg_weight if calorie == avg_calorie else w4)
  
        else:
            st.write("There is nothing to show!! Please add file to see data.")

if __name__ == "__main__":
    
    footer = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 12px;'>
        <p>Made with ❤️ by Cognozire</p>
    </div>
    """
    # background-color: #f5f5f5;
    st.markdown(footer, unsafe_allow_html=True)
    sidebar()
    st.markdown("<h1 style='text-align: center; color: black;'>Women Weekly Health Check</h1>", unsafe_allow_html=True)
    mainContent()
