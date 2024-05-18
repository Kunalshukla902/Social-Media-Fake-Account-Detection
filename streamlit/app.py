import streamlit as st

from streamlit_option_menu import option_menu


import home, Insights, about, ann_predictor_csv, decision_tree_predictor_csv, random_forest_predictor_csv
st.set_page_config(
        page_title="Fake Social Media Detection",
        
)



class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Fake Social Media Detection ',
                options=['Home','ANN Predictior', 'Decision Tree Predictor' ,'Random Forest Predictor' , 'Insights' ,'about'],
                icons=['house-fill','person-circle','person-circle' ,'person-circle' ,'chat-fill','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == "Home":
            home.app()
        if app == "ANN Predictior":
            ann_predictor_csv.app()    
        if app == "Decision Tree Predictor":
            decision_tree_predictor_csv.app()
        if app == "Random Forest Predictor":
            random_forest_predictor_csv.app()
        if app == "Insights":
            Insights.app()        
        if app == 'about':
            about.app()    
             
          
             
    run()         
    
#'trophy-fill'
