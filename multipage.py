"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
#rom turtle import title
import streamlit as st

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
        self.page = {}
    
    def current_page(self, title, func) -> None:
        self.page = {"title": title, "function": func}
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        self.pages.append({
          
                "title": title, 
                "function": func
            })

    def run(self):
        
        no_buttons = len(self.pages)

        button, *buttons = st.columns(no_buttons)

        buttons.insert(0,button)

        

        for i, page in enumerate(self.pages):
            #if i==0:
                #continue
            with buttons[i]:
                if st.button(page["title"]):
                    self.page = page
                    st.session_state.current_page = page["title"]

        
        
        self.page["function"]()