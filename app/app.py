import streamlit as st

def app() -> None:
    sidebar()
    
    st.title('Brain tumor detector 🧠🩺')

    file_uploaded = st.file_uploader('File uploader', type=['jpg'])

    if file_uploaded is not None:
        st.image(file_uploaded, caption='Uploaded Image.')

def sidebar() -> None:
    st.sidebar.title('📔 Info')

if __name__=='__main__':
    app()