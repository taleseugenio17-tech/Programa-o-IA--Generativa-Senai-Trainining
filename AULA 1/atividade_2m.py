import streamlit as st
st.header('Data analysis')
st.subheader('Fill in the blanks ')


st.text_input('Type your name: ')
st.text_input('How tall are you?: ')
st.text_input('How old are you: ')


if st.button ('Submit data'):
    st.subheader('Fill in the blanks')
  



