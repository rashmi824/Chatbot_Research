My development environment is Windows 11 and I solved it with the following commands
pip install langchain==0.1.6
pip uninstall langchain-community
pip install langchain-community==0.0.19
pip install pypdf



RuntimeError: no validator found for <class 're.Pattern'>, see `arbitrary_types_allowed` in Config
Changing the version to pydantic==1.10.8 worked for me using a flask api in pythonanywhere



 streamlit run streamlit.py