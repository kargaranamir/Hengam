import streamlit as st
import torch
from utils import *

torch.hub.download_url_to_file('https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransW.pth',
                               'HengamTransW.pth')
torch.hub.download_url_to_file('https://huggingface.co/kargaranamir/Hengam/resolve/main/HengamTransA.pth',
                               'HengamTransA.pth')

# APP
st.set_page_config(
    page_title="Hengam",
    page_icon="🕒",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("🕒 Hengam")
    st.header("")

with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   Online Demo for Hengam: An Adversarially Trained Transformer for Persian Temporal Tagging [Code](https://github.com/kargaranamir/hengam)!
-   This paper introduces Hengam, an adversarially trained transformer for Persian temporal tagging outperforming state-of-the-art approaches on a diverse and manually created dataset.
	    """
    )
    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste any persian (farsi) text you want to extract its temporal markers.**")
with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.05, 1.5, 0.05, 4, 0.05])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["HengamTransW.pth", "HengamTransA.pth"],
            help="At present, you can choose between 2 models (HengamTransW or HengamTransA) to exrtact temporal markers. More to come!",
        )
        ner = NER(model_path='.', model_name=ModelType, tags=['B-TIM', 'I-TIM', 'B-DAT', 'I-DAT', 'O'])

    with c2:
        doc = st.text_area(
            "Paste your text below",
            '''ساعت ۸ صبح من و علی قرار گذاشتیم که شانزده بهمن ۱۳۷۵ هم دیگر را در دوشنبه بازار ببینیم.''',
            height=110,
        )

        submit_button = st.form_submit_button(label="✨ Extract Temporal Markers!")

if not submit_button:
    st.stop()

result = "\n".join([str(t) for t in ner(doc)])

st.markdown("## **🎈Check results**")

c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    st.write(result)
