import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.agent import RunOutput
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_env()

if "GOOGLE_API_KEY" not in st.session_state or not st.session_state.GOOGLE_API_KEY:
    st.session_state.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

with st.sidebar:
    #st.title("ℹ️ Configuration")
    
    if not st.session_state.GOOGLE_API_KEY:
        # If no key, try to use the hardcoded one or show input (currently input is commented out)
        if "GOOGLE_API_KEY" not in st.session_state or not st.session_state.GOOGLE_API_KEY:
             # Fallback to hardcoded key if available/intended, or just pass
             pass
    else:
        st.success("API Key is configured")
        # if st.button("🔄 Reset API Key"):
        #     st.session_state.GOOGLE_API_KEY = None
        #     st.rerun()
    
    st.info(
        "Este agente proporciona un análisis de imágenes médicas utilizando "
        "IA y radiología."
    )
    st.warning(
        "⚠DISCLAIMER: Este agente es para fines educativos y informativos solo. "
        "Todos los análisis deben ser revisados por profesionales de la salud calificados. "
        "No se debe tomar decisiones médicas solo basadas en este análisis."
    )

medical_agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools(verify_ssl=False, fixed_max_results=3, timeout=10)],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

#if not medical_agent:
#    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows, results should be in spanish:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

st.title("🩺 Agente de diagnóstico por imágenes médicas")
st.write("Sube una imagen médica para su análisis profesional.")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: navy;
        color: white;
        border: 2px solid navy;
    }
    div.stButton > button:hover {
        background-color: #1976D2;
        color: white;
        border: 2px solid #1976D2;
    }
    </style>
    """, unsafe_allow_html=True)

# Crear contenedores para una mejor organización
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    st.subheader("Subir Imagen")
    uploaded_file = st.file_uploader(
        "Seleccione una imagen médica",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Formatos soportados: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Subida de imagen médica",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🧐 Analizar imagen",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            with st.spinner("🧐 Analizando imagen... Por favor, espere."):
                try:
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)
                    
                    # Create AgnoImage object
                    agno_image = AgnoImage(filepath=temp_path)
                    
                    # Run analysis
                    response: RunOutput = medical_agent.run(query, images=[agno_image])
                    st.markdown("### 📃 Resultados del análisis")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Nota: Este análisis se genera por IA y debe ser revisado por "
                        "un profesional de la salud calificado."
                    )
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg:
                        st.error("⚠️ Error 429: Demasiadas solicitudes. Por favor, espere un momento antes de intentar de nuevo o verifique su cuota de API.")
                    else:
                        st.error(f"Error al analizar la imagen: {e}")
else:
    st.info("☝🏻 Por favor, sube una imagen médica para comenzar el análisis")
